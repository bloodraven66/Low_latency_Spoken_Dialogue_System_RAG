import argparse
import asyncio
from collections import deque
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import sphn

from unmute.kyutai_constants import SAMPLE_RATE, SAMPLES_PER_FRAME
from unmute.unmute_handler import UnmuteHandler
from unmute.unmute_handler_rag import collapse_rag_results_compact as collapse_rag_results
import unmute.openai_realtime_api_events as ora


class RagRetriever:
    def __init__(
        self,
        rag_url: str,
        top_k: int,
        timeout_sec: float,
        max_retries: int = 1,
        retry_backoff_sec: float = 0.2,
    ):
        self.rag_url = rag_url.rstrip("/")
        self.top_k = top_k
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec

    async def retrieve(self, query: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "top_k": self.top_k,
        }

        url = f"{self.rag_url}/api/rag/retrieve"

        def _call_once() -> dict[str, Any]:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read().decode("utf-8")
            decoded = json.loads(body)
            assert isinstance(decoded, dict)
            return decoded

        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.to_thread(_call_once)
            except (urllib.error.URLError, TimeoutError):
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(self.retry_backoff_sec * (attempt + 1))

        raise RuntimeError("Unexpected RAG retry loop fallthrough")


RAG_CONTEXT_BLOCK_RE = re.compile(
    r"\n?\[RAG_CONTEXT\].*?\[/RAG_CONTEXT\]\s*",
    flags=re.DOTALL,
)


def strip_rag_context(text: str) -> str:
    return RAG_CONTEXT_BLOCK_RE.sub("", text).strip()



def get_last_nonempty_user_index(chat_history: list[dict[str, Any]]) -> int | None:
    for idx in range(len(chat_history) - 1, -1, -1):
        row = chat_history[idx]
        if row.get("role") == "user" and str(row.get("content", "")).strip() != "":
            return idx
    return None


async def main(
    input_path: Path,
    output_path: Path,
    voice: str,
    rag_url: str,
    rag_top_k: int,
    rag_timeout_sec: float,
):
    print(f"Loading input audio from {input_path}")
    data, _sr = sphn.read(input_path, sample_rate=SAMPLE_RATE)
    if data.ndim > 1:
        data = data.mean(axis=0)  # Convert to mono if needed

    retriever = RagRetriever(
        rag_url=rag_url,
        top_k=rag_top_k,
        timeout_sec=rag_timeout_sec,
        max_retries=1,
        retry_backoff_sec=0.25,
    )

    # The UnmuteHandler processes logic natively
    handler = UnmuteHandler()

    from unmute.llm.system_prompt import RagInstructions

    print("Configuring session for evaluation (RAG mode)...")
    await handler.update_session(
        ora.SessionConfig(
            instructions=RagInstructions(),
            voice=voice,
            allow_recording=False,
        )
    )

    async with handler:
        print("Starting UnmuteHandler...")
        await handler.start_up()

        feed_done = asyncio.Event()

        timed_chunks = []
        transcription_log = []
        response_log = []
        cumulative_transcript = ""
        timing_trace: dict[str, Any] = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "generated_at_unix_sec": time.time(),
            "rag": {
                "url": rag_url,
                "top_k": rag_top_k,
                "mode": "generation_gated_before_llm_start",
            },
            "vad_pause_detection_times_sec": [],
            "rag_calls": [],
            "generations": [],
        }

        def ensure_generation(index_hint: int | None = None) -> tuple[int, dict[str, Any]]:
            generations = timing_trace["generations"]
            assert isinstance(generations, list)
            if index_hint is not None and 0 <= index_hint < len(generations):
                gen = generations[index_hint]
                assert isinstance(gen, dict)
                return index_hint, gen
            gen = {
                "generation_index": len(generations),
                "vad_pause_detected_sec": None,
                "start_trigger": "vad_pause",
                "status": "pending",
                "status_reason": None,
                "rag_started_sec": None,
                "rag_finished_sec": None,
                "rag_input_query": None,
                "rag_top_k_output": [],
                "rag_context_collapsed": None,
                "rag_error": None,
                "rag_latency_ms": None,
                "rag_applied_to_prompt": False,
                "rag_not_applied_reason": None,
                "llm_generation_started_sec": None,
                "llm_first_token_sec": None,
                "tts_generation_started_sec": None,
                "tts_first_token_received_sec": None,
                "tts_first_audio_chunk_received_sec": None,
                "tts_first_signal_source": None,
                "llm_text_generated": "",
                "llm_input_transcript_at_start": None,
            }
            generations.append(gen)
            return len(generations) - 1, gen

        pending_pause_generation_indices: deque[int] = deque()
        original_generate_response = handler._generate_response  # pyright: ignore[reportPrivateUsage]

        async def generate_response_with_rag_gate() -> None:
            gen_idx: int | None = pending_pause_generation_indices.popleft() if pending_pause_generation_indices else None
            gen_idx, gen = ensure_generation(gen_idx)

            query_source = handler.chatbot.last_message("user") or cumulative_transcript
            query = strip_rag_context(query_source)
            rag_started = handler.audio_received_sec()
            rag_started_wall = time.perf_counter()

            gen["rag_started_sec"] = rag_started
            gen["rag_input_query"] = query

            rag_call: dict[str, Any] = {
                "generation_index": gen["generation_index"],
                "rag_started_sec": rag_started,
                "rag_input_query": query,
                "rag_top_k": rag_top_k,
                "rag_finished_sec": None,
                "rag_top_k_output": [],
                "rag_context_collapsed": None,
                "rag_error": None,
            }

            if not query.strip():
                rag_finished = handler.audio_received_sec()
                rag_finished_wall = time.perf_counter()
                rag_latency_ms = (rag_finished_wall - rag_started_wall) * 1000.0

                gen["rag_finished_sec"] = rag_finished
                gen["rag_latency_ms"] = rag_latency_ms
                gen["rag_not_applied_reason"] = "empty_query"

                rag_call["rag_finished_sec"] = rag_finished
                rag_call["rag_latency_ms"] = rag_latency_ms
                rag_call["rag_not_applied_reason"] = "empty_query"

                rag_calls = timing_trace["rag_calls"]
                assert isinstance(rag_calls, list)
                rag_calls.append(rag_call)

                await original_generate_response()
                return

            latest_user_before = strip_rag_context(handler.chatbot.last_message("user") or "")

            try:
                rag_response = await retriever.retrieve(query)
                results_raw = rag_response.get("results", [])
                results = (
                    [x for x in results_raw if isinstance(x, dict)]
                    if isinstance(results_raw, list)
                    else []
                )
                topk_output = [
                    {
                        "rank": row.get("rank"),
                        "score": row.get("score"),
                        "chunk_id": row.get("chunk_id"),
                        "source_path": row.get("source_path"),
                        "preview": row.get("preview"),
                    }
                    for row in results[:rag_top_k]
                ]
                collapsed = collapse_rag_results(results[:rag_top_k])
                rag_finished = handler.audio_received_sec()
                rag_finished_wall = time.perf_counter()
                rag_latency_ms = (rag_finished_wall - rag_started_wall) * 1000.0

                gen["rag_finished_sec"] = rag_finished
                gen["rag_top_k_output"] = topk_output
                gen["rag_context_collapsed"] = collapsed
                gen["rag_latency_ms"] = rag_latency_ms

                rag_call["rag_finished_sec"] = rag_finished
                rag_call["rag_top_k_output"] = topk_output
                rag_call["rag_context_collapsed"] = collapsed
                rag_call["rag_latency_ms"] = rag_latency_ms

                latest_user_after = strip_rag_context(handler.chatbot.last_message("user") or "")
                if latest_user_after != latest_user_before:
                    gen["rag_not_applied_reason"] = "user_resumed_or_changed_before_rag_ready"
                    rag_call["rag_not_applied_reason"] = gen["rag_not_applied_reason"]
                elif collapsed:
                    idx = get_last_nonempty_user_index(handler.chatbot.chat_history)
                    if idx is not None:
                        current_user_text = str(handler.chatbot.chat_history[idx].get("content", ""))
                        base_user_text = strip_rag_context(current_user_text)
                        handler.chatbot.chat_history[idx]["content"] = (
                            base_user_text.rstrip()
                            + "\n\n[RAG_CONTEXT]\n"
                            + collapsed
                            + "\n[/RAG_CONTEXT]"
                        )
                        gen["rag_applied_to_prompt"] = True
                    else:
                        gen["rag_not_applied_reason"] = "missing_user_message"
                        rag_call["rag_not_applied_reason"] = gen["rag_not_applied_reason"]

                print(f"[{rag_finished:.2f}s] 📚 RAG retrieved {len(topk_output)} chunks")
            except (urllib.error.URLError, TimeoutError, asyncio.TimeoutError, json.JSONDecodeError, ValueError) as exc:
                rag_finished = handler.audio_received_sec()
                rag_finished_wall = time.perf_counter()
                rag_latency_ms = (rag_finished_wall - rag_started_wall) * 1000.0

                gen["rag_finished_sec"] = rag_finished
                gen["rag_error"] = repr(exc)
                gen["rag_latency_ms"] = rag_latency_ms

                rag_call["rag_finished_sec"] = rag_finished
                rag_call["rag_error"] = repr(exc)
                rag_call["rag_latency_ms"] = rag_latency_ms
                rag_call["rag_not_applied_reason"] = "rag_error"
                print(f"[{rag_finished:.2f}s] ⚠️ RAG call failed: {exc!r}")

            rag_calls = timing_trace["rag_calls"]
            assert isinstance(rag_calls, list)
            rag_calls.append(rag_call)

            await original_generate_response()

        handler._generate_response = generate_response_with_rag_gate  # pyright: ignore[reportPrivateUsage]

        async def feed_audio():
            print(f"Feeding audio ({len(data) / SAMPLE_RATE:.2f}s) in driftless simulated real-time...")
            next_feed_time = time.perf_counter()

            for i in range(0, len(data), SAMPLES_PER_FRAME):
                chunk = data[i : i + SAMPLES_PER_FRAME]
                if len(chunk) < SAMPLES_PER_FRAME:
                    chunk = np.pad(chunk, (0, SAMPLES_PER_FRAME - len(chunk)))

                await handler.receive((SAMPLE_RATE, chunk[np.newaxis, :]))

                next_feed_time += SAMPLES_PER_FRAME / SAMPLE_RATE
                sleep_duration = next_feed_time - time.perf_counter()
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)

            print("Finished feeding input audio.")

            for _ in range(0, int(15 * SAMPLE_RATE), SAMPLES_PER_FRAME):
                if feed_done.is_set():
                    break
                chunk = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
                await handler.receive((SAMPLE_RATE, chunk[np.newaxis, :]))

                next_feed_time += SAMPLES_PER_FRAME / SAMPLE_RATE
                sleep_duration = next_feed_time - time.perf_counter()
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)

            if not feed_done.is_set():
                print("Max silence margin reached. Stopping.")

            feed_done.set()

        async def collect_audio():
            nonlocal cumulative_transcript
            print("Collecting server output...")
            ignore_bot_output = True
            has_seen_first_token = False
            input_end_wall_time: float | None = None
            active_generation_index: int | None = None

            def maybe_mark_tts_signal(timestamp: float, signal_source: str) -> None:
                nonlocal active_generation_index
                active_generation_index, gen = ensure_generation(active_generation_index)
                if gen["tts_generation_started_sec"] is None:
                    gen["tts_generation_started_sec"] = timestamp
                    gen["tts_first_signal_source"] = signal_source
                if gen["tts_first_token_received_sec"] is None:
                    gen["tts_first_token_received_sec"] = timestamp

            while True:
                try:
                    output = await asyncio.wait_for(handler.emit(), timeout=0.05)
                except asyncio.TimeoutError:
                    output = None

                if output is None:
                    if feed_done.is_set():
                        if input_end_wall_time is None:
                            input_end_wall_time = time.perf_counter()
                        elif time.perf_counter() - input_end_wall_time > 10.0:
                            print("10s passed since input finished. Stopping collector.")
                            break
                    continue

                if input_end_wall_time is not None:
                    input_end_wall_time = time.perf_counter()

                timestamp = handler.audio_received_sec()

                if isinstance(output, tuple):
                    if not ignore_bot_output:
                        _sr, chunk = output
                        maybe_mark_tts_signal(timestamp, "pcm_tuple")
                        active_generation_index, gen = ensure_generation(active_generation_index)
                        if gen["tts_first_audio_chunk_received_sec"] is None:
                            gen["tts_first_audio_chunk_received_sec"] = timestamp
                        if not timed_chunks:
                            print(f"\n[{timestamp:.2f}s] 🔈 FIRST TTS AUDIO CHUNK RECEIVED!")
                        timed_chunks.append((timestamp, chunk))

                elif isinstance(output, ora.ServerEvent):
                    if output.type == "input_audio_buffer.speech_stopped":
                        ignore_bot_output = False
                        has_seen_first_token = False
                        active_generation_index, gen = ensure_generation()
                        pending_pause_generation_indices.append(active_generation_index)
                        gen["vad_pause_detected_sec"] = timestamp
                        vad_events = timing_trace["vad_pause_detection_times_sec"]
                        assert isinstance(vad_events, list)
                        vad_events.append(timestamp)
                        print(f"\n[{timestamp:.2f}s] 🛑 VAD PAUSE DETECTED (Speech Stopped)")

                    elif output.type == "input_audio_buffer.speech_started":
                        pass
                    elif output.type == "response.created":
                        if not ignore_bot_output:
                            active_generation_index, gen = ensure_generation(active_generation_index)
                            if gen["llm_generation_started_sec"] is None:
                                gen["llm_generation_started_sec"] = timestamp
                                gen["llm_input_transcript_at_start"] = cumulative_transcript
                            gen["status"] = "started"
                            gen["status_reason"] = "response_created"
                            print(f"[{timestamp:.2f}s] 🧠 LLM GENERATION STARTED")
                    elif output.type == "unmute.response.text.delta.ready":
                        if not ignore_bot_output and not has_seen_first_token:
                            active_generation_index, gen = ensure_generation(active_generation_index)
                            if gen["llm_first_token_sec"] is None:
                                gen["llm_first_token_sec"] = timestamp
                            print(f"[{timestamp:.2f}s] ✍️ FIRST TEXT TOKEN FROM vLLM!")
                            has_seen_first_token = True
                        if not ignore_bot_output:
                            active_generation_index, gen = ensure_generation(active_generation_index)
                            delta = output.delta
                            if delta:
                                gen["llm_text_generated"] += delta
                    elif output.type in {
                        "response.audio.delta",
                        "unmute.response.audio.delta.ready",
                    }:
                        if not ignore_bot_output:
                            maybe_mark_tts_signal(timestamp, output.type)
                    elif output.type == "conversation.item.input_audio_transcription.delta":
                        transcription_log.append(output.delta)
                        cumulative_transcript += output.delta
                    elif output.type == "response.text.delta":
                        if not ignore_bot_output:
                            response_log.append(output.delta)
                    elif output.type == "unmute.interrupted_by_vad":
                        if not ignore_bot_output:
                            active_generation_index, gen = ensure_generation(active_generation_index)
                            gen["status"] = "interrupted"
                            gen["status_reason"] = "early_start_from_vad_pause"
                            active_generation_index = None
                    elif output.type == "response.audio.done":
                        if not ignore_bot_output:
                            active_generation_index, gen = ensure_generation(active_generation_index)
                            if gen["status"] not in {"interrupted", "failed"}:
                                gen["status"] = "completed"
                                gen["status_reason"] = "response_audio_done"
                            print(f"\n[{timestamp:.2f}s] ✅ Response audio finished.")
                            active_generation_index = None
                            if gen.get("llm_text_generated", "").strip():
                                feed_done.set()

        vad_times = []
        vad_values = []

        async def track_vad():
            last_t = -99.0
            while not feed_done.is_set():
                if handler.stt is not None:
                    curr_t = handler.stt.current_time
                    if curr_t > last_t and curr_t >= 0.0:
                        sim_time = handler.audio_received_sec()
                        vad_times.append(sim_time)
                        vad_values.append(handler.stt.pause_prediction.value)
                        last_t = curr_t
                await asyncio.sleep(0.01)

        await asyncio.gather(feed_audio(), collect_audio(), track_vad())

        print("\nRendering stereo aligned output...")

        max_time = len(data) / SAMPLE_RATE
        if timed_chunks:
            last_timestamp, last_chunk = timed_chunks[-1]
            max_time = max(max_time, last_timestamp + (len(last_chunk) / SAMPLE_RATE))

        output_samples = int(np.ceil((max_time + 5.0) * SAMPLE_RATE))
        left_channel = np.zeros(output_samples, dtype=np.float32)
        right_channel = np.zeros(output_samples, dtype=np.float32)

        left_channel[: len(data)] = data

        current_cursor = 0.0
        for ts, chunk in timed_chunks:
            if ts - current_cursor > 0.5:
                current_cursor = ts

            start_idx = int(current_cursor * SAMPLE_RATE)
            end_idx = start_idx + len(chunk)

            if end_idx > len(right_channel):
                pad_size = end_idx - len(right_channel) + (SAMPLE_RATE * 5)
                right_channel = np.pad(right_channel, (0, pad_size))
                left_channel = np.pad(left_channel, (0, pad_size))

            right_channel[start_idx:end_idx] += chunk
            current_cursor += len(chunk) / SAMPLE_RATE

        active_indices = np.where(np.abs(right_channel) + np.abs(left_channel) > 0.0)[0]
        if len(active_indices) > 0:
            final_idx = active_indices[-1] + int(1.0 * SAMPLE_RATE)
            left_channel = left_channel[:final_idx]
            right_channel = right_channel[:final_idx]

        right_channel = np.clip(right_channel, -1.0, 1.0)

        stereo = np.stack([left_channel, right_channel], axis=0)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        stereo_path = output_path.with_name("output.stereo.wav")
        user_path = output_path.with_name("user.wav")
        sphn.write_wav(stereo_path, stereo, SAMPLE_RATE)
        sphn.write_wav(user_path, left_channel[np.newaxis, :], SAMPLE_RATE)
        sphn.write_wav(output_path.with_name("output.wav"), right_channel[np.newaxis, :], SAMPLE_RATE)
        print(f"✓ Saved stereo output to {stereo_path}")
        print(f"✓ Saved user stream to {user_path}")
        print(f"✓ Saved bot stream to {output_path.with_name('output.wav')}")

        timings_path = output_path.with_suffix(".timings.json")
        with timings_path.open("w", encoding="utf-8") as f:
            json.dump(timing_trace, f, indent=2)
        print(f"✓ Saved timing trace JSON to {timings_path}")

        try:
            import matplotlib.pyplot as plt
            if vad_times:
                times = vad_times
                vad = vad_values

                time_axis = np.linspace(0, len(left_channel) / SAMPLE_RATE, num=len(left_channel))

                _fig, ax1 = plt.subplots(figsize=(14, 6))

                ax1.plot(time_axis, left_channel, color="blue", alpha=0.4, label="Input Audio (User)")
                ax1.plot(time_axis, right_channel, color="green", alpha=0.4, label="Output Audio (Bot)")
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Audio Amplitude", color="black")
                ax1.tick_params(axis="y", labelcolor="black")
                if len(time_axis) > 0:
                    ax1.set_xlim(left=0.0, right=float(time_axis[-1]))
                else:
                    ax1.set_xlim(left=0.0)

                from matplotlib.ticker import MultipleLocator, FuncFormatter

                ax1.xaxis.set_major_locator(MultipleLocator(1.0))
                ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
                ax1.tick_params(axis="x", which="major", labelsize=10)
                ax1.tick_params(axis="x", which="minor", labelsize=7, labelbottom=True)

                def minor_formatter(x: float, pos: int) -> str:
                    return f"{x:.1f}" if x % 1 != 0 else ""

                ax1.xaxis.set_minor_formatter(FuncFormatter(minor_formatter))

                ax1.grid(which="major", axis="x", linestyle="-", linewidth=1.2, color="gray", alpha=0.6)
                ax1.grid(which="minor", axis="x", linestyle="--", linewidth=0.6, color="gray", alpha=0.4)
                ax1.grid(which="major", axis="y", alpha=0.2)

                ax2 = ax1.twinx()
                ax2.plot(times, vad, color="red", linewidth=2.0, label="VAD Pause Prediction")
                ax2.set_ylabel("VAD Probability", color="red")
                ax2.tick_params(axis="y", labelcolor="red")
                ax2.set_ylim(-0.05, 1.05)

                crossings = [times[i] for i in range(1, len(vad)) if vad[i - 1] < 0.6 and vad[i] >= 0.6]
                for i, cross_t in enumerate(crossings):
                    label = "VAD Trigger (0.6)" if i == 0 else None
                    ax2.axvline(x=cross_t, color="darkred", linestyle="-.", linewidth=1.5, alpha=0.8, label=label)

                lines_1, labels_1 = ax1.get_legend_handles_labels()
                lines_2, labels_2 = ax2.get_legend_handles_labels()
                ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

                plt.title("Audio Activity & VAD Tracking (RAG mode)")
                plt.tight_layout()

                plot_path = output_path.with_suffix(".vad_plot.png")
                plt.savefig(plot_path)
                print(f"✓ Saved 3x1 audio & VAD prediction plot to {plot_path}")
        except ImportError:
            print("⚠ matplotlib is not installed. Skipping VAD plot generation.")

        print("\n===== Evaluation Summary =====")
        print("Transcription computed by Server Engine:")
        print("".join(transcription_log))
        print("\nResponse generated by LLM Engine:")
        print("".join(response_log))
        print("==============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate UnmuteHandler with pause-time RAG retrieval (non-speculative)."
    )
    parser.add_argument("input_path", type=Path, help="Path to input .wav file (will be placed on Left Channel)")
    parser.add_argument("output_path", type=Path, help="Path to output stereo .wav file")
    parser.add_argument(
        "--voice",
        type=str,
        default="unmute-prod-website/ex04_narration_longform_00001.wav",
        help="The TTS voice reference file for cloning",
    )
    parser.add_argument("--rag-url", type=str, default="http://127.0.0.1:8095", help="RAG server URL")
    parser.add_argument("--rag-top-k", type=int, default=1, help="Top-k RAG chunks")
    parser.add_argument("--rag-timeout-sec", type=float, default=1.2, help="RAG request timeout in seconds")
    args = parser.parse_args()

    asyncio.run(
        main(
            args.input_path,
            args.output_path,
            args.voice,
            rag_url=args.rag_url,
            rag_top_k=args.rag_top_k,
            rag_timeout_sec=args.rag_timeout_sec,
        )
    )
