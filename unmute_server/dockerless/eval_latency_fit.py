"""
Latency evaluation for FIT RAG Benchmark results.
Mirrors eval_smooth_turn_taking from Full-Duplex-Bench but adapted for the
FIT benchmark directory structure (no hardcoded data_dir, no turn_taking.json).

input_end_time  = duration of input.wav (the audio fed to the system)
TOR             = 1 if bot produced transcribed output after input_end_time
latency         = first bot word timestamp - input_end_time
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def eval_latency_fit(results_dir: str) -> None:
    root = Path(results_dir)
    item_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    take_turn_list: list[int] = []
    latency_list: list[float] = []

    vad_pause_latency_list: list[float] = []
    llm_generation_latency_list: list[float] = []
    llm_first_token_latency_list: list[float] = []
    tts_generation_latency_list: list[float] = []
    tts_first_token_latency_list: list[float] = []
    tts_first_audio_chunk_latency_list: list[float] = []

    num_speculations: list[int] = []

    for item_dir in tqdm(item_dirs, desc="evaluate"):
        output_json = item_dir / "output.json"
        timings_json = item_dir / "output.timings.json"

        if not output_json.exists() or not timings_json.exists():
            print(f"[skip] missing files in {item_dir.name}")
            continue

        # input_end_time = end of last word in user ASR transcript
        input_json = item_dir / "input.json"
        if not input_json.exists():
            print(f"[skip] missing input.json in {item_dir.name}")
            continue
        with open(input_json) as f:
            input_asr = json.load(f)
        input_chunks = input_asr.get("chunks", [])
        if not input_chunks:
            print(f"[skip] empty input.json in {item_dir.name}")
            continue
        input_end_time = input_chunks[-1]["timestamp"][-1]

        # Load bot ASR transcript
        with open(output_json) as f:
            output_asr = json.load(f)
        chunks = output_asr.get("chunks", [])

        # Load timings
        with open(timings_json) as f:
            timings_data = json.load(f)

        # ── Timing breakdown from timings file ──────────────────────────────
        vad_pause_detected_sec = None
        llm_generation_started_sec = None
        llm_first_token_sec = None
        tts_generation_started_sec = None
        tts_first_token_received_sec = None
        tts_first_audio_chunk_received_sec = None

        for gen in timings_data.get("generations", []):
            if gen.get("status") != "completed":
                continue
            vad_pause_detected_sec = gen.get("vad_pause_detected_sec")
            llm_generation_started_sec = gen.get("llm_generation_started_sec")
            llm_first_token_sec = gen.get("llm_first_token_sec")
            tts_generation_started_sec = gen.get("tts_generation_started_sec")
            tts_first_token_received_sec = gen.get("tts_first_token_received_sec")
            tts_first_audio_chunk_received_sec = gen.get("tts_first_audio_chunk_received_sec")
            break  # use the first completed generation

        if vad_pause_detected_sec is not None:
            vad_pause_latency = round(vad_pause_detected_sec - input_end_time, 3)
            if vad_pause_latency > 0:
                vad_pause_latency_list.append(vad_pause_latency)
                if llm_generation_started_sec is not None:
                    llm_generation_latency_list.append(round(llm_generation_started_sec - input_end_time, 3))
                if llm_first_token_sec is not None:
                    llm_first_token_latency_list.append(round(llm_first_token_sec - input_end_time, 3))
                if tts_generation_started_sec is not None:
                    tts_generation_latency_list.append(round(tts_generation_started_sec - input_end_time, 3))
                if tts_first_token_received_sec is not None:
                    tts_first_token_latency_list.append(round(tts_first_token_received_sec - input_end_time, 3))
                if tts_first_audio_chunk_received_sec is not None:
                    tts_first_audio_chunk_latency_list.append(round(tts_first_audio_chunk_received_sec - input_end_time, 3))

        num_speculations.append(len(timings_data.get("speculation_attempts", [])))

        # ── TOR + latency from ASR transcript ───────────────────────────────
        # Keep only chunks that start after the input ended
        post_input_chunks = [c for c in chunks if c["timestamp"][0] >= input_end_time]

        TOR = 0
        latency = None
        if post_input_chunks:
            output_start_time = post_input_chunks[0]["timestamp"][0]
            duration = post_input_chunks[-1]["timestamp"][-1] - output_start_time
            turn_duration_threshold = 1.0
            turn_num_words_threshold = 3
            print(duration, len(post_input_chunks))
            if duration < turn_duration_threshold and len(post_input_chunks) <= turn_num_words_threshold:
                TOR = 0
            else:
                TOR = 1
                latency = round(output_start_time - input_end_time, 3)

        take_turn_list.append(TOR)
        if TOR == 1 and latency is not None:
            latency_list.append(max(0.0, latency))

        print(f"{item_dir.name}  TOR={TOR}  latency={latency}  input_end={input_end_time:.2f}s")
        print("---------------------------------------------------")

    def avg(lst: list[float]) -> float | None:
        return round(sum(lst) / len(lst), 3) if lst else None

    print("\n===================================================")
    print("[Result]")
    print(f"Items evaluated:      {len(take_turn_list)}")
    print(f"Average TOR:          {avg([float(x) for x in take_turn_list])}")
    print(f"Average latency:      {avg(latency_list)}")
    print("---------------------------------------------------")
    print("[Speculation Analysis]")
    print(f"  With speculation:    {sum(1 for s in num_speculations if s > 0)}")
    print(f"  Without speculation: {sum(1 for s in num_speculations if s == 0)}")
    lat_w_spec  = [l for l, s in zip(latency_list, num_speculations) if s > 0]
    lat_wo_spec = [l for l, s in zip(latency_list, num_speculations) if s == 0]
    print(f"  Avg latency w/ spec:   {avg(lat_w_spec)}")
    print(f"  Avg latency w/o spec:  {avg(lat_wo_spec)}")
    print("---------------------------------------------------")
    print("[Detailed Latency Results]")
    print(f"  Avg VAD Pause Latency:          {avg(vad_pause_latency_list)}")
    print(f"  Avg LLM Generation Latency:     {avg(llm_generation_latency_list)}")
    print(f"  Avg LLM First Token Latency:    {avg(llm_first_token_latency_list)}")
    print(f"  Avg TTS Generation Latency:     {avg(tts_generation_latency_list)}")
    print(f"  Avg TTS First Token Latency:    {avg(tts_first_token_latency_list)}")
    print(f"  Avg TTS First Audio Chunk:      {avg(tts_first_audio_chunk_latency_list)}")
    print("===================================================\n")

    # Save summary JSON next to the results dir
    summary = {
        "results_dir": str(root),
        "n_items": len(take_turn_list),
        "average_TOR": avg([float(x) for x in take_turn_list]),
        "average_latency": avg(latency_list),
        "average_vad_pause_latency": avg(vad_pause_latency_list),
        "average_llm_generation_latency": avg(llm_generation_latency_list),
        "average_llm_first_token_latency": avg(llm_first_token_latency_list),
        "average_tts_generation_latency": avg(tts_generation_latency_list),
        "average_tts_first_token_latency": avg(tts_first_token_latency_list),
        "average_tts_first_audio_chunk_latency": avg(tts_first_audio_chunk_latency_list),
    }
    out_path = root / "_latency_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency eval for FIT RAG Benchmark results.")
    parser.add_argument("--results_dir", required=True, help="Path to model results dir, e.g. results_tmp/fit_rag_benchmark/unmute_rag_gemma3_12b")
    args = parser.parse_args()
    eval_latency_fit(args.results_dir)
