"""
evaluate_recording_speculativerag.py

Runs the existing speculative evaluator flow, but injects
UnmuteHandlerSpeculativeRAG as the handler implementation.

This keeps existing speculative evaluation plumbing unchanged while enabling:
- speculative RAG prefetch before speculative LLM starts,
- VAD-boundary full-context RAG refresh before commit/continuation.

Usage:
    PYTHONPATH=. python3.12 unmute/scripts/evaluate_recording_speculativerag.py \
        /path/to/input.wav /path/to/output.wav \
        --rag-url http://127.0.0.1:8095 --top-k 5
"""

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

from unmute.unmute_handler_speculative_rag import UnmuteHandlerSpeculativeRAG


def _load_base_speculative_eval_module():
    base_file = Path(__file__).resolve().parent / "evaluate_recording_speculative.py"
    spec = importlib.util.spec_from_file_location(
        "evaluate_recording_speculative_base",
        str(base_file),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base evaluator from {base_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def main(
    input_path: Path,
    output_path: Path,
    voice: str,
    rag_url: str,
    rag_top_k: int,
    rag_timeout_sec: float,
) -> None:
    base_module = _load_base_speculative_eval_module()

    class ConfiguredUnmuteHandlerSpeculativeRAG(UnmuteHandlerSpeculativeRAG):
        last_instance: "ConfiguredUnmuteHandlerSpeculativeRAG | None" = None

        def __init__(self) -> None:
            super().__init__(
                rag_url=rag_url,
                rag_top_k=rag_top_k,
                rag_timeout_sec=rag_timeout_sec,
            )
            ConfiguredUnmuteHandlerSpeculativeRAG.last_instance = self

    def _merge_rag_into_generations(
        generations: list[dict[str, Any]],
        rag_calls: list[dict[str, Any]],
    ) -> None:
        latest_by_generation: dict[int, dict[str, Any]] = {}
        for call in rag_calls:
            if call.get("stage") != "vad_boundary_refresh":
                continue
            generation_index = call.get("generation_index")
            if not isinstance(generation_index, int):
                continue
            latest_by_generation[generation_index] = call

        for generation in generations:
            generation_index = generation.get("generation_index")
            if not isinstance(generation_index, int):
                continue
            call = latest_by_generation.get(generation_index)
            if call is None:
                continue
            generation["rag_started_sec"] = call.get("rag_started_sec")
            generation["rag_finished_sec"] = call.get("rag_finished_sec")
            generation["rag_input_query"] = call.get("rag_input_query")
            generation["rag_top_k_output"] = call.get("rag_top_k_output", [])
            generation["rag_context_collapsed"] = call.get("rag_context_collapsed")
            generation["rag_error"] = call.get("rag_error")
            generation["rag_latency_ms"] = call.get("rag_latency_ms")
            generation["rag_applied_to_prompt"] = bool(call.get("rag_applied_to_prompt", False))
            generation["rag_not_applied_reason"] = call.get("rag_not_applied_reason")
            generation["rag_stage"] = call.get("stage")

    def _augment_timing_json(
        out_path: Path,
        rag_calls: list[dict[str, Any]],
    ) -> None:
        if not out_path.exists():
            return
        try:
            payload = json.loads(out_path.read_text())
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        payload["rag"] = {
            "url": rag_url,
            "top_k": rag_top_k,
            "timeout_sec": rag_timeout_sec,
            "mode": "speculative_prefetch_plus_vad_boundary_refresh",
        }
        payload["rag_calls"] = rag_calls

        generations = payload.get("generations")
        if isinstance(generations, list):
            _merge_rag_into_generations(generations, rag_calls)

        out_path.write_text(json.dumps(payload, indent=2))

        compact = _build_compact(payload)
        out_path.with_name(out_path.stem.replace(".timings", "") + ".compact.json").write_text(
            json.dumps(compact, indent=2)
        )

    def _build_compact(payload: dict[str, Any]) -> dict[str, Any]:
        gen_keys = [
            "generation_index", "vad_pause_detected_sec", "start_trigger",
            "status", "status_reason", "llm_generation_started_sec",
            "llm_input_transcript_at_start", "rag_input_query",
            "rag_context_collapsed", "rag_latency_ms", "rag_stage",
            "rag_applied_to_prompt", "rag_error",
        ]
        spec_keys = [
            "attempt_index", "trigger_time_sec", "status", "status_reason",
            "llm_started_sec", "llm_input_transcript", "generated_text",
            "rag_stage", "rag_started_sec", "rag_context_collapsed",
            "rag_error", "rag_latency_ms", "rag_not_applied_reason",
        ]
        return {
            "input_path": payload.get("input_path"),
            "vad_pause_detection_times_sec": payload.get("vad_pause_detection_times_sec"),
            "rag": payload.get("rag"),
            "generations": [
                {k: g.get(k) for k in gen_keys}
                for g in (payload.get("generations") or [])
            ],
            "speculation_attempts": [
                {k: s.get(k) for k in spec_keys}
                for s in (payload.get("speculation_attempts") or [])
            ],
        }

    # Monkey-patch the handler symbol used inside base main().
    setattr(base_module, "UnmuteHandlerSpeculative", ConfiguredUnmuteHandlerSpeculativeRAG)

    await base_module.main(input_path, output_path, voice)

    handler_instance = ConfiguredUnmuteHandlerSpeculativeRAG.last_instance
    rag_calls: list[dict[str, Any]] = []
    if handler_instance is not None:
        rag_calls = handler_instance.get_rag_calls()
    _augment_timing_json(output_path.with_suffix(".timings.json"), rag_calls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate speculative mode with RAG-first anticipation flow."
    )
    parser.add_argument("input_wav", type=Path, help="Path to input WAV file")
    parser.add_argument("output_wav", type=Path, help="Path to output WAV file")
    parser.add_argument(
        "--voice",
        type=str,
        default="default",
        help="Voice preset for TTS",
    )
    parser.add_argument(
        "--rag-url",
        type=str,
        default="http://127.0.0.1:8095",
        help="Base URL of RAG server",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of retrieval results to include",
    )
    parser.add_argument(
        "--rag-timeout-sec",
        type=float,
        default=1.2,
        help="Timeout for each RAG request",
    )

    args = parser.parse_args()
    asyncio.run(
        main(
            args.input_wav,
            args.output_wav,
            args.voice,
            args.rag_url,
            args.top_k,
            args.rag_timeout_sec,
        )
    )
