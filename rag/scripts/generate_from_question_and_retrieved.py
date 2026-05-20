from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _format_tool_response(retrieval: list[dict[str, Any]], top_n: int) -> str:
    rows = retrieval[: max(top_n, 0)]
    if not rows:
        return "No retrieved content."

    out: list[str] = []
    for i, row in enumerate(rows, start=1):
        preview = str(row.get("preview") or "").strip()
        entity_type = str(row.get("entity_type") or "")
        entity_id = str(row.get("entity_id") or "")
        source_field = str(row.get("source_field") or "")
        out.append(
            f"top{i} - {preview} [entity_type={entity_type}; entity_id={entity_id}; source_field={source_field}]"
        )
    return "\n".join(out)


def _build_answer_messages(*, question: str, tool_response: str, top_k_mode: int) -> list[dict]:
    system = (
        "You are a voice assistant for the Faculty of Information Technology (FIT). "
        "Answer the user's question directly and factually using the retrieved context below. "
        "This knowledge base contains public professional information (staff profiles, contact details, "
        "course info, research groups, projects, publications, etc.) that you are authorised to share. "
        "Do not refuse or hedge information that is present in the context — if it is there, share it. "
        "If the context is insufficient, answer from general knowledge and say so briefly. "
        "Never fabricate information not in the context. "
        "Be concise — one or two sentences at most."
    )
    suffix = (
        " Answer using only the single best retrieved item."
        if top_k_mode == 1
        else " It is sufficient if the answer is present in any one of the retrieved items."
    )
    user = (
        f"Retrieved context:\n{tool_response}\n\n"
        f"Question: {question}{suffix}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class _AnswerGenerator:
    def generate(self, messages: list[dict]) -> str:
        raise NotImplementedError


class _MockAnswerGenerator(_AnswerGenerator):
    def generate(self, messages: list[dict]) -> str:
        return "[mock-answer]"


class _VllmAnswerGenerator(_AnswerGenerator):
    def __init__(
        self,
        *,
        model_name: str,
        max_model_len: int,
        gpu_memory_utilization: float,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        dtype: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        repetition_penalty: float,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as e:  # pragma: no cover - runtime dependency
            raise RuntimeError("vLLM is required for --backend vllm") from e

        self._llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            dtype=dtype,
        )
        self._sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
        )

    def generate(self, messages: list[dict]) -> str:
        out = self._llm.chat([messages], self._sampling, use_tqdm=False)
        if not out or not out[0].outputs:
            return ""
        return str(out[0].outputs[0].text).strip()


def _build_generator(args: argparse.Namespace) -> _AnswerGenerator:
    if args.backend == "mock":
        return _MockAnswerGenerator()
    return _VllmAnswerGenerator(
        model_name=args.model,
        max_model_len=args.llm_max_model_len,
        gpu_memory_utilization=args.llm_gpu_memory_utilization,
        tensor_parallel_size=args.llm_tensor_parallel_size,
        pipeline_parallel_size=args.llm_pipeline_parallel_size,
        dtype=args.llm_dtype,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
        max_tokens=args.llm_max_tokens,
        repetition_penalty=args.llm_repetition_penalty,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_question_files(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*.json")) if p.is_file() and p.name != "_run_summary.json"]


def _sanitize_name(value: str) -> str:
    value = (value or "").strip().lower()
    if not value:
        return "unknown"
    value = value.replace("/", "__")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: generation-only from saved question+retrieval JSON pairs (no retrieval)."
    )
    parser.add_argument("--results_root", type=str, default="FIT_RAG_Benchmark_results")
    parser.add_argument("--embedding_data_name", type=str, required=True)
    parser.add_argument("--query_model_name", type=str, required=True)
    parser.add_argument("--backend", type=str, choices=["vllm", "mock"], default="vllm")
    parser.add_argument("--model", type=str, default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--llm_max_model_len", type=int, default=1024)
    parser.add_argument("--llm_gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--llm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--llm_pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--llm_dtype", type=str, default="half")
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_top_p", type=float, default=0.85)
    parser.add_argument("--llm_max_tokens", type=int, default=256)
    parser.add_argument("--llm_repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    query_root = root / args.embedding_data_name / args.query_model_name
    retrieved_jsons_root = query_root / "retrieved_jsons"

    llm_output_name = f"{args.backend}__{_sanitize_name(args.model)}"
    llm_root = query_root / llm_output_name
    response_jsons_root = llm_root / "response_jsons"
    response_jsons_root.mkdir(parents=True, exist_ok=True)

    files = _iter_question_files(retrieved_jsons_root)
    if args.max_questions is not None:
        files = files[: args.max_questions]

    generator = _build_generator(args)
    processed = 0

    for src_path in files:
        rel = src_path.relative_to(retrieved_jsons_root)
        dst_path = response_jsons_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists() and not args.overwrite:
            processed += 1
            continue

        data = _load_json(src_path)
        if not isinstance(data, dict):
            continue

        question = str(data.get("question") or "").strip()
        retrieval = data.get("retrieval") if isinstance(data.get("retrieval"), list) else []

        tool_response_top1 = _format_tool_response(retrieval, top_n=1)
        tool_response_top5 = _format_tool_response(retrieval, top_n=5)
        top1_messages = _build_answer_messages(question=question, tool_response=tool_response_top1, top_k_mode=1)
        top5_messages = _build_answer_messages(question=question, tool_response=tool_response_top5, top_k_mode=5)

        payload = {
            **data,
            "generation_mode": "from_saved_retrieval",
            "llm_answers": {
                "enabled": True,
                "backend": args.backend,
                "model": args.model,
                "top1_prompt": top1_messages,
                "top1_answer": generator.generate(top1_messages),
                "top5_prompt": top5_messages,
                "top5_answer": generator.generate(top5_messages),
            },
        }

        dst_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        processed += 1

    summary = {
        "stage": "generation",
        "retrieval_performed": False,
        "results_root": str(root),
        "embedding_data_name": args.embedding_data_name,
        "query_model_name": args.query_model_name,
        "retrieved_jsons_dir": str(retrieved_jsons_root),
        "llm_output_name": llm_output_name,
        "llm_root": str(llm_root),
        "response_jsons_dir": str(response_jsons_root),
        "planned_questions": len(files),
        "processed_questions": processed,
    }
    (llm_root / "_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[generate_from_question_and_retrieved]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
