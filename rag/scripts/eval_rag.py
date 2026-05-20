from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_question_files(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*.json")) if p.is_file() and p.name != "_run_summary.json"]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def _extract_topic(data: dict[str, Any], dataset_name: str) -> str:
    topic = data.get("topic")
    if isinstance(topic, str) and topic.strip():
        return topic.strip()

    item = data.get("item")
    if isinstance(item, dict):
        item_type = item.get("type")
        if isinstance(item_type, str) and item_type.strip():
            return item_type.strip()

    answer_field = data.get("answer_field")
    if isinstance(answer_field, str) and answer_field.strip():
        return answer_field.strip()

    return dataset_name


def _build_score_rubric() -> str:
    rubric_data = {
        "criteria": (
            "Grade ONLY factual correctness against the reference answer. "
            "Do NOT penalize concise answers for being short. "
            "Semantic equivalence is sufficient, even if wording differs."
        ),
        "score1_description": (
            "Incorrect, contradictory, hallucinated, or missing answer. "
            "Includes 'Not found in retrieved content.' when the reference answer is present and specific."
        ),
        "score2_description": "Mostly incorrect answer with only weak topical relevance.",
        "score3_description": "Partially correct answer; some core facts are missing or mixed with errors.",
        "score4_description": "Largely correct answer with minor omissions or imprecision.",
        "score5_description": (
            "Fully correct answer (exact or semantically equivalent). "
            "For yes/no questions, exact polarity match (e.g., 'Yes', 'Yes.') should receive 5."
        ),
    }

    try:
        from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE  # type: ignore

        return SCORE_RUBRIC_TEMPLATE.format(**rubric_data)
    except Exception:
        return (
            "Criteria: {criteria}\n"
            "1: {score1_description}\n"
            "2: {score2_description}\n"
            "3: {score3_description}\n"
            "4: {score4_description}\n"
            "5: {score5_description}"
        ).format(**rubric_data)


@dataclass
class GradeResult:
    feedback: str
    score: float


class _Judge:
    def single_absolute_grade(self, *, instruction: str, response: str, reference_answer: str, rubric: str) -> GradeResult:
        raise NotImplementedError


class _PrometheusJudge(_Judge):
    def __init__(
        self,
        model_name: str,
        vllm_kwargs: dict[str, Any] | None = None,
        grade_params: dict[str, Any] | None = None,
    ) -> None:
        from prometheus_eval import PrometheusEval  # type: ignore
        from prometheus_eval.prompts import ABSOLUTE_PROMPT  # type: ignore
        from prometheus_eval.vllm import VLLM  # type: ignore

        model = VLLM(model=model_name, **(vllm_kwargs or {}))
        self._judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)
        self._grade_params = grade_params or {}

    def single_absolute_grade(self, *, instruction: str, response: str, reference_answer: str, rubric: str) -> GradeResult:
        feedback, score = self._judge.single_absolute_grade(
            instruction=instruction,
            response=response,
            rubric=rubric,
            reference_answer=reference_answer,
            params=self._grade_params,
        )
        return GradeResult(feedback=str(feedback), score=float(score))


def _parse_result_score(text: str) -> int | None:
    match = re.search(r"\[RESULT\]\s*([0-5])", text, flags=re.IGNORECASE)
    if not match:
        return None
    score = int(match.group(1))
    return max(score, 1)  # clamp 0 → 1 (rubric minimum; judge uses 0 for fully wrong answers)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "unknown"


def _render_progress(done: int, total: int, width: int = 30) -> None:
    if total <= 0:
        sys.stderr.write("[eval_rag] progress [------------------------------] 0/0\n")
        sys.stderr.flush()
        return
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\r[eval_rag] progress [{bar}] {done}/{total}")
    if done >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _progress_iter(files: list[Path]) -> tuple[Any, bool]:
    total = len(files)
    try:
        from tqdm import tqdm  # type: ignore

        return (
            enumerate(
            tqdm(files, total=total, desc="[eval_rag] questions", unit="q", dynamic_ncols=True),
            start=1,
            ),
            True,
        )
    except Exception:
        return enumerate(files, start=1), False


class _VllmJudge(_Judge):
    def __init__(
        self,
        model_name: str,
        vllm_kwargs: dict[str, Any] | None = None,
        grade_params: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from vllm import LLM, SamplingParams  # type: ignore

        self._llm = LLM(model=model_name, **(vllm_kwargs or {}))
        self._chat_template_kwargs = chat_template_kwargs
        p = grade_params or {}
        self._sampling = SamplingParams(
            max_tokens=int(p.get("max_tokens", 256)),
            temperature=float(p.get("temperature", 0.0)),
            top_p=float(p.get("top_p", 0.95)),
            repetition_penalty=float(p.get("repetition_penalty", 1.0)),
        )

    def single_absolute_grade(self, *, instruction: str, response: str, reference_answer: str, rubric: str) -> GradeResult:
        prompt = (
            "Given QUESTION, REFERENCE ANSWER, and CANDIDATE ANSWER, "
            "assign one integer score from 1 to 5 where 1 is completely wrong/missing and 5 is fully correct. "
            "Do not penalize concise answers. For yes/no questions, exact polarity match should be 5."
            "Ignore statements such as 'let me check' and do not deduct points for them.\n\n"
            f"QUESTION:\n{instruction}\n\n"
            f"REFERENCE ANSWER:\n{reference_answer}\n\n"
            f"CANDIDATE ANSWER:\n{response}\n\n"
            "Return exactly one line:\n"
            "[RESULT] <1-5>"
        )
        messages = [
            {"role": "system", "content": "You are a strict grading assistant."},
            {"role": "user", "content": prompt},
        ]
        out = self._llm.chat(
            messages,
            self._sampling,
            use_tqdm=False,
            chat_template_kwargs=self._chat_template_kwargs,
        )
        if not out or not out[0].outputs:
            raise RuntimeError("vLLM judge returned empty output")
        text = str(out[0].outputs[0].text).strip()
        score = _parse_result_score(text)
        if score is None:
            raise RuntimeError(f"Could not parse judge score from output: {text}")
        return GradeResult(feedback=text, score=float(score))


class _MockJudge(_Judge):
    def single_absolute_grade(self, *, instruction: str, response: str, reference_answer: str, rubric: str) -> GradeResult:
        resp = response.lower()
        ref = reference_answer.lower().strip()
        if not ref:
            return GradeResult(feedback="No reference answer.", score=3.0)
        if ref in resp:
            return GradeResult(feedback="Reference answer found.", score=5.0)
        ref_tokens = [t for t in ref.split() if len(t) > 3]
        overlap = sum(1 for t in ref_tokens if t in resp)
        denom = max(len(ref_tokens), 1)
        frac = overlap / denom
        if frac <= 0.05:
            return GradeResult(feedback="No answer evidence.", score=1.0)
        if frac <= 0.2:
            return GradeResult(feedback="Weak relevance.", score=2.0)
        if frac <= 0.45:
            return GradeResult(feedback="Partially relevant.", score=3.0)
        return GradeResult(feedback="Mostly relevant.", score=4.0)


def _normalize_binary_answer(text: str) -> str | None:
    t = (text or "").strip().lower().rstrip(".!")
    if t in {"yes", "true", "y", "1"}:
        return "yes"
    if t in {"no", "false", "n", "0"}:
        return "no"
    return None


def _apply_binary_exact_match_override(*, expected_answer: str, predicted_answer: str, result: GradeResult) -> tuple[GradeResult, bool]:
    exp = _normalize_binary_answer(expected_answer)
    pred = _normalize_binary_answer(predicted_answer)
    if exp is None or pred is None:
        return result, False
    if exp == pred:
        return GradeResult(feedback=f"Binary exact-match override applied ({exp}).", score=5.0), True
    return result, False


def _build_judge(
    backend: str,
    model_name: str,
    vllm_kwargs: dict[str, Any] | None = None,
    grade_params: dict[str, Any] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> _Judge:
    if backend == "mock":
        return _MockJudge()
    if backend == "vllm":
        return _VllmJudge(
            model_name=model_name,
            vllm_kwargs=vllm_kwargs,
            grade_params=grade_params,
            chat_template_kwargs=chat_template_kwargs,
        )
    return _PrometheusJudge(
        model_name=model_name,
        vllm_kwargs=vllm_kwargs,
        grade_params=grade_params,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: evaluate generated answers and write eval_jsons.")
    parser.add_argument("--results_root", type=str, default="FIT_RAG_Benchmark_results")
    parser.add_argument("--embedding_data_name", type=str, required=True)
    parser.add_argument("--query_model_name", type=str, required=True)
    parser.add_argument("--gen_llm_name", type=str, required=True, help="Folder name under query model, e.g., mock__allenai_olmo_3_7b_instruct")
    parser.add_argument("--backend", type=str, choices=["prometheus", "vllm", "mock"], default="prometheus")
    parser.add_argument("--model", type=str, default="prometheus-eval/prometheus-7b-v2.0")
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--store_feedback", action="store_true")

    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--vllm_max_model_len", type=int, default=512)
    parser.add_argument("--vllm_dtype", type=str, default="half")
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--vllm_swap_space", type=float, default=4.0)
    parser.add_argument(
        "--vllm_max_num_seqs",
        type=int,
        default=32,
        help="Limit concurrent sequences in vLLM sampler warmup/runtime to reduce OOM risk.",
    )
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--vllm_language_model_only",
        action="store_true",
        help="Pass language_model_only=True to vLLM engine (useful for Qwen3.5 style models).",
    )
    parser.add_argument(
        "--vllm_reasoning_parser",
        type=str,
        default=None,
        help="Optional vLLM reasoning parser name (e.g., qwen3).",
    )
    parser.add_argument(
        "--vllm_default_chat_template_kwargs",
        type=str,
        default=None,
        help="JSON string for vLLM default_chat_template_kwargs, e.g. '{\"enable_thinking\": false}'.",
    )

    parser.add_argument("--judge_max_tokens", type=int, default=1024)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--judge_top_p", type=float, default=0.95)
    parser.add_argument("--judge_repetition_penalty", type=float, default=1.03)

    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    llm_root = root / args.embedding_data_name / args.query_model_name / args.gen_llm_name
    response_jsons_root = llm_root / "response_jsons"
    judge_id = f"{args.backend}__{_slugify(args.model)}"
    eval_jsons_root = llm_root / "eval_jsons" / judge_id
    eval_jsons_root.mkdir(parents=True, exist_ok=True)

    files = _iter_question_files(response_jsons_root)
    if args.max_questions is not None:
        files = files[: args.max_questions]

    vllm_kwargs: dict[str, Any] = {
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "max_model_len": args.vllm_max_model_len,
        "dtype": args.vllm_dtype,
        "enforce_eager": bool(args.vllm_enforce_eager),
        "swap_space": float(args.vllm_swap_space),
        "max_num_seqs": int(args.vllm_max_num_seqs),
        "tensor_parallel_size": int(args.vllm_tensor_parallel_size),
        "pipeline_parallel_size": int(args.vllm_pipeline_parallel_size),
    }
    if args.vllm_language_model_only:
        vllm_kwargs["language_model_only"] = True
    if args.vllm_reasoning_parser:
        vllm_kwargs["reasoning_parser"] = str(args.vllm_reasoning_parser)
    chat_template_kwargs: dict[str, Any] | None = None
    if args.vllm_default_chat_template_kwargs:
        try:
            chat_template_kwargs = json.loads(args.vllm_default_chat_template_kwargs)
        except Exception as e:
            raise ValueError(
                "--vllm_default_chat_template_kwargs must be valid JSON, "
                "for example: '{\"enable_thinking\": false}'"
            ) from e

    grade_params: dict[str, Any] = {
        "max_tokens": int(args.judge_max_tokens),
        "temperature": float(args.judge_temperature),
        "top_p": float(args.judge_top_p),
        "repetition_penalty": float(args.judge_repetition_penalty),
    }

    judge = _build_judge(
        backend=args.backend,
        model_name=args.model,
        vllm_kwargs=vllm_kwargs,
        grade_params=grade_params,
        chat_template_kwargs=chat_template_kwargs,
    )
    rubric = _build_score_rubric()

    top1_scores: list[float] = []
    top5_scores: list[float] = []
    topic_scores: dict[str, dict[str, list[float]]] = {}

    if not files:
        _render_progress(0, 0)

    progress_entries, use_tqdm = _progress_iter(files)

    for idx, f in progress_entries:
        data = _load_json(f)
        if not isinstance(data, dict):
            if not use_tqdm:
                _render_progress(idx, len(files))
            continue

        llm_answers = data.get("llm_answers") if isinstance(data.get("llm_answers"), dict) else {}
        question = _safe_text(data.get("question"))
        expected_answer = _safe_text(data.get("expected_answer"))
        top1_answer = _safe_text(llm_answers.get("top1_answer"))
        top5_answer = _safe_text(llm_answers.get("top5_answer"))

        g1 = judge.single_absolute_grade(
            instruction=question,
            response=top1_answer,
            reference_answer=expected_answer,
            rubric=rubric,
        )
        g5 = judge.single_absolute_grade(
            instruction=question,
            response=top5_answer,
            reference_answer=expected_answer,
            rubric=rubric,
        )

        g1, g1_binary_override = _apply_binary_exact_match_override(
            expected_answer=expected_answer,
            predicted_answer=top1_answer,
            result=g1,
        )
        g5, g5_binary_override = _apply_binary_exact_match_override(
            expected_answer=expected_answer,
            predicted_answer=top5_answer,
            result=g5,
        )

        top1_scores.append(g1.score)
        top5_scores.append(g5.score)

        dataset_name = str(data.get("dataset") or "unknown")
        topic_name = _extract_topic(data, dataset_name)
        item_id = str(data.get("item_id") or f.stem)

        if topic_name not in topic_scores:
            topic_scores[topic_name] = {"top1": [], "top5": []}
        topic_scores[topic_name]["top1"].append(g1.score)
        topic_scores[topic_name]["top5"].append(g5.score)

        row = {
            "file": str(f),
            "dataset": dataset_name,
            "topic": topic_name,
            "item_id": item_id,
            "question": question,
            "expected_answer": expected_answer,
            "top1_answer": top1_answer,
            "top5_answer": top5_answer,
            "top1_score": g1.score,
            "top5_score": g5.score,
            "top1_binary_override": g1_binary_override,
            "top5_binary_override": g5_binary_override,
        }
        if args.store_feedback:
            row["top1_feedback"] = g1.feedback
            row["top5_feedback"] = g5.feedback

        dataset_eval_dir = eval_jsons_root / dataset_name
        dataset_eval_dir.mkdir(parents=True, exist_ok=True)
        (dataset_eval_dir / f"{item_id}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        if not use_tqdm:
            _render_progress(idx, len(files))

    top1_mean, top1_std = _mean_std(top1_scores)
    top5_mean, top5_std = _mean_std(top5_scores)

    topic_means: dict[str, Any] = {}
    topic_top1_means: list[float] = []
    topic_top5_means: list[float] = []
    for topic_name in sorted(topic_scores.keys()):
        t_top1 = topic_scores[topic_name]["top1"]
        t_top5 = topic_scores[topic_name]["top5"]
        t_top1_mean, t_top1_std = _mean_std(t_top1)
        t_top5_mean, t_top5_std = _mean_std(t_top5)

        if t_top1_mean is not None:
            topic_top1_means.append(t_top1_mean)
        if t_top5_mean is not None:
            topic_top5_means.append(t_top5_mean)

        topic_means[topic_name] = {
            "question_files": len(t_top1),
            "top1_mean": t_top1_mean,
            "top1_std": t_top1_std,
            "top5_mean": t_top5_mean,
            "top5_std": t_top5_std,
        }

    top1_macro_mean_topics = float(statistics.mean(topic_top1_means)) if topic_top1_means else None
    top5_macro_mean_topics = float(statistics.mean(topic_top5_means)) if topic_top5_means else None

    summary = {
        "stage": "evaluation",
        "results_root": str(root),
        "embedding_data_name": args.embedding_data_name,
        "query_model_name": args.query_model_name,
        "gen_llm_name": args.gen_llm_name,
        "llm_root": str(llm_root),
        "response_jsons_dir": str(response_jsons_root),
        "eval_jsons_dir": str(eval_jsons_root),
        "judge": {
            "judge_id": judge_id,
            "backend": args.backend,
            "model": args.model,
            "vllm_kwargs": vllm_kwargs if args.backend in {"prometheus", "vllm"} else None,
            "grade_params": grade_params,
            "retries_enabled": False if args.backend == "vllm" else None,
            "fail_fast": True if args.backend == "vllm" else None,
        },
        "overall": {
            "question_files": len(files),
            "topics": len(topic_means),
            "top1_mean": top1_mean,
            "top1_std": top1_std,
            "top5_mean": top5_mean,
            "top5_std": top5_std,
            "top1_macro_mean_across_topics": top1_macro_mean_topics,
            "top5_macro_mean_across_topics": top5_macro_mean_topics,
        },
        "topics": topic_means,
    }

    (llm_root / f"_eval_summary__{judge_id}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[eval_rag]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
