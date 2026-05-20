from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from scripts.query_vector_index import _build_query_text, _embed_query


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_chunks_by_id(chunks_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row.get("chunk_id"))] = row
    return out


def _iter_benchmark_files(benchmark_root: Path) -> list[Path]:
    files = sorted(benchmark_root.rglob("*.json"))
    return [p for p in files if p.is_file()]


def _count_planned_questions(benchmark_files: list[Path], max_questions: int | None) -> int:
    planned = 0
    for bf in benchmark_files:
        data = _load_json(bf)
        if not isinstance(data, dict):
            continue
        items = data.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question") or "").strip()
            if not q:
                continue
            planned += 1
            if max_questions is not None and planned >= max_questions:
                return planned
    return planned


def _create_progress_bar(total: int):
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(total=total, desc="Retrieval", unit="q", leave=False)
    except Exception:
        return None


def _sanitize_query_model_name(value: str) -> str:
    value = (value or "").strip().lower()
    if not value:
        return "unknown"
    value = value.replace("/", "__")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def _extract_answer_field_hints(answer_field: str | None) -> list[str]:
    if not answer_field:
        return []
    raw = str(answer_field)
    parts = re.split(r"[+/,]", raw)
    out: list[str] = []
    for part in parts:
        p = re.sub(r"\([^)]*\)", "", part).strip()
        if not p:
            continue
        p = re.sub(r"\[\]", "", p)
        p = p.replace(".*.", ".")
        p = p.replace("*", "")
        p = re.sub(r"\.+", ".", p).strip(".")
        if p:
            out.append(p)
    # Preserve order while deduplicating.
    deduped: list[str] = []
    seen: set[str] = set()
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _score_row_by_intent(
    *,
    row: dict[str, Any],
    question: str,
    item_type: str,
    answer_field_hints: list[str],
) -> float:
    source_field = str(row.get("source_field") or "")
    source_field_l = source_field.lower()
    q = (question or "").lower()
    t = (item_type or "").lower()

    bonus = 0.0

    for hint in answer_field_hints:
        h = hint.lower()
        if source_field_l == h:
            bonus += 0.50
        elif source_field_l.startswith(h):
            bonus += 0.35
        elif h in source_field_l:
            bonus += 0.20

    if "phone" in q or "work phone" in q or "phone" in t:
        if "work_phone" in source_field_l or source_field_l.endswith("phone"):
            bonus += 0.65
        if "e_mail" in source_field_l:
            bonus -= 0.20

    if "email" in q or "e-mail" in q or "email" in t:
        if "e_mail" in source_field_l:
            bonus += 0.65
        if "work_phone" in source_field_l:
            bonus -= 0.15

    if "office" in q or "located" in q or "office" in t:
        if "contact.details.room" in source_field_l:
            bonus += 0.60

    if "contact" in q and "group" in q:
        if source_field_l.startswith("team."):
            bonus += 0.70
        if source_field_l.startswith("about."):
            bonus -= 0.35

    if "department" in q or "department" in t:
        if source_field_l.endswith("department"):
            bonus += 0.80
        if "lecturer" in source_field_l or "instructor" in source_field_l:
            bonus -= 0.20

    if "learning_objectives" in t:
        if source_field_l.endswith("learning_objectives"):
            bonus += 0.90

    if "study_literature" in t or "literature" in q:
        if "fundamental_literature" in source_field_l or "study_literature" in source_field_l:
            bonus += 0.85

    if "member_role" in t or "member_of_group" in t:
        if source_field_l.startswith("team."):
            bonus += 0.75
        if source_field_l.startswith("about."):
            bonus -= 0.25

    return bonus


def _retrieve_for_query(
    *,
    query_text: str,
    question: str,
    item_type: str,
    answer_field: str | None,
    emb: np.ndarray,
    chunk_ids: list[str],
    chunks_by_id: dict[str, dict[str, Any]],
    embedding_cfg: dict[str, Any],
    top_k: int,
    rerank_pool_k: int,
    intent_rerank: bool,
) -> list[dict[str, Any]]:
    q = _embed_query(query_text, embedding_cfg)
    scores = emb @ q[0]

    if not intent_rerank:
        top_idx = np.argsort(scores)[::-1][:top_k]
        out_dense: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_idx, start=1):
            idx_i = int(idx)
            cid = chunk_ids[idx_i]
            row = chunks_by_id.get(cid, {})
            out_dense.append(
                {
                    "rank": rank,
                    "score": float(scores[idx_i]),
                    "chunk_id": cid,
                    "entity_type": row.get("entity_type"),
                    "entity_id": row.get("entity_id"),
                    "source_field": row.get("source_field"),
                    "source_path": row.get("source_path"),
                    "preview": str(row.get("chunk_text", ""))[:280],
                }
            )
        return out_dense

    pool_k = max(top_k, rerank_pool_k)
    pool_idx = np.argsort(scores)[::-1][:pool_k]

    hints = _extract_answer_field_hints(answer_field)
    scored_pool: list[tuple[float, int, float]] = []
    for idx in pool_idx:
        idx_i = int(idx)
        cid = chunk_ids[idx_i]
        row = chunks_by_id.get(cid, {})
        dense_score = float(scores[idx_i])
        intent_bonus = _score_row_by_intent(
            row=row,
            question=question,
            item_type=item_type,
            answer_field_hints=hints,
        )
        scored_pool.append((dense_score + intent_bonus, idx_i, intent_bonus))

    scored_pool.sort(key=lambda x: x[0], reverse=True)
    top_idx = scored_pool[:top_k]

    out: list[dict[str, Any]] = []
    for rank, (final_score, idx, intent_bonus) in enumerate(top_idx, start=1):
        cid = chunk_ids[idx]
        row = chunks_by_id.get(cid, {})
        out.append(
            {
                "rank": rank,
                "score": final_score,
                "dense_score": float(scores[idx]),
                "intent_bonus": intent_bonus,
                "chunk_id": cid,
                "entity_type": row.get("entity_type"),
                "entity_id": row.get("entity_id"),
                "source_field": row.get("source_field"),
                "source_path": row.get("source_path"),
                "preview": str(row.get("chunk_text", ""))[:280],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: retrieve chunks from questions and write retrieved_jsons.")
    parser.add_argument("--benchmark_root", type=str, default="FIT_RAG_Benchmark")
    parser.add_argument("--output_root", type=str, default="FIT_RAG_Benchmark_results")
    parser.add_argument("--embeddings_root", type=str, default="embeddings")
    parser.add_argument("--embedding_data_name", type=str, required=True)
    parser.add_argument("--vector_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--query_builder", type=str, choices=["identity", "external"], default="identity")
    parser.add_argument("--query_builder_cmd", type=str, default=None)
    parser.add_argument("--query_model_name", type=str, default=None)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--rerank_pool_k", type=int, default=40)
    parser.add_argument("--intent_rerank", action="store_true", help="Enable answer-field/type-based retrieval reranking (disabled by default).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root).resolve()
    output_root = Path(args.output_root).resolve()

    if args.vector_dir:
        vector_dir = Path(args.vector_dir).resolve()
    else:
        vector_dir = Path(args.embeddings_root).resolve() / args.embedding_data_name

    cfg = _load_json(vector_dir / "generation_config.json")
    artifacts = cfg.get("artifacts") or {}

    chunks_path = Path(str(artifacts.get("chunks_jsonl", vector_dir / "chunks.jsonl"))).resolve()
    emb_path = Path(str(artifacts.get("embeddings_npy", vector_dir / "embeddings.npy"))).resolve()
    ids_path = Path(str(artifacts.get("chunk_ids_json", vector_dir / "chunk_ids.json"))).resolve()

    emb = np.load(emb_path)
    chunk_ids = _load_json(ids_path)
    if not isinstance(chunk_ids, list):
        raise RuntimeError(f"chunk_ids must be a list in {ids_path}")
    chunk_ids = [str(x) for x in chunk_ids]

    chunks_by_id = _load_chunks_by_id(chunks_path)
    embedding_cfg = cfg.get("embedding") or {}

    if args.query_builder == "identity":
        query_model_name = "raw"
    else:
        query_model_name = _sanitize_query_model_name(args.query_model_name or "external")

    query_root = output_root / args.embedding_data_name / query_model_name
    retrieved_jsons_root = query_root / "retrieved_jsons"
    retrieved_jsons_root.mkdir(parents=True, exist_ok=True)

    benchmark_files = _iter_benchmark_files(benchmark_root)
    planned_total = _count_planned_questions(benchmark_files, args.max_questions)
    progress_bar = _create_progress_bar(planned_total)

    processed = 0
    for bf in benchmark_files:
        data = _load_json(bf)
        if not isinstance(data, dict):
            continue
        items = data.get("items")
        if not isinstance(items, list):
            continue

        dataset_name = str(data.get("dataset") or bf.stem)
        dataset_dir = retrieved_jsons_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            if not question:
                continue

            if args.max_questions is not None and processed >= args.max_questions:
                break

            qid = str(item.get("id") or f"q_{processed + 1:05d}")
            out_file = dataset_dir / f"{qid}.json"
            if out_file.exists() and not args.overwrite:
                processed += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                continue

            query_text, query_meta = _build_query_text(
                question=question,
                query_builder=args.query_builder,
                query_builder_cmd=args.query_builder_cmd,
            )
            retrieval = _retrieve_for_query(
                query_text=query_text,
                question=question,
                item_type=str(item.get("type") or ""),
                answer_field=str(item.get("answer_field")) if item.get("answer_field") is not None else None,
                emb=emb,
                chunk_ids=chunk_ids,
                chunks_by_id=chunks_by_id,
                embedding_cfg=embedding_cfg,
                top_k=args.top_k,
                rerank_pool_k=args.rerank_pool_k,
                intent_rerank=bool(args.intent_rerank),
            )

            payload = {
                "benchmark_file": str(bf),
                "dataset": dataset_name,
                "item_id": qid,
                "question": question,
                "query": query_text,
                "query_meta": {"mode": "from_question", **query_meta},
                "top_k": args.top_k,
                "retrieval": retrieval,
                "expected_answer": item.get("expected_answer"),
                "answer_field": item.get("answer_field"),
                "item": item,
            }
            out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            processed += 1
            if progress_bar is not None:
                progress_bar.update(1)

        if args.max_questions is not None and processed >= args.max_questions:
            break

    if progress_bar is not None:
        progress_bar.close()

    summary = {
        "stage": "retrieval",
        "benchmark_root": str(benchmark_root),
        "embedding_data_name": args.embedding_data_name,
        "query_model_name": query_model_name,
        "query_root": str(query_root),
        "retrieved_jsons_dir": str(retrieved_jsons_root),
        "planned_questions": planned_total,
        "processed_questions": processed,
        "top_k": args.top_k,
        "intent_rerank": bool(args.intent_rerank),
        "rerank_pool_k": args.rerank_pool_k if args.intent_rerank else None,
        "query_builder": args.query_builder,
    }
    (query_root / "_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[retrieve_from_question]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
