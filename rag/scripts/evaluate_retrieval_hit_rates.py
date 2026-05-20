from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DATASET_COLUMN_ORDER = [
    "overall",
    "longform",
    "courses",
    "groups",
    "personnel",
    "projects",
    "publications",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_text(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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
            out.append(p.lower())
    deduped: list[str] = []
    seen: set[str] = set()
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _iter_strings(value: Any) -> list[str]:
    out: list[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        t = value.strip()
        if t:
            out.append(t)
        return out
    if isinstance(value, (int, float)):
        out.append(str(value))
        return out
    if isinstance(value, bool):
        return out
    if isinstance(value, list):
        for v in value:
            out.extend(_iter_strings(v))
        return out
    if isinstance(value, dict):
        for v in value.values():
            out.extend(_iter_strings(v))
        return out
    return out


def _extract_entity_targets(item: dict[str, Any]) -> set[str]:
    keys = ["course_code", "person", "group", "project", "publication_title"]
    out: set[str] = set()
    for key in keys:
        if key in item and item.get(key) is not None:
            norm = _normalize_text(item.get(key))
            if norm:
                out.add(norm)
    return out


def _extract_answer_targets(item: dict[str, Any], min_len: int) -> set[str]:
    out: set[str] = set()
    for s in _iter_strings(item.get("actual_answer")):
        norm = _normalize_text(s)
        if len(norm) >= min_len:
            out.add(norm)
    return out


def _technique_dirs(results_root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(results_root.rglob("retrieved_jsons")):
        if p.is_dir():
            out.append(p.parent)
    return out


def _iter_retrieved_jsons(retrieved_root: Path) -> list[Path]:
    files = [p for p in sorted(retrieved_root.rglob("*.json")) if p.is_file()]
    return [p for p in files if not p.name.startswith("_")]


def _first_entity_match_rank(retrieval_rows: list[dict[str, Any]], targets: set[str], top_k: int) -> int | None:
    if not targets:
        return None
    for i, row in enumerate(retrieval_rows[:top_k], start=1):
        entity_id = _normalize_text(row.get("entity_id"))
        if entity_id and entity_id in targets:
            return i
    return None


def _field_hit(retrieval_rows: list[dict[str, Any]], field_hints: list[str], top_k: int) -> bool:
    if not field_hints:
        return False
    for row in retrieval_rows[:top_k]:
        src = str(row.get("source_field") or "").lower()
        if not src:
            continue
        for h in field_hints:
            if src == h or src.startswith(h) or h in src:
                return True
    return False


def _answer_hit(retrieval_rows: list[dict[str, Any]], targets: set[str], top_k: int) -> bool:
    if not targets:
        return False
    for row in retrieval_rows[:top_k]:
        hay = " ".join(
            [
                _normalize_text(row.get("preview")),
                _normalize_text(row.get("entity_id")),
                _normalize_text(row.get("source_field")),
            ]
        ).strip()
        if not hay:
            continue
        for t in targets:
            if t and t in hay:
                return True
    return False


def _empty_stats() -> dict[str, Any]:
    return {
        "questions": 0,
        "entity_hits": 0,
        "field_hits": 0,
        "answer_hits": 0,
        "any_hits": 0,
        "mrr_entity_sum": 0.0,
    }


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    q = int(stats["questions"])
    out = {
        "questions": q,
        "entity_hit_rate": (stats["entity_hits"] / q) if q else None,
        "field_hit_rate": (stats["field_hits"] / q) if q else None,
        "answer_hit_rate": (stats["answer_hits"] / q) if q else None,
        "any_hit_rate": (stats["any_hits"] / q) if q else None,
        "entity_mrr": (stats["mrr_entity_sum"] / q) if q else None,
    }
    return out


def _dataset_bucket_name(dataset_name: str) -> str:
    d = (dataset_name or "").lower()
    if "longform" in d:
        return "longform"
    if "courses" in d:
        return "courses"
    if "groups" in d:
        return "groups"
    if "personnel" in d:
        return "personnel"
    if "projects" in d:
        return "projects"
    if "publications" in d:
        return "publications"
    return "other"


def _evaluate_single_topk(
    *,
    techniques: list[Path],
    top_k: int,
    answer_min_len: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for tech_dir in techniques:
        retrieved_root = tech_dir / "retrieved_jsons"
        files = _iter_retrieved_jsons(retrieved_root)
        if not files:
            continue

        stats = _empty_stats()
        by_dataset: dict[str, dict[str, Any]] = {}

        for f in files:
            data = _load_json(f)
            if not isinstance(data, dict):
                continue

            retrieval_rows = data.get("retrieval") if isinstance(data.get("retrieval"), list) else []
            item = data.get("item") if isinstance(data.get("item"), dict) else {}
            dataset = str(data.get("dataset") or "unknown")

            ds = by_dataset.setdefault(dataset, _empty_stats())

            entity_targets = _extract_entity_targets(item)
            answer_targets = _extract_answer_targets(item, min_len=answer_min_len)
            field_hints = _extract_answer_field_hints(str(data.get("answer_field") or ""))

            rank = _first_entity_match_rank(retrieval_rows, entity_targets, top_k)
            hit_entity = rank is not None
            hit_field = _field_hit(retrieval_rows, field_hints, top_k)
            hit_answer = _answer_hit(retrieval_rows, answer_targets, top_k)
            hit_any = hit_entity or hit_field or hit_answer

            for bucket in (stats, ds):
                bucket["questions"] += 1
                bucket["entity_hits"] += int(hit_entity)
                bucket["field_hits"] += int(hit_field)
                bucket["answer_hits"] += int(hit_answer)
                bucket["any_hits"] += int(hit_any)
                if rank is not None:
                    bucket["mrr_entity_sum"] += 1.0 / float(rank)

        embedding_name = tech_dir.parent.name if tech_dir.parent else "unknown_embedding"
        query_model_name = tech_dir.name
        tech_key = f"{embedding_name}/{query_model_name}"

        out[tech_key] = {
            "path": str(tech_dir),
            "overall": _finalize_stats(stats),
            "by_dataset": {k: _finalize_stats(v) for k, v in sorted(by_dataset.items())},
        }

    return out


def _build_rate_table(
    *,
    techniques_by_topk: dict[str, dict[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    topk_keys = sorted(techniques_by_topk.keys(), key=lambda x: int(x))
    all_techniques: list[str] = sorted({tk for by_k in techniques_by_topk.values() for tk in by_k.keys()})

    def _extract_metric(summary: dict[str, Any], column: str) -> float | None:
        return _extract_metric_from_summary(summary=summary, metric_name=metric_name, column=column)

    tables: dict[str, list[dict[str, Any]]] = {}
    for tk in topk_keys:
        rows: list[dict[str, Any]] = []
        by_tech = techniques_by_topk[tk]
        for tech in all_techniques:
            s = by_tech.get(tech)
            if not s:
                continue
            row: dict[str, Any] = {"technique": tech}
            for col in DATASET_COLUMN_ORDER:
                row[col] = _extract_metric(s, col)
            rows.append(row)
        tables[f"top{tk}"] = rows

    return {
        "columns": ["technique", *DATASET_COLUMN_ORDER],
        "tables": tables,
    }


def _build_compare_summary(techniques_by_topk: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "field_hit_rate": _build_rate_table(techniques_by_topk=techniques_by_topk, metric_name="field_hit_rate"),
        "answer_hit_rate": _build_rate_table(techniques_by_topk=techniques_by_topk, metric_name="answer_hit_rate"),
    }


def _extract_metric_from_summary(*, summary: dict[str, Any], metric_name: str, column: str) -> float | None:
    if column == "overall":
        return summary.get("overall", {}).get(metric_name)
    wanted = [
        ds_stats.get(metric_name)
        for ds_name, ds_stats in (summary.get("by_dataset") or {}).items()
        if _dataset_bucket_name(ds_name) == column
    ]
    wanted = [x for x in wanted if x is not None]
    if not wanted:
        return None
    return float(sum(wanted) / len(wanted))


def _pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{100.0 * float(v):.2f}%"


def _score(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{float(v):.2f}"


def _collect_llm_eval_rows(
    *,
    results_root: Path,
    embedding_name: str,
    query_model_name: str,
) -> list[dict[str, Any]]:
    query_root = results_root / embedding_name / query_model_name
    if not query_root.is_dir():
        return []

    rows: list[dict[str, Any]] = []
    for gen_dir in sorted([p for p in query_root.iterdir() if p.is_dir()]):
        eval_root = gen_dir / "eval_jsons"
        if not eval_root.is_dir():
            continue

        for judge_dir in sorted([p for p in eval_root.iterdir() if p.is_dir()]):
            values_by_col_top1: dict[str, list[float]] = {c: [] for c in DATASET_COLUMN_ORDER}
            values_by_col_top1["overall"] = []
            values_by_col_top5: dict[str, list[float]] = {c: [] for c in DATASET_COLUMN_ORDER}
            values_by_col_top5["overall"] = []

            for dataset_dir in sorted([p for p in judge_dir.iterdir() if p.is_dir()]):
                bucket = _dataset_bucket_name(dataset_dir.name)
                for f in sorted(dataset_dir.rglob("*.json")):
                    if not f.is_file():
                        continue
                    try:
                        data = _load_json(f)
                    except Exception:
                        continue
                    if not isinstance(data, dict):
                        continue
                    raw_top1 = data.get("top1_score")
                    if raw_top1 is not None:
                        try:
                            val_top1 = float(raw_top1)
                            values_by_col_top1["overall"].append(val_top1)
                            if bucket in values_by_col_top1:
                                values_by_col_top1[bucket].append(val_top1)
                        except Exception:
                            pass

                    raw_top5 = data.get("top5_score")
                    if raw_top5 is not None:
                        try:
                            val_top5 = float(raw_top5)
                            values_by_col_top5["overall"].append(val_top5)
                            if bucket in values_by_col_top5:
                                values_by_col_top5[bucket].append(val_top5)
                        except Exception:
                            pass

            if values_by_col_top1["overall"]:
                row_top1: dict[str, Any] = {
                    "k": f"llm_top1:{gen_dir.name}|{judge_dir.name}",
                    "is_llm": True,
                }
                for col in DATASET_COLUMN_ORDER:
                    vals = values_by_col_top1.get(col, [])
                    row_top1[col] = (sum(vals) / len(vals)) if vals else None
                rows.append(row_top1)

            if values_by_col_top5["overall"]:
                row_top5: dict[str, Any] = {
                    "k": f"llm_top5:{gen_dir.name}|{judge_dir.name}",
                    "is_llm": True,
                }
                for col in DATASET_COLUMN_ORDER:
                    vals = values_by_col_top5.get(col, [])
                    row_top5[col] = (sum(vals) / len(vals)) if vals else None
                rows.append(row_top5)

    return rows


def _print_metric_table_for_embedding(
    *,
    embedding_name: str,
    query_model_name: str,
    techniques_by_topk: dict[str, dict[str, Any]],
    topks: list[int],
    metric_name: str,
    llm_rows: list[dict[str, Any]],
) -> None:
    tech_key = f"{embedding_name}/{query_model_name}"
    cols = ["k", *DATASET_COLUMN_ORDER]
    label_candidates = [f"top{k}" for k in topks] + [str(r.get("k") or "llm") for r in llm_rows]
    first_col_width = max([12, len("k")] + [len(x) for x in label_candidates]) + 2
    other_col_width = 12

    print(f"\n{metric_name}")
    header = [cols[0].ljust(first_col_width)] + [c.ljust(other_col_width) for c in cols[1:]]
    print(" ".join(header))
    print("-" * (first_col_width + (other_col_width + 1) * (len(cols) - 1)))

    for k in topks:
        summary = (techniques_by_topk.get(str(k)) or {}).get(tech_key)
        if not summary:
            continue
        row_vals = [f"top{k}"]
        for col in DATASET_COLUMN_ORDER:
            val = _extract_metric_from_summary(summary=summary, metric_name=metric_name, column=col)
            row_vals.append(_pct(val))
        formatted = [row_vals[0].ljust(first_col_width)[:first_col_width]] + [
            v.ljust(other_col_width)[:other_col_width] for v in row_vals[1:]
        ]
        print(" ".join(formatted))

    for llm_row in llm_rows:
        row_vals = [str(llm_row.get("k") or "llm")]
        for col in DATASET_COLUMN_ORDER:
            val = llm_row.get(col)
            if metric_name in {"field_hit_rate", "answer_hit_rate"}:
                row_vals.append(_score(val))
            else:
                row_vals.append(_score(val))
        formatted = [row_vals[0].ljust(first_col_width)[:first_col_width]] + [
            v.ljust(other_col_width)[:other_col_width] for v in row_vals[1:]
        ]
        print(" ".join(formatted))


def _print_tables_per_embedding(
    *,
    techniques_by_topk: dict[str, dict[str, Any]],
    topks: list[int],
    results_root: Path,
) -> None:
    all_techniques = sorted({tk for by_k in techniques_by_topk.values() for tk in by_k.keys()})
    embeddings: dict[str, list[str]] = {}
    for tk in all_techniques:
        if "/" in tk:
            emb, qm = tk.rsplit("/", 1)
        else:
            emb, qm = tk, "raw"
        embeddings.setdefault(emb, []).append(qm)

    print("[retrieval_hit_rates]")
    for emb in sorted(embeddings.keys()):
        for qm in sorted(set(embeddings[emb])):
            llm_rows = _collect_llm_eval_rows(
                results_root=results_root,
                embedding_name=emb,
                query_model_name=qm,
            )
            print(f"\n=== {emb} / {qm} ===")
            _print_metric_table_for_embedding(
                embedding_name=emb,
                query_model_name=qm,
                techniques_by_topk=techniques_by_topk,
                topks=topks,
                metric_name="field_hit_rate",
                llm_rows=llm_rows,
            )
            _print_metric_table_for_embedding(
                embedding_name=emb,
                query_model_name=qm,
                techniques_by_topk=techniques_by_topk,
                topks=topks,
                metric_name="answer_hit_rate",
                llm_rows=llm_rows,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone retrieval-only hit-rate evaluator over all technique folders in FIT_RAG_Benchmark_results."
    )
    parser.add_argument("--results_root", type=str, default="FIT_RAG_Benchmark_results")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--compare_topks",
        type=str,
        default="1,5",
        help="Comma-separated top-k values for comparative tables (e.g. 1,5).",
    )
    parser.add_argument(
        "--answer_min_len",
        type=int,
        default=4,
        help="Minimum normalized token-string length used as answer target for lexical hit checks.",
    )
    parser.add_argument(
        "--llm_score_field",
        type=str,
        default="both",
        help="Deprecated; LLM rows now always include both top1_score and top5_score.",
    )
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to write summary JSON")
    parser.add_argument("--print_json", action="store_true", help="Also print full JSON summary to stdout.")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top_k must be > 0")

    results_root = Path(args.results_root).resolve()
    techniques = _technique_dirs(results_root)

    compare_topks = [x.strip() for x in str(args.compare_topks or "").split(",") if x.strip()]
    compare_topks_int: list[int] = []
    for tok in compare_topks:
        v = int(tok)
        if v <= 0:
            raise ValueError("--compare_topks values must be > 0")
        compare_topks_int.append(v)

    if args.top_k not in compare_topks_int:
        compare_topks_int.append(args.top_k)
    compare_topks_int = sorted(set(compare_topks_int))

    techniques_by_topk: dict[str, dict[str, Any]] = {}
    for k in compare_topks_int:
        techniques_by_topk[str(k)] = _evaluate_single_topk(
            techniques=techniques,
            top_k=k,
            answer_min_len=args.answer_min_len,
        )

    summary: dict[str, Any] = {
        "results_root": str(results_root),
        "top_k": args.top_k,
        "compare_topks": compare_topks_int,
        "answer_min_len": args.answer_min_len,
        "techniques": techniques_by_topk[str(args.top_k)],
        "compare": _build_compare_summary(techniques_by_topk),
    }

    if args.output_json:
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_tables_per_embedding(
        techniques_by_topk=techniques_by_topk,
        topks=compare_topks_int,
        results_root=results_root,
    )
    if args.print_json:
        print("\n[retrieval_hit_rates_json]")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
