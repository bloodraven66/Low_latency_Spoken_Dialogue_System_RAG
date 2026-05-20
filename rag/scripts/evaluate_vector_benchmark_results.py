from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_question_files(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*.json")) if p.is_file() and p.name != "_run_summary.json"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vector benchmark result JSON files for hit/miss metrics.")
    parser.add_argument("--results_dir", type=str, required=True, help="Path like FIT_RAG_Benchmark_results/{embedding}/{querymodel}")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    files = _iter_question_files(results_dir / "jsons") if (results_dir / "jsons").exists() else _iter_question_files(results_dir)

    total = 0
    mapped = 0
    entity_hits = 0
    field_hits = 0
    mrr_sum = 0.0

    by_dataset: dict[str, dict[str, float]] = {}

    for f in files:
        data = _load_json(f)
        if not isinstance(data, dict):
            continue
        total += 1

        dataset = str(data.get("dataset") or "unknown")
        ds = by_dataset.setdefault(
            dataset,
            {
                "total": 0,
                "mapped": 0,
                "entity_hits": 0,
                "field_hits": 0,
                "mrr_sum": 0.0,
            },
        )
        ds["total"] += 1

        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        target_count = int(metrics.get("target_entity_count") or 0)
        if target_count <= 0:
            continue

        mapped += 1
        ds["mapped"] += 1

        if bool(metrics.get("hit_entity_at_k")):
            entity_hits += 1
            ds["entity_hits"] += 1
        if bool(metrics.get("hit_field_at_k")):
            field_hits += 1
            ds["field_hits"] += 1

        mrr = float(metrics.get("mrr_entity") or 0.0)
        mrr_sum += mrr
        ds["mrr_sum"] += mrr

    summary = {
        "results_dir": str(results_dir),
        "question_files": total,
        "mapped_questions": mapped,
        "entity_hit_questions": entity_hits,
        "field_hit_questions": field_hits,
        "entity_hit_rate": (entity_hits / mapped) if mapped else None,
        "field_hit_rate": (field_hits / mapped) if mapped else None,
        "entity_mrr": (mrr_sum / mapped) if mapped else None,
        "by_dataset": {},
    }

    for ds_name, ds in sorted(by_dataset.items()):
        mapped_ds = int(ds["mapped"])
        summary["by_dataset"][ds_name] = {
            "total": int(ds["total"]),
            "mapped": mapped_ds,
            "entity_hit_rate": (ds["entity_hits"] / mapped_ds) if mapped_ds else None,
            "field_hit_rate": (ds["field_hits"] / mapped_ds) if mapped_ds else None,
            "entity_mrr": (ds["mrr_sum"] / mapped_ds) if mapped_ds else None,
        }

    print("[vector_benchmark_eval]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
