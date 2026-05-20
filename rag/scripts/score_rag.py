from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def mean_or_none(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def dataset_to_category(dataset: str) -> str:
    if dataset.startswith("FIT_"):
        rem = dataset[4:]
        group = rem.split("_", 1)[0] if rem else dataset
    else:
        group = dataset.split("_", 1)[0]
    return group.lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize existing eval_jsons into overall, by-dataset, and by-category means."
    )
    parser.add_argument(
        "--eval_jsons_root",
        type=str,
        required=True,
        help="Path to eval_jsons/<judge_id> directory containing per-question eval JSON files.",
    )
    args = parser.parse_args()

    root = Path(args.eval_jsons_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--eval_jsons_root is not a directory: {root}")

    files = sorted(p for p in root.rglob("*.json") if p.is_file())
    if not files:
        raise RuntimeError(f"No eval json files found under: {root}")

    overall_top1: list[float] = []
    overall_top5: list[float] = []
    by_dataset_top1: dict[str, list[float]] = defaultdict(list)
    by_dataset_top5: dict[str, list[float]] = defaultdict(list)
    by_category_top1: dict[str, list[float]] = defaultdict(list)
    by_category_top5: dict[str, list[float]] = defaultdict(list)

    for f in files:
        row = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(row, dict):
            raise RuntimeError(f"Eval row is not an object: {f}")

        dataset = str(row.get("dataset") or f.parent.name)
        try:
            top1 = float(row["top1_score"])
            top5 = float(row["top5_score"])
        except Exception as exc:
            raise RuntimeError(f"Missing/invalid top1_score or top5_score in {f}") from exc

        category = dataset_to_category(dataset)

        overall_top1.append(top1)
        overall_top5.append(top5)
        by_dataset_top1[dataset].append(top1)
        by_dataset_top5[dataset].append(top5)
        by_category_top1[category].append(top1)
        by_category_top5[category].append(top5)

    summary = {
        "eval_jsons_root": str(root),
        "files": len(files),
        "overall": {
            "top1_mean": mean_or_none(overall_top1),
            "top5_mean": mean_or_none(overall_top5),
        },
        "by_dataset": {
            d: {
                "n": len(by_dataset_top1[d]),
                "top1_mean": mean_or_none(by_dataset_top1[d]),
                "top5_mean": mean_or_none(by_dataset_top5[d]),
            }
            for d in sorted(by_dataset_top1)
        },
        "by_category": {
            g: {
                "n": len(by_category_top1[g]),
                "top1_mean": mean_or_none(by_category_top1[g]),
                "top5_mean": mean_or_none(by_category_top5[g]),
            }
            for g in sorted(by_category_top1)
        },
    }

    print("[existing_eval_scores]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
