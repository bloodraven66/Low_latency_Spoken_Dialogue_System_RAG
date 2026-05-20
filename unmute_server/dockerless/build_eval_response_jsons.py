"""
Bridge script: convert unmute FIT RAG benchmark results into the response_jsons
format expected by /mnt/matylda4/udupa/exps/RAG/scripts/eval_rag.py.

Output layout (mirrors eval_rag.py conventions):
  <results_root>/<embedding_data_name>/<query_model_name>/<gen_llm_name>/
      response_jsons/
          <category>/
              <item_id>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_benchmark_items(benchmark_dir: Path) -> dict[str, dict]:
    """Return {item_id: item_dict} for every item across all benchmark JSONs."""
    items: dict[str, dict] = {}
    for json_file in sorted(benchmark_dir.rglob("*.json")):
        category = json_file.parent.name
        json_type = json_file.stem
        data = json.loads(json_file.read_text(encoding="utf-8"))
        for item in data.get("items", []):
            item_id = item["id"]
            items[item_id] = {
                "item_id": item_id,
                "question": item.get("question", ""),
                "expected_answer": item.get("expected_answer", ""),
                "dataset": category,
                "topic": item.get("type", json_type),
                "json_type": json_type,
            }
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build response_jsons/ from unmute FIT RAG benchmark outputs for eval_rag.py"
    )
    parser.add_argument(
        "--unmute_results_dir",
        type=str,
        required=True,
        help="Path to the unmute model results dir, e.g. results_tmp/fit_rag_benchmark/unmute_rag_gemma3_12b",
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default="/mnt/matylda4/udupa/exps/RAG/FIT_RAG_Benchmark_with_audio",
        help="Root of the FIT RAG benchmark (contains courses/, groups/, etc.)",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="eval_results",
        help="Output root passed to eval_rag.py via --results_root",
    )
    parser.add_argument(
        "--embedding_data_name",
        type=str,
        default="fit",
        help="eval_rag.py --embedding_data_name",
    )
    parser.add_argument(
        "--query_model_name",
        type=str,
        default="unmute",
        help="eval_rag.py --query_model_name",
    )
    parser.add_argument(
        "--gen_llm_name",
        type=str,
        required=True,
        help="eval_rag.py --gen_llm_name, e.g. gemma3_12b",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip items with no output.json instead of erroring",
    )
    args = parser.parse_args()

    unmute_dir = Path(args.unmute_results_dir).resolve()
    benchmark_dir = Path(args.benchmark_dir).resolve()
    out_root = (
        Path(args.results_root).resolve()
        / args.embedding_data_name
        / args.query_model_name
        / args.gen_llm_name
        / "response_jsons"
    )

    print(f"Loading benchmark items from {benchmark_dir} ...")
    all_items = load_benchmark_items(benchmark_dir)
    print(f"  {len(all_items)} items found in benchmark")

    written = 0
    skipped = 0
    missing_output = 0

    for item_id, meta in sorted(all_items.items()):
        item_dir = unmute_dir / item_id
        output_json = item_dir / "output.json"

        if not output_json.exists():
            if args.skip_missing:
                missing_output += 1
                continue
            raise FileNotFoundError(
                f"output.json not found for {item_id} at {output_json}. "
                "Run fd_asr.py first, or use --skip_missing."
            )

        asr = json.loads(output_json.read_text(encoding="utf-8"))
        bot_text = asr.get("text", "").strip()

        record = {
            "item_id": item_id,
            "question": meta["question"],
            "expected_answer": meta["expected_answer"],
            "dataset": meta["dataset"],
            "topic": meta["topic"],
            "json_type": meta["json_type"],
            # eval_rag.py reads llm_answers.top1_answer and top5_answer;
            # unmute produces a single response so both are set to the same text.
            "llm_answers": {
                "top1_answer": bot_text,
                "top5_answer": bot_text,
            },
            "asr_bot_text": bot_text,
        }

        category_dir = out_root / meta["dataset"]
        category_dir.mkdir(parents=True, exist_ok=True)
        out_file = category_dir / f"{item_id}.json"
        out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1

    if missing_output:
        skipped += missing_output
        print(f"  Skipped {missing_output} items with no output.json")

    print(f"Done. Wrote {written} response JSONs to {out_root}")
    print()
    print("Next — run eval_rag.py:")
    print(
        f"  python3.12 /mnt/matylda4/udupa/exps/RAG/scripts/eval_rag.py \\\n"
        f"    --results_root {args.results_root} \\\n"
        f"    --embedding_data_name {args.embedding_data_name} \\\n"
        f"    --query_model_name {args.query_model_name} \\\n"
        f"    --gen_llm_name {args.gen_llm_name} \\\n"
        f"    --backend mock"
    )


if __name__ == "__main__":
    main()
