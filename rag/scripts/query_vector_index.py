from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from scripts.build_vector_index import _embed_transformers, _hash_embedding, _normalize_rows


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


def _build_query_text(question: str, query_builder: str, query_builder_cmd: str | None = None) -> tuple[str, dict[str, Any]]:
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be non-empty")

    if query_builder == "identity":
        return question, {"builder": "identity"}

    if query_builder == "external":
        if not query_builder_cmd:
            raise ValueError("--query_builder_cmd is required when --query_builder external")

        args = shlex.split(query_builder_cmd)
        env = os.environ.copy()
        env["QUESTION"] = question
        proc = subprocess.run(
            args,
            input=question,
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"external query builder failed with code {proc.returncode}: {proc.stderr.strip()}"
            )
        built = (proc.stdout or "").strip()
        if not built:
            raise RuntimeError("external query builder returned empty query text")
        return built, {
            "builder": "external",
            "command": query_builder_cmd,
        }

    raise ValueError(f"unsupported query_builder: {query_builder}")


def _embed_query(query_text: str, embedding_cfg: dict[str, Any]) -> np.ndarray:
    backend = str(embedding_cfg.get("backend", "hash"))
    if backend == "hash":
        dim = int(embedding_cfg.get("embedding_dim", 768))
        q = _hash_embedding(query_text, dim).reshape(1, -1)
    elif backend == "transformers":
        model_name = str(embedding_cfg.get("model_name", "intfloat/multilingual-e5-base"))
        max_length = int(embedding_cfg.get("max_length", 512))
        q = _embed_transformers([query_text], model_name=model_name, max_length=max_length, batch_size=1)
    else:
        raise ValueError(f"Unsupported embedding backend in generation_config.json: {backend}")

    if bool(embedding_cfg.get("normalize", False)):
        q = _normalize_rows(q)
    return q


def main() -> None:
    parser = argparse.ArgumentParser(description="Query vector chunk store with hash-embedding brute-force retrieval.")
    parser.add_argument("--query", type=str, default=None, help="Retrieval query text (if omitted, derives from --question)")
    parser.add_argument("--question", type=str, default=None, help="Natural-language question used to build query text")
    parser.add_argument(
        "--query_builder",
        type=str,
        choices=["identity", "external"],
        default="identity",
        help="How to build query text from --question. 'external' enables LLM/tool command hook.",
    )
    parser.add_argument(
        "--query_builder_cmd",
        type=str,
        default=None,
        help="External command for query building. Receives question on stdin and env QUESTION.",
    )
    parser.add_argument("--vector_dir", type=str, default=None, help="Optional explicit vector artifact directory")
    parser.add_argument("--embeddings_root", type=str, default="embeddings")
    parser.add_argument("--embedding_data_name", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    if not args.query and not args.question:
        raise ValueError("Provide either --query or --question")

    if args.vector_dir:
        vector_dir = Path(args.vector_dir).resolve()
    else:
        if not args.embedding_data_name:
            raise ValueError("Provide --embedding_data_name when --vector_dir is not specified")
        vector_dir = (Path(args.embeddings_root).resolve() / args.embedding_data_name)

    cfg = _load_json(vector_dir / "generation_config.json")
    artifacts = cfg.get("artifacts") or {}

    chunks_path = Path(str(artifacts.get("chunks_jsonl", vector_dir / "chunks.jsonl"))).resolve()
    emb_path = Path(str(artifacts.get("embeddings_npy", vector_dir / "embeddings.npy"))).resolve()
    ids_path = Path(str(artifacts.get("chunk_ids_json", vector_dir / "chunk_ids.json"))).resolve()

    emb = np.load(emb_path)
    ids = _load_json(ids_path)
    if not isinstance(ids, list):
        raise RuntimeError(f"chunk_ids must be a list in {ids_path}")

    chunks = _load_chunks_by_id(chunks_path)

    embedding_cfg = cfg.get("embedding") or {}
    query_text = args.query
    query_meta: dict[str, Any] = {"mode": "direct_query"}
    if not query_text:
        query_text, builder_meta = _build_query_text(
            question=str(args.question or ""),
            query_builder=args.query_builder,
            query_builder_cmd=args.query_builder_cmd,
        )
        query_meta = {"mode": "from_question", **builder_meta}

    q = _embed_query(query_text, embedding_cfg)

    scores = emb @ q[0]
    top_idx = np.argsort(scores)[::-1][: args.top_k]

    results = []
    for idx in top_idx:
        chunk_id = ids[int(idx)]
        row = chunks.get(chunk_id, {})
        results.append(
            {
                "rank": len(results) + 1,
                "score": float(scores[int(idx)]),
                "chunk_id": chunk_id,
                "entity_type": row.get("entity_type"),
                "entity_id": row.get("entity_id"),
                "source_field": row.get("source_field"),
                "source_path": row.get("source_path"),
                "preview": str(row.get("chunk_text", ""))[:220],
            }
        )

    print("[vector_query]")
    print(
        json.dumps(
            {
                "question": args.question,
                "query": query_text,
                "query_meta": query_meta,
                "top_k": args.top_k,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
