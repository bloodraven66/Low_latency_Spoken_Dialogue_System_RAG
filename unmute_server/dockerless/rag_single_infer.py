from __future__ import annotations

import argparse
import hashlib
import json
import site
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def _hash_embedding(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.split()
    if not tokens:
        return vec

    for tok in tokens:
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=16).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if (h[4] % 2 == 0) else -1.0
        weight = 1.0 + (h[5] / 255.0)
        vec[idx] += sign * weight

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _embed_transformers(
    texts: list[str],
    model_name: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    user_site = site.getusersitepackages()
    if user_site and user_site in sys.path:
        sys.path = [p for p in sys.path if p != user_site]

    tokenizer, model, device = _get_transformers_encoder(model_name)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "transformers+torch are required for transformers embeddings. "
            "If they are installed in conda but import still fails, run with PYTHONNOUSERSITE=1 "
            "to avoid ~/.local package conflicts."
        ) from exc

    rows: list[np.ndarray] = []
    starts = list(range(0, len(texts), batch_size))
    with torch.no_grad():
        for i in starts:
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
            rows.append(pooled.detach().cpu().numpy().astype(np.float32))

    emb = np.concatenate(rows, axis=0) if rows else np.zeros((0, 768), dtype=np.float32)
    return emb


@lru_cache(maxsize=4)
def _get_transformers_encoder(model_name: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers+torch are required for transformers embeddings. "
            "If they are installed in conda but import still fails, run with PYTHONNOUSERSITE=1 "
            "to avoid ~/.local package conflicts."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


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
    return q.astype(np.float32)


def _resolve_vector_dir(args: argparse.Namespace) -> Path:
    if args.vector_dir:
        return Path(args.vector_dir).resolve()
    if not args.embedding_data_name:
        raise ValueError("Provide --embedding_data_name when --vector_dir is not specified")
    return (Path(args.embeddings_root).resolve() / args.embedding_data_name)


def _search_faiss(index_path: Path, q: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "faiss is not installed in this Python environment. "
            "Install faiss-cpu (recommended via conda) before running this script."
        ) from exc

    index = faiss.read_index(str(index_path))
    scores, indices = index.search(q, top_k)
    return scores[0], indices[0]


def load_json(path: Path) -> Any:
    return _load_json(path)


def load_chunks_by_id(chunks_path: Path) -> dict[str, dict[str, Any]]:
    return _load_chunks_by_id(chunks_path)


def embed_query(query_text: str, embedding_cfg: dict[str, Any]) -> np.ndarray:
    return _embed_query(query_text, embedding_cfg)


def search_faiss(index_path: Path, q: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    return _search_faiss(index_path, q, top_k)


def hash_embedding(text: str, dim: int) -> np.ndarray:
    return _hash_embedding(text, dim)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return _normalize_rows(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS-backed inference over vector chunks (top-k retrieval).")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top chunks to return")
    parser.add_argument("--vector_dir", type=str, default=None, help="Optional explicit vector artifact directory")
    parser.add_argument("--embeddings_root", type=str, default="embeddings")
    parser.add_argument("--embedding_data_name", type=str, default=None)
    args = parser.parse_args()

    query_text = (args.query or "").strip()
    if not query_text:
        raise ValueError("--query must be non-empty")
    if args.top_k <= 0:
        raise ValueError("--top_k must be > 0")

    vector_dir = _resolve_vector_dir(args)
    cfg_path = vector_dir / "generation_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing generation config: {cfg_path}")

    cfg = _load_json(cfg_path)
    artifacts = cfg.get("artifacts") or {}

    chunks_path = Path(str(artifacts.get("chunks_jsonl", vector_dir / "chunks.jsonl"))).resolve()
    ids_path = Path(str(artifacts.get("chunk_ids_json", vector_dir / "chunk_ids.json"))).resolve()
    faiss_path_raw = artifacts.get("faiss_index")
    faiss_path = Path(str(faiss_path_raw)).resolve() if faiss_path_raw else (vector_dir / "index.faiss").resolve()

    if not faiss_path.is_file():
        raise FileNotFoundError(
            f"Missing FAISS index at: {faiss_path}. "
            "Build it first with scripts/build_vector_index.py --build_faiss."
        )

    chunk_ids = _load_json(ids_path)
    if not isinstance(chunk_ids, list):
        raise RuntimeError(f"chunk_ids must be a list in {ids_path}")
    chunk_ids = [str(x) for x in chunk_ids]
    chunks_by_id = _load_chunks_by_id(chunks_path)

    embedding_cfg = cfg.get("embedding") or {}
    q = _embed_query(query_text, embedding_cfg)
    scores, indices = _search_faiss(faiss_path, q, args.top_k)

    results: list[dict[str, Any]] = []
    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= len(chunk_ids):
            continue
        chunk_id = chunk_ids[idx_i]
        row = chunks_by_id.get(chunk_id, {})
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk_id,
                "entity_type": row.get("entity_type"),
                "entity_id": row.get("entity_id"),
                "source_field": row.get("source_field"),
                "source_path": row.get("source_path"),
                "preview": str(row.get("chunk_text", ""))[:280],
            }
        )

    print("[infer_basic_ctx_faiss]")
    print(
        json.dumps(
            {
                "query": query_text,
                "top_k": args.top_k,
                "vector_dir": str(vector_dir),
                "faiss_index": str(faiss_path),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

###conda run -n .faiss python scripts/infer_basic_ctx_faiss.py --embedding_data_name baai-bge-large-en-v1-5__basic-ctx__cs384_ov64_min20_hdr1 --query "What literature is recommended for course BRIa?" --top_k 5

