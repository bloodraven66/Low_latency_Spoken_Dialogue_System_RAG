from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from utils.common import load_json


def _sanitize_name_token(value: str) -> str:
    value = (value or "").strip().lower()
    out = []
    for ch in value:
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            out.append(ch)
        else:
            out.append("-")
    token = "".join(out)
    while "--" in token:
        token = token.replace("--", "-")
    token = token.strip("-")
    return token or "unknown"


def _infer_embedding_data_name(model_name: str, chunks_path: Path) -> str:
    if chunks_path.parent.name:
        return chunks_path.parent.name
    return _sanitize_name_token(model_name)


def _read_chunks_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _progress_iter(iterable: Iterable[Any], total: int, desc: str, enabled: bool) -> Iterable[Any]:
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc, leave=False)
    except Exception:
        # Fallback without tqdm dependency.
        print(f"[{desc}] progress enabled (tqdm unavailable, using periodic logging)")

        def _fallback_gen() -> Iterable[Any]:
            last_percent = -1
            for i, item in enumerate(iterable, start=1):
                percent = int((i * 100) / max(total, 1))
                if percent % 10 == 0 and percent != last_percent:
                    print(f"[{desc}] {percent}% ({i}/{total})")
                    last_percent = percent
                yield item

        return _fallback_gen()


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


def _embed_hash(texts: list[str], dim: int, show_progress: bool = False) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    iterator = _progress_iter(enumerate(texts), total=len(texts), desc="Embedding (hash)", enabled=show_progress)
    for i, t in iterator:
        out[i] = _hash_embedding(t, dim)
    return out


def _embed_transformers(
    texts: list[str],
    model_name: str,
    max_length: int,
    batch_size: int,
    show_progress: bool = False,
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers+torch are required for --embedding_backend transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rows: list[np.ndarray] = []
    starts = list(range(0, len(texts), batch_size))
    with torch.no_grad():
        iterator = _progress_iter(starts, total=len(starts), desc="Embedding (transformers)", enabled=show_progress)
        for i in iterator:
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
            hidden = out.last_hidden_state  # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
            rows.append(pooled.detach().cpu().numpy().astype(np.float32))

    emb = np.concatenate(rows, axis=0) if rows else np.zeros((0, 768), dtype=np.float32)
    return emb


def _load_generation_config(path: Path) -> dict[str, Any]:
    if path.is_file():
        data = load_json(str(path))
        if isinstance(data, dict):
            return data
    return {}


def _write_generation_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_index_faiss(embeddings: np.ndarray, out_path: Path) -> bool:
    try:
        import faiss  # type: ignore
    except ImportError:
        return False

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(out_path))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector embeddings/index from chunk JSONL.")
    parser.add_argument("--embeddings_root", type=str, default="embeddings")
    parser.add_argument("--embedding_data_name", type=str, default=None)
    parser.add_argument("--chunks", type=str, default=None, help="Optional explicit chunks.jsonl path")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional explicit output directory")
    parser.add_argument(
        "--chunk_ids_output",
        type=str,
        default=None,
        help="Optional explicit output path for chunk IDs JSON (defaults to <out_dir>/chunk_ids.json).",
    )
    parser.add_argument("--embedding_backend", type=str, default="hash", choices=["hash", "transformers"])
    parser.add_argument("--embedding_model", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--normalize", action="store_true", help="L2 normalize embeddings for cosine/IP retrieval")
    parser.add_argument("--no_progress", action="store_true", help="Disable embedding progress display")
    parser.add_argument("--build_faiss", action="store_true", help="Build FAISS flat IP index if faiss is installed")
    args = parser.parse_args()

    embeddings_root = Path(args.embeddings_root).resolve()
    if args.chunks:
        chunks_path = Path(args.chunks).resolve()
        embedding_data_name = args.embedding_data_name or _infer_embedding_data_name(args.embedding_model, chunks_path)
    else:
        if not args.embedding_data_name:
            raise ValueError("Provide --embedding_data_name when --chunks is not specified")
        embedding_data_name = args.embedding_data_name
        chunks_path = embeddings_root / embedding_data_name / "chunks.jsonl"

    out_dir = Path(args.out_dir).resolve() if args.out_dir else chunks_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_chunks_jsonl(chunks_path)
    texts = [str(r.get("embedding_text", "")) for r in rows]
    chunk_ids = [str(r.get("chunk_id", f"chunk-{i}")) for i, r in enumerate(rows)]
    show_progress = not args.no_progress

    if args.embedding_backend == "hash":
        embeddings = _embed_hash(texts, dim=args.embedding_dim, show_progress=show_progress)
        effective_model = "hash-blake2b"
    else:
        embeddings = _embed_transformers(
            texts,
            model_name=args.embedding_model,
            max_length=args.max_length,
            batch_size=args.batch_size,
            show_progress=show_progress,
        )
        effective_model = args.embedding_model

    if args.normalize:
        embeddings = _normalize_rows(embeddings)

    emb_path = out_dir / "embeddings.npy"
    ids_path = Path(args.chunk_ids_output).resolve() if args.chunk_ids_output else (out_dir / "chunk_ids.json")
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    ids_path.write_text(json.dumps(chunk_ids, ensure_ascii=False, indent=2), encoding="utf-8")

    faiss_path = out_dir / "index.faiss"
    faiss_built = False
    if args.build_faiss:
        faiss_built = _build_index_faiss(embeddings, faiss_path)

    config_path = out_dir / "generation_config.json"
    generation_config = _load_generation_config(config_path)
    generation_config["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    generation_config["embedding_data_name"] = embedding_data_name
    generation_config["artifacts"] = {
        **(generation_config.get("artifacts") or {}),
        "chunks_jsonl": str(chunks_path),
        "embeddings_npy": str(emb_path),
        "chunk_ids_json": str(ids_path),
        "faiss_index": str(faiss_path) if faiss_built else None,
    }
    generation_config["embedding"] = {
        "backend": args.embedding_backend,
        "model_name": effective_model,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] > 0 else args.embedding_dim,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "normalize": bool(args.normalize),
        "num_vectors": int(embeddings.shape[0]),
    }
    generation_config["index"] = {
        "engine": "faiss-flatip" if faiss_built else "numpy-bruteforce",
        "metric": "cosine_or_ip" if args.normalize else "inner_product",
        "faiss_requested": bool(args.build_faiss),
        "faiss_built": bool(faiss_built),
    }

    _write_generation_config(config_path, generation_config)

    print("[vector_index_build]")
    print(
        json.dumps(
            {
                "embedding_data_name": embedding_data_name,
                "chunks": str(chunks_path),
                "out_dir": str(out_dir),
                "num_chunks": len(rows),
                "embedding_backend": args.embedding_backend,
                "embedding_shape": list(embeddings.shape),
                "normalize": bool(args.normalize),
                "faiss_built": bool(faiss_built),
                "generation_config": str(config_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
