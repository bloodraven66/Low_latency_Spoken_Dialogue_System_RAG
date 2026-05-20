from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

try:
    from dockerless.rag_single_infer import (
        embed_query,
        load_chunks_by_id,
        load_json,
        search_faiss,
    )
except ModuleNotFoundError:
    from rag_single_infer import (
        embed_query,
        load_chunks_by_id,
        load_json,
        search_faiss,
    )

logger = logging.getLogger("rag_server")


def _resolve_vector_dir(
    *,
    vector_dir: str | None,
    embeddings_root: str,
    embedding_data_name: str | None,
) -> Path:
    if vector_dir:
        return Path(vector_dir).resolve()
    if not embedding_data_name:
        raise ValueError("Provide either vector_dir or embedding_data_name")
    return (Path(embeddings_root).resolve() / embedding_data_name).resolve()


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Query text")
    top_k: int = Field(default=5, ge=1, le=100)

    @model_validator(mode="after")
    def _validate_location(self) -> "RetrieveRequest":
        if not self.query.strip():
            raise ValueError("query must be non-empty")
        return self


class RagServer:
    def __init__(self, default_vector_dir: str | None, default_embeddings_root: str, default_embedding_data_name: str | None):
        self.default_vector_dir = default_vector_dir
        self.default_embeddings_root = default_embeddings_root
        self.default_embedding_data_name = default_embedding_data_name
        self._artifact_cache: dict[str, dict[str, Any]] = {}

    def cached_vector_dirs(self) -> list[str]:
        return sorted(self._artifact_cache.keys())

    def _effective_vector_dir(self, payload: RetrieveRequest) -> Path:
        return _resolve_vector_dir(
            vector_dir=self.default_vector_dir,
            embeddings_root=self.default_embeddings_root,
            embedding_data_name=self.default_embedding_data_name,
        )

    def _load_artifacts(self, vector_dir: Path) -> dict[str, Any]:
        key = str(vector_dir)
        if key in self._artifact_cache:
            return self._artifact_cache[key]

        cfg_path = vector_dir / "generation_config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing generation config: {cfg_path}")

        cfg = load_json(cfg_path)
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

        chunk_ids = load_json(ids_path)
        if not isinstance(chunk_ids, list):
            raise RuntimeError(f"chunk_ids must be a list in {ids_path}")
        chunk_ids = [str(x) for x in chunk_ids]
        chunks_by_id = load_chunks_by_id(chunks_path)
        embedding_cfg = cfg.get("embedding") or {}

        artifact_obj = {
            "vector_dir": vector_dir,
            "cfg": cfg,
            "embedding_cfg": embedding_cfg,
            "chunks_by_id": chunks_by_id,
            "chunk_ids": chunk_ids,
            "faiss_path": faiss_path,
        }
        self._artifact_cache[key] = artifact_obj
        return artifact_obj

    def retrieve(self, payload: RetrieveRequest) -> dict[str, Any]:
        vector_dir = self._effective_vector_dir(payload)
        art = self._load_artifacts(vector_dir)

        q = embed_query(payload.query.strip(), art["embedding_cfg"])
        scores, indices = search_faiss(art["faiss_path"], q, payload.top_k)

        chunk_ids: list[str] = art["chunk_ids"]
        chunks_by_id: dict[str, dict[str, Any]] = art["chunks_by_id"]

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
                    "preview": str(row.get("chunk_text", ""))[:1200],
                }
            )

        return {
            "query": payload.query.strip(),
            "top_k": payload.top_k,
            "vector_dir": str(vector_dir),
            "faiss_index": str(art["faiss_path"]),
            "results": results,
        }


def build_app(
    *,
    default_vector_dir: str | None,
    default_embeddings_root: str,
    default_embedding_data_name: str | None,
) -> FastAPI:
    app = FastAPI(title="RAG Inference Server")
    rag = RagServer(default_vector_dir, default_embeddings_root, default_embedding_data_name)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        location: str | None = None
        try:
            if default_vector_dir or default_embedding_data_name:
                resolved = _resolve_vector_dir(
                    vector_dir=default_vector_dir,
                    embeddings_root=default_embeddings_root,
                    embedding_data_name=default_embedding_data_name,
                )
                location = str(resolved)
        except Exception:
            location = None

        return {
            "status": "ok",
            "service": "rag",
            "default_vector_dir": location,
            "cached_vector_dirs": rag.cached_vector_dirs(),
        }

    @app.post("/api/rag/retrieve")
    def retrieve(payload: RetrieveRequest) -> dict[str, Any]:
        try:
            return rag.retrieve(payload)
        except (ValueError,) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (FileNotFoundError,) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unhandled error during RAG retrieval")
            raise HTTPException(status_code=500, detail=f"Unhandled error: {exc}") from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG inference server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--vector_dir", type=str, default=None)
    parser.add_argument("--embeddings_root", type=str, default="embeddings")
    parser.add_argument("--embedding_data_name", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = build_app(
        default_vector_dir=args.vector_dir,
        default_embeddings_root=args.embeddings_root,
        default_embedding_data_name=args.embedding_data_name,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
