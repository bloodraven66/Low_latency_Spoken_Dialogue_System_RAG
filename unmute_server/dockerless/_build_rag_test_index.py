from pathlib import Path
import json
import numpy as np
import faiss

try:
    from dockerless.rag_single_infer import hash_embedding, normalize_rows
except ModuleNotFoundError:
    from rag_single_infer import hash_embedding, normalize_rows


def main() -> None:
    root = Path("results_tmp/rag_test_index").resolve()
    root.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "chunk_id": "c1",
            "entity_type": "faq",
            "entity_id": "weather",
            "source_field": "answer",
            "source_path": "toy/faqs.json",
            "chunk_text": "The weather bot can answer forecast, temperature, and rain probability questions.",
        },
        {
            "chunk_id": "c2",
            "entity_type": "faq",
            "entity_id": "books",
            "source_field": "answer",
            "source_path": "toy/faqs.json",
            "chunk_text": "Recommended literature for BRIa includes foundational AI and signal processing reading lists.",
        },
        {
            "chunk_id": "c3",
            "entity_type": "faq",
            "entity_id": "voicebot",
            "source_field": "answer",
            "source_path": "toy/faqs.json",
            "chunk_text": "Voice bot evaluation writes output.wav user.wav and output.stereo.wav artifacts.",
        },
    ]

    chunk_ids = [r["chunk_id"] for r in rows]
    texts = [r["chunk_text"] for r in rows]

    (root / "chunk_ids.json").write_text(json.dumps(chunk_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dim = 768
    emb = np.vstack([hash_embedding(t, dim) for t in texts]).astype("float32")
    emb = normalize_rows(emb).astype("float32")

    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, str(root / "index.faiss"))

    cfg = {
        "embedding": {
            "backend": "hash",
            "embedding_dim": dim,
            "normalize": True,
        },
        "artifacts": {
            "chunks_jsonl": str((root / "chunks.jsonl").resolve()),
            "chunk_ids_json": str((root / "chunk_ids.json").resolve()),
            "faiss_index": str((root / "index.faiss").resolve()),
        },
    }
    (root / "generation_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(root))


if __name__ == "__main__":
    main()
