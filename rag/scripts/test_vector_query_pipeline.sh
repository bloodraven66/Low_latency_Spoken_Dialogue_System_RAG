#!/bin/bash

set -euo pipefail

RAG_ROOT="/mnt/matylda4/udupa/exps/RAG"
PYTHON_BIN="${PYTHON_BIN:-$RAG_ROOT/.venv/bin/python}"
EMBEDDINGS_ROOT="${EMBEDDINGS_ROOT:-$RAG_ROOT/embeddings}"
EMBEDDING_DATA_NAME="${EMBEDDING_DATA_NAME:-intfloat-multilingual-e5-base__cs384_ov64_min20_hdr0}"
VECTOR_DIR="${VECTOR_DIR:-$EMBEDDINGS_ROOT/$EMBEDDING_DATA_NAME}"
TOP_K="${TOP_K:-5}"

QUESTION="${1:-What does BAYa cover in machine learning?}"
QUERY_BUILDER="${QUERY_BUILDER:-identity}"
QUERY_BUILDER_CMD="${QUERY_BUILDER_CMD:-}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: Python binary not executable: $PYTHON_BIN"
  exit 1
fi

echo "[vector_test]"
echo "question=$QUESTION"
echo "query_builder=$QUERY_BUILDER"
echo "vector_dir=$VECTOR_DIR"

CMD=(
  "$PYTHON_BIN"
  "$RAG_ROOT/scripts/query_vector_index.py"
  "--vector_dir" "$VECTOR_DIR"
  "--question" "$QUESTION"
  "--query_builder" "$QUERY_BUILDER"
  "--top_k" "$TOP_K"
)

if [ -n "$QUERY_BUILDER_CMD" ]; then
  CMD+=("--query_builder_cmd" "$QUERY_BUILDER_CMD")
fi

PYTHONPATH="$RAG_ROOT" "${CMD[@]}"
