#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   ./dockerless/start_rag.sh
#   HOST=127.0.0.1 PORT=8095 ./dockerless/start_rag.sh
#
# To forward RAG port from a remote node to localhost (run on the backend machine):
#   ssh -L 8095:localhost:8095 pcgpu2.fit.vutbr.cz -N

# By default this launcher runs inside conda env `.faiss`.
# Override behavior with:
#   USE_CONDA=0 PYTHON_BIN=python3.12 ./dockerless/start_rag.sh ...
#   CONDA_ENV=my_env ./dockerless/start_rag.sh ...
#
# Vector index selection is STRICT here (session-fixed):
#   default -> <repo>/rag_data
#   optional override -> RAG_VECTOR_DIR=/abs/path ./dockerless/start_rag.sh

PYTHON_BIN="${PYTHON_BIN:-/mnt/matylda4/udupa/miniconda3/envs/.faiss/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8095}"
USE_CONDA="${USE_CONDA:-0}"
CONDA_ENV="${CONDA_ENV:-.faiss}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAG_VECTOR_DIR="${RAG_VECTOR_DIR:-${REPO_ROOT}/rag_data}"

for arg in "$@"; do
	case "${arg}" in
		--vector_dir|--vector_dir=*|--embedding_data_name|--embedding_data_name=*|--embeddings_root|--embeddings_root=*)
			echo "Error: vector/index args are not accepted by start_rag.sh; set RAG_VECTOR_DIR env var instead." >&2
			exit 2
			;;
	esac
done

if [ "${USE_CONDA}" = "1" ]; then
	exec conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/dockerless/rag_inference_server.py" --host "${HOST}" --port "${PORT}" --vector_dir "${RAG_VECTOR_DIR}" "$@"
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/dockerless/rag_inference_server.py" --host "${HOST}" --port "${PORT}" --vector_dir "${RAG_VECTOR_DIR}" "$@"
