#!/bin/bash

# set -e
echo "--------------------------------------------------" 
echo "Current time: $(date)"
start_time=$(date +%s)

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HOME="/mnt/matylda4/udupa/hugging-face"
export WANDB_MODE="offline"

# ------------------------- stage flags (edit these) -------------------------
# requested style: true/false flags at the top
EXTRACT_EMBED="false"
GET_RESPONSES="true"
SCORE_RESPONSES="true"

# ------------------------- pipeline config -------------------------
ngpus=4

# Embedding experiment presets (same chunks, different embedding model)
# Chunking profile flags (choose exactly one)
USE_CHUNK_BASIC="false"
USE_CHUNK_BASIC_V2="false"
USE_CHUNK_BASIC_CTX="true"

case "$USE_CHUNK_BASIC" in
  1|true|TRUE|yes|YES|on|ON) _use_basic=1 ;;
  *) _use_basic=0 ;;
esac
case "$USE_CHUNK_BASIC_V2" in
  1|true|TRUE|yes|YES|on|ON) _use_basic_v2=1 ;;
  *) _use_basic_v2=0 ;;
esac
case "$USE_CHUNK_BASIC_CTX" in
  1|true|TRUE|yes|YES|on|ON) _use_basic_ctx=1 ;;
  *) _use_basic_ctx=0 ;;
esac

selected_chunk_profiles=$((_use_basic + _use_basic_v2 + _use_basic_ctx))
if [ "$selected_chunk_profiles" -gt 1 ]; then
  echo "[run.sh] Invalid chunk profile flags: set only one of USE_CHUNK_BASIC/USE_CHUNK_BASIC_V2/USE_CHUNK_BASIC_CTX to true."
  exit 1
fi
if [ "$selected_chunk_profiles" -eq 0 ]; then
  echo "[run.sh] Invalid chunk profile flags: set one of USE_CHUNK_BASIC/USE_CHUNK_BASIC_V2/USE_CHUNK_BASIC_CTX to true."
  exit 1
fi

if [ "$_use_basic" -eq 1 ]; then
  CHUNK_IMPL="basic"
elif [ "$_use_basic_v2" -eq 1 ]; then
  CHUNK_IMPL="basic_v2"
else
  CHUNK_IMPL="basic_ctx"
fi

CHUNKS_DIR="/mnt/matylda4/udupa/exps/RAG/embeddings/chunks"
CHUNKS_JSONL="${CHUNKS_DIR}/${CHUNK_IMPL}.jsonl"
CHUNK_IDS_JSON="${CHUNKS_DIR}/${CHUNK_IMPL}_ids.json"
CHUNK_CONFIG_JSON="${CHUNKS_DIR}/${CHUNK_IMPL}_generation_config.json"
CHUNK_SIZE=384
CHUNK_OVERLAP=64
CHUNK_MIN_TOKENS=20
CHUNK_INCLUDE_HEADER="true"
CHUNK_V2_SHORT_MIN_TOKENS=1
CHUNK_V2_KEEP_NOISY_FIELDS="false"
CHUNK_CTX_MAX_TOKENS=48

EMB_MODEL="BAAI/bge-large-en-v1.5"
EMB_BATCH_SIZE=8
EMB_MAX_LENGTH=512

# Optional explicit override; when empty, EMB_NAME is derived from EMB_MODEL + chunk params.
EMB_NAME_OVERRIDE=""

EMB_MODEL_TOKEN=$(echo "$EMB_MODEL" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/-+/-/g; s/^-+//; s/-+$//')
if [ -z "$EMB_MODEL_TOKEN" ]; then
  EMB_MODEL_TOKEN="unknown"
fi

CHUNK_IMPL_TOKEN=$(echo "$CHUNK_IMPL" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/-+/-/g; s/^-+//; s/-+$//')
if [ -z "$CHUNK_IMPL_TOKEN" ]; then
  CHUNK_IMPL_TOKEN="basic"
fi

case "$CHUNK_INCLUDE_HEADER" in
  1|true|TRUE|yes|YES|on|ON)
  CHUNK_HEADER_TOKEN="hdr1"
    ;;
  *)
  CHUNK_HEADER_TOKEN="hdr0"
    ;;
esac

EMB_NAME_DERIVED="${EMB_MODEL_TOKEN}__${CHUNK_IMPL_TOKEN}__cs${CHUNK_SIZE}_ov${CHUNK_OVERLAP}_min${CHUNK_MIN_TOKENS}_${CHUNK_HEADER_TOKEN}"
if [ -n "$EMB_NAME_OVERRIDE" ]; then
  EMB_NAME="$EMB_NAME_OVERRIDE"
else
  EMB_NAME="$EMB_NAME_DERIVED"
fi

REBUILD_EMBEDDINGS=0
REBUILD_CHUNKS=0

QUERY_MODEL_NAME="raw"
GEN_BACKEND="vllm"
GEN_MODEL="google/gemma-3-12b-it"
GEN_LLM_NAME="${GEN_BACKEND}__google_gemma_3_12b_it"

JUDGE_BACKEND="vllm"
PROM_MODEL="Qwen/Qwen3.5-4B"
PROM_REASONING_PARSER="qwen3"
PROM_LANGUAGE_MODEL_ONLY=1
PROM_DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
PROM_MAX_LEN=2048
PROM_GPU_UTIL=0.95
PROM_DTYPE="half"
PROM_MAX_NUM_SEQS=16

# Force overwrite/recompute controls
FORCE_RETRIEVAL="false"
FORCE_GENERATION="false"
FORCE_EVAL=1

# ------------------------- helpers -------------------------
is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

slugify() {
  echo "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^[._-]+//; s/[._-]+$//; s/^$/unknown/'
}

# ------------------------- derived paths -------------------------
RESULTS_ROOT="FIT_RAG_Benchmark_results"
LLM_ROOT="${RESULTS_ROOT}/${EMB_NAME}/${QUERY_MODEL_NAME}/${GEN_LLM_NAME}"
RETRIEVED_ROOT="${RESULTS_ROOT}/${EMB_NAME}/${QUERY_MODEL_NAME}/retrieved_jsons"
RESPONSE_ROOT="${LLM_ROOT}/response_jsons"

JUDGE_MODEL_SLUG="$(slugify "$PROM_MODEL")"
JUDGE_ID="${JUDGE_BACKEND}__${JUDGE_MODEL_SLUG}"
EVAL_JSONS_ROOT="${LLM_ROOT}/eval_jsons/${JUDGE_ID}"
EVAL_SUMMARY="${LLM_ROOT}/_eval_summary__${JUDGE_ID}.json"

echo ""
echo "=== Pipeline paths ==="
echo "  Gen model      : ${GEN_MODEL}"
echo "  Judge model    : ${PROM_MODEL}"
echo "  Retrieved jsons: ${RETRIEVED_ROOT}"
echo "  Response jsons : ${RESPONSE_ROOT}"
echo "  Eval jsons     : ${EVAL_JSONS_ROOT}"
echo "  Score summary  : ${EVAL_SUMMARY}"
echo "======================================"
echo ""

EMB_CONFIG_PATH="/mnt/matylda4/udupa/exps/RAG/embeddings/${EMB_NAME}/generation_config.json"
if is_true "$EXTRACT_EMBED" && [ "$REBUILD_EMBEDDINGS" -ne 1 ] && [ ! -f "$EMB_CONFIG_PATH" ]; then
  echo "[run.sh] Missing embedding artifacts for EMB_NAME=${EMB_NAME}"
  echo "[run.sh] Auto-enabling REBUILD_EMBEDDINGS=1 so generation_config is created."
  REBUILD_EMBEDDINGS=1
fi

# ------------------------- GPU allocation only when needed -------------------------
NEED_GPU="false"

if is_true "$GET_RESPONSES"; then
  NEED_GPU="true"
fi

if is_true "$EXTRACT_EMBED" && [ "$REBUILD_EMBEDDINGS" -eq 1 ]; then
  NEED_GPU="true"
fi

if is_true "$SCORE_RESPONSES"; then
  if [ "$FORCE_EVAL" -eq 1 ] || [ ! -d "$EVAL_JSONS_ROOT" ] || ! find "$EVAL_JSONS_ROOT" -type f -name '*.json' | grep -q .; then
    NEED_GPU="true"
  fi
fi

if is_true "$NEED_GPU"; then
  export CUDA_VISIBLE_DEVICES=$(/mnt/matylda4/udupa/exps/archive/NLP-project-whisper/sge_utils/free-gpus.sh $ngpus) || {
    echo "Could not obtain GPU."
    exit 1
  }
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
  echo "[run.sh] Skipping GPU allocation (enabled stages can run without it)."
fi
echo "hostname: $(hostname)"

# ------------------------- stage: extract_embed -------------------------
if is_true "$EXTRACT_EMBED"; then
  echo "[run.sh] Stage enabled: extract_embed"

  if is_true "$REBUILD_CHUNKS"; then
    CHUNK_ARGS=(
      --clean_root extracted_data_clean/fit
      --embeddings_root /mnt/matylda4/udupa/exps/RAG/embeddings
      --embedding_model_name "$EMB_MODEL"
      --output "$CHUNKS_JSONL"
      --config_output "$CHUNK_CONFIG_JSON"
      --chunk_impl "$CHUNK_IMPL"
      --chunk_size "$CHUNK_SIZE"
      --chunk_overlap "$CHUNK_OVERLAP"
      --min_tokens "$CHUNK_MIN_TOKENS"
      --v2_short_field_min_tokens "$CHUNK_V2_SHORT_MIN_TOKENS"
      --context_max_tokens "$CHUNK_CTX_MAX_TOKENS"
    )
    if is_true "$CHUNK_V2_KEEP_NOISY_FIELDS"; then
      CHUNK_ARGS+=(--v2_keep_noisy_fields)
    fi
    if is_true "$CHUNK_INCLUDE_HEADER"; then
      CHUNK_ARGS+=(--include_metadata_header)
    fi
    PYTHONPATH=/mnt/matylda4/udupa/exps/RAG /mnt/matylda4/udupa/exps/RAG/.venv/bin/python /mnt/matylda4/udupa/exps/RAG/scripts/build_vector_chunks.py "${CHUNK_ARGS[@]}"
  fi

  if [ "$REBUILD_EMBEDDINGS" -eq 1 ]; then
    SOURCE_CHUNKS="$CHUNKS_JSONL"
    TARGET_EMB_DIR="/mnt/matylda4/udupa/exps/RAG/embeddings/${EMB_NAME}"

    if [ ! -f "$SOURCE_CHUNKS" ]; then
      echo "[run.sh] Missing source chunks: $SOURCE_CHUNKS"
      exit 1
    fi

    echo "[run.sh] Rebuilding embeddings with model: $EMB_MODEL"
    PYTHONPATH=/mnt/matylda4/udupa/exps/RAG /mnt/matylda4/udupa/exps/RAG/.venv/bin/python /mnt/matylda4/udupa/exps/RAG/scripts/build_vector_index.py \
      --embeddings_root /mnt/matylda4/udupa/exps/RAG/embeddings \
      --chunks "$SOURCE_CHUNKS" \
      --out_dir "$TARGET_EMB_DIR" \
      --chunk_ids_output "$CHUNK_IDS_JSON" \
      --embedding_data_name "$EMB_NAME" \
      --embedding_backend transformers \
      --embedding_model "$EMB_MODEL" \
      --batch_size "$EMB_BATCH_SIZE" \
      --max_length "$EMB_MAX_LENGTH" \
      --normalize
  fi

  if [ ! -f "$EMB_CONFIG_PATH" ]; then
    echo "[run.sh] Missing embedding config: $EMB_CONFIG_PATH"
    echo "[run.sh] Embedding artifacts are not ready yet. Set REBUILD_EMBEDDINGS=1 and rerun extract_embed."
    exit 1
  fi

  RETRIEVE_ARGS=(
    --benchmark_root FIT_RAG_Benchmark
    --output_root "$RESULTS_ROOT"
    --embeddings_root embeddings
    --embedding_data_name "$EMB_NAME"
    --query_builder identity
    --top_k 5
  )
  if is_true "$FORCE_RETRIEVAL"; then
    RETRIEVE_ARGS+=(--overwrite)
  fi

  PYTHONPATH=/mnt/matylda4/udupa/exps/RAG /mnt/matylda4/udupa/exps/RAG/.venv/bin/python scripts/retrieve_from_question.py "${RETRIEVE_ARGS[@]}"
else
  echo "[run.sh] Stage disabled: extract_embed"
fi

# ------------------------- stage: get_responses -------------------------
if is_true "$GET_RESPONSES"; then
  echo "[run.sh] Stage enabled: get_responses"

  if [ ! -d "$RETRIEVED_ROOT" ] || ! find "$RETRIEVED_ROOT" -type f -name '*.json' | grep -q .; then
    echo "[run.sh] Missing retrieved_jsons at: $RETRIEVED_ROOT"
    echo "[run.sh] Enable EXTRACT_EMBED=true first, or provide retrieval outputs."
    exit 1
  fi

  GENERATE_ARGS=(
    --results_root "$RESULTS_ROOT"
    --embedding_data_name "$EMB_NAME"
    --query_model_name "$QUERY_MODEL_NAME"
    --backend "$GEN_BACKEND"
    --model "$GEN_MODEL"
    --llm_tensor_parallel_size "$ngpus"
    --llm_dtype bfloat16
    --llm_max_model_len 8192
    --llm_gpu_memory_utilization 0.7
  )
  if is_true "$FORCE_GENERATION"; then
    GENERATE_ARGS+=(--overwrite)
  fi

  PYTHONPATH=/mnt/matylda4/udupa/exps/RAG /mnt/matylda4/udupa/exps/RAG/.venv/bin/python scripts/generate_from_question_and_retrieved.py "${GENERATE_ARGS[@]}"
else
  echo "[run.sh] Stage disabled: get_responses"
fi

# ------------------------- stage: score_responses -------------------------
if is_true "$SCORE_RESPONSES"; then
  echo "[run.sh] Stage enabled: score_responses"

  if [ ! -d "$RESPONSE_ROOT" ] || ! find "$RESPONSE_ROOT" -type f -name '*.json' | grep -q .; then
    echo "[run.sh] Missing response_jsons at: $RESPONSE_ROOT"
    echo "[run.sh] Enable GET_RESPONSES=true first, or provide response outputs."
    exit 1
  fi

  EVAL_EXTRA_ARGS=()
  if [ "$PROM_LANGUAGE_MODEL_ONLY" -eq 1 ]; then
    EVAL_EXTRA_ARGS+=(--vllm_language_model_only)
  fi
  if [ -n "$PROM_REASONING_PARSER" ]; then
    EVAL_EXTRA_ARGS+=(--vllm_reasoning_parser "$PROM_REASONING_PARSER")
  fi
  if [ -n "$PROM_DEFAULT_CHAT_TEMPLATE_KWARGS" ]; then
    EVAL_EXTRA_ARGS+=(--vllm_default_chat_template_kwargs "$PROM_DEFAULT_CHAT_TEMPLATE_KWARGS")
  fi

  if [ "$FORCE_EVAL" -ne 1 ] && [ -d "$EVAL_JSONS_ROOT" ] && find "$EVAL_JSONS_ROOT" -type f -name '*.json' | grep -q .; then
    echo "[run.sh] Found existing eval outputs at: $EVAL_JSONS_ROOT"
    echo "[run.sh] Skipping eval rerun and reporting scores from existing files."
  else
    PYTHONPATH=/mnt/matylda4/udupa/exps/RAG /mnt/matylda4/udupa/exps/RAG/.venv/bin/python scripts/eval_rag.py \
      --results_root "$RESULTS_ROOT" \
      --embedding_data_name "$EMB_NAME" \
      --query_model_name "$QUERY_MODEL_NAME" \
      --gen_llm_name "$GEN_LLM_NAME" \
      --backend "$JUDGE_BACKEND" \
      --model "$PROM_MODEL" \
      --vllm_tensor_parallel_size "$ngpus" \
      --vllm_pipeline_parallel_size 1 \
      --vllm_enforce_eager \
      --vllm_dtype "$PROM_DTYPE" \
      --vllm_gpu_memory_utilization "$PROM_GPU_UTIL" \
      --vllm_max_model_len "$PROM_MAX_LEN" \
      --vllm_max_num_seqs "$PROM_MAX_NUM_SEQS" \
      "${EVAL_EXTRA_ARGS[@]}"
  fi

  # Always print final scores after eval (fresh or skipped)
  if [ -d "$EVAL_JSONS_ROOT" ] && find "$EVAL_JSONS_ROOT" -type f -name '*.json' | grep -q .; then
    PYTHONPATH=/mnt/matylda4/udupa/exps/RAG /mnt/matylda4/udupa/exps/RAG/.venv/bin/python /mnt/matylda4/udupa/exps/RAG/scripts/score_rag.py \
      --eval_jsons_root "$EVAL_JSONS_ROOT"
  else
    echo "[run.sh] No eval outputs found at: $EVAL_JSONS_ROOT — score_rag skipped."
  fi
else
  echo "[run.sh] Stage disabled: score_responses"
fi

echo "Job finished at: $(date)"
end_time=$(date +%s)
time_taken_minutes=$(echo "scale=2; ($end_time - $start_time) / 60" | bc)
echo "Time taken: $time_taken_minutes minutes"
