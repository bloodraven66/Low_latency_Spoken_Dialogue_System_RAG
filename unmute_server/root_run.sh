#!/bin/bash
# run.sh - root launcher with default environment

set -e

# -------------------
# Environment and limits (constants)
# -------------------
ulimit -f unlimited
ulimit -v unlimited
ulimit -u 4096

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export CUDA_VISIBLE_DEVICES=$(/mnt/matylda4/udupa/exps/archive/NLP-project-whisper/sge_utils/free-gpus.sh 4) || {
  echo "Could not obtain GPU."
  exit 1
}
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
# -------------------
# Get per-job script from argument
# -------------------
# VENV_PYTHON="./.llm-server-venv/bin/python3"

root_path=/mnt/matylda4/udupa/exps/full_duplex/unmute

# instruction_type=smalltalk_no_starter
# instruction_type=anticipate
instruction_type=rag
# instruction_type=anticipate_rag
m=gemma3_12b

model_name=unmute_${instruction_type}_${m}_cache_test
save_folder=anticipation_exps_mar2026-v1
path="/mnt/matylda4/udupa/data/Full-Duplex-Bench-Data/"
version="v1.0"
task="candor_turn_taking"

root_dir="${root_path}/results_tmp/fd-${version}-${save_folder}/${model_name}/${task}/"
echo "Output path: ${root_dir}"

# python3 fd_asr.py --root_dir $root_dir

# ── FIT RAG Benchmark ASR transcription ─────────────────────────────────────
export PYTHONPATH="${root_path}"
fit_base="${root_path}/results_tmp/fit_rag_benchmark"

for model in unmute_base_gemma3_12b; do
    item_root="${fit_base}/${model}"
    echo ""
    echo "=== ASR: ${model} ==="

    echo "  Transcribing bot output (output.wav → output.json)..."
    python3.12 fd_asr.py --root_dir "${item_root}" --task full

    echo "  Transcribing user input (input.wav → input.json)..."
    python3.12 fd_asr.py --root_dir "${item_root}" --task full --audio_filename input.wav
done

# ── Build response_jsons + run eval_rag.py ───────────────────────────────────
eval_script="/mnt/matylda4/udupa/exps/RAG/scripts/eval_rag.py"
bridge_script="${root_path}/dockerless/build_eval_response_jsons.py"
eval_results_root="${root_path}/results_tmp/fit_rag_eval"
rag_python="/mnt/matylda4/udupa/exps/RAG/.venv/bin/python"
export PYTHONPATH="/mnt/matylda4/udupa/exps/RAG"

# Judge settings — mirrors /mnt/matylda4/udupa/exps/RAG/scripts/run.sh
JUDGE_MODEL="Qwen/Qwen3.5-4B"
JUDGE_NGPUS=2

for model in unmute_base_gemma3_12b; do
    # strip "unmute_rag_" for baseline → gemma3_12b
    # strip "unmute_"     for others  → anticipate_rag_gemma3_12b
    if [[ "${model}" == unmute_rag_* ]]; then
        gen_llm="${model#unmute_rag_}"
    else
        gen_llm="${model#unmute_}"
    fi
    unmute_dir="${fit_base}/${model}"
    echo ""
    echo "=== Eval bridge: ${model} ==="
    python3.12 "${bridge_script}" \
        --unmute_results_dir "${unmute_dir}" \
        --results_root "${eval_results_root}" \
        --embedding_data_name fit \
        --query_model_name unmute \
        --gen_llm_name "${gen_llm}" \
        --skip_missing

    echo "=== Scoring: ${model} (judge: ${JUDGE_MODEL}) ==="
    "${rag_python}" "${eval_script}" \
        --results_root "${eval_results_root}" \
        --embedding_data_name fit \
        --query_model_name unmute \
        --gen_llm_name "${gen_llm}" \
        --backend vllm \
        --model "${JUDGE_MODEL}" \
        --vllm_tensor_parallel_size "${JUDGE_NGPUS}" \
        --vllm_pipeline_parallel_size 1 \
        --vllm_enforce_eager \
        --vllm_dtype half \
        --vllm_gpu_memory_utilization 0.95 \
        --vllm_max_model_len 2048 \
        --vllm_max_num_seqs 16 \
        --vllm_language_model_only \
        --vllm_reasoning_parser qwen3 \
        --vllm_default_chat_template_kwargs '{"enable_thinking": false}' \
        --store_feedback

    echo "=== Summary scores: ${model} ==="
    for eval_jsons_dir in "${eval_results_root}/fit/unmute/${gen_llm}/eval_jsons"/*/; do
        [ -d "${eval_jsons_dir}" ] || continue
        python3.12 /mnt/matylda4/udupa/exps/RAG/scripts/score_rag.py \
            --eval_jsons_root "${eval_jsons_dir}"
    done
done

