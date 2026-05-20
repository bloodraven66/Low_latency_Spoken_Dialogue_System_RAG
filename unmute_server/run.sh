#!/bin/bash

# script=/mnt/matylda4/udupa/exps/full_duplex/Hum-Dial/Full-Duplex_Interaction/evaluation/get_transcript/infer_en.py
# folder="/mnt/matylda4/udupa/exps/full_duplex/unmute/results/humdial-HD-Track2-dev-en/unmute_smalltalk_no_starter/"

# for folder_ in "$folder"/*; do
#     if [ -d "$folder_" ]; then
#         echo "Processing folder: $folder_"
#         python3 $script --root_dir "$folder_"
#     else
#         echo "Skipping non-directory item: $folder_"
#     fi
# done

script="/mnt/matylda4/udupa/exps/full_duplex/Full-Duplex-Bench/evaluation/evaluate.py"

# setup=anticipate
# setup=smalltalk_no_starter
setup=rag
# setup=anticipate_rag

# folder="/mnt/matylda4/udupa/exps/full_duplex/unmute/results_tmp/fd-v1.0-anticipation_exps_mar2026-v1/unmute_"$setup"_gemma3_1b_cache_test/candor_turn_taking/"

# python3 $script --root_dir $folder --task smooth_turn_taking

# ── FIT RAG Benchmark latency eval ──────────────────────────────────────────
latency_script="/mnt/matylda4/udupa/exps/full_duplex/unmute/dockerless/eval_latency_fit.py"
fit_base="/mnt/matylda4/udupa/exps/full_duplex/unmute/results_tmp/fit_rag_benchmark"
export PYTHONPATH="/mnt/matylda4/udupa/exps/full_duplex/unmute"

for model in unmute_base_gemma3_12b; do
    echo ""
    echo "=== Latency eval: ${model} ==="
    python3.12 "${latency_script}" --results_dir "${fit_base}/${model}"
done
