#!/bin/bash

set -er

root_path=/mnt/matylda4/udupa/exps/full_duplex/unmute

# instruction_type=smalltalk_no_starter
# instruction_type=anticipate
instruction_type=rag
# instruction_type=anticipate_rag

m=gemma3_1b
model_name=unmute_${instruction_type}_${m}_cache_test
save_folder=anticipation_exps_mar2026

# Used when instruction_type=rag
rag_url="http://127.0.0.1:8095"
rag_top_k=1

path="/mnt/matylda4/udupa/data/Full-Duplex-Bench-Data/"
version="v1.0"
task="candor_turn_taking"

skip_if_present=false

output_path="${root_path}/results_tmp/fd-${version}-${save_folder}-v1/${model_name}/${task}/"
echo "Output path: ${output_path}"

mkdir -p ${output_path} || exit
port=8020

input_dir="${path}/${version}/${task}/"

    
current_id=0
total_files=$(ls ${input_dir} | wc -l)
for id in $(ls ${input_dir}); do
    current_id=$((current_id + 1))
    folder="${input_dir}/${id}/"
    audio_path="${folder}/input.wav"

    if [ ! -f "${audio_path}" ]; then
        echo "File not found: ${audio_path}"
        exit
    fi
    save_folder="${output_path}/${id}/"
    mkdir -p ${save_folder} || exit
    save_path="${save_folder}/output.wav"

    if [ "${skip_if_present}" = true ] && [ -f "${save_path}" ]; then
        echo "Output file already exists, skipping: ${save_path}"
        continue
    fi

    echo ""
    
    echo "Processing file ${current_id}/${total_files}: ${audio_path}"
    echo ""

    ##if instruction_type use unmute_handler_from_file else use unmute_handler_from_file_no_starter

    if [ "${instruction_type}" = "smalltalk_no_starter" ]; then
        # handler="unmute.unmute_handler_from_file_no_starter"
        export PYTHONPATH=.
        cmd="python3.12 unmute/scripts/evaluate_recording.py ${audio_path} ${save_path}"
    elif [ "${instruction_type}" = "anticipate" ]; then
        # handler="unmute.unmute_handler_from_file_forecasting"
        export PYTHONPATH=.
        cmd="python3.12 unmute/scripts/evaluate_recording_speculative.py ${audio_path} ${save_path}"
    elif [ "${instruction_type}" = "rag" ]; then
        export PYTHONPATH=.
        cmd="python3.12 unmute/scripts/evaluate_recording_rag.py ${audio_path} ${save_path} --rag-url ${rag_url} --rag-top-k ${rag_top_k}"
    elif [ "${instruction_type}" = "anticipate_rag" ]; then
        export PYTHONPATH=.
        cmd="python3.12 unmute/scripts/evaluate_recording_speculativerag.py ${audio_path} ${save_path} --rag-url ${rag_url}"
    else
        echo "Not implemented yet"
        exit
    fi
    echo "Using handler: ${cmd}"

    ${cmd}
    num_files_to_run=3
    if [ ${current_id} -ge ${num_files_to_run} ]; then
        echo "Processed ${num_files_to_run} files, exiting."
        break
    fi
    

done
