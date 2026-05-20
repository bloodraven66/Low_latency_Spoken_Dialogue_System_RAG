#!/bin/bash

set -er

root_path=/mnt/matylda4/udupa/exps/full_duplex/unmute

#instruction_type=smalltalk_no_starter
# instruction_type=smalltalk_no_starter
instruction_type=anticipate
m=gemma3_4b
model_name=unmute_${instruction_type}_${m}

path="/mnt/matylda4/udupa/data/Full-Duplex-Bench-Data/"
version="v1.0"
task="candor_turn_taking"

# task=candor_pause_handling

skip_if_present=true


output_path="${root_path}/results/fd-${version}-{anticipation_exps}-v1/${model_name}/${task}/"
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
        handler="unmute.unmute_handler_from_file_no_starter"
    elif [ "${instruction_type}" = "anticipate" ]; then
        handler="unmute.unmute_handler_from_file_forecasting"
    else
        handler="unmute.unmute_handler_from_file"
    fi
    echo "Using handler: ${handler}"

    # python3.12 -m unmute.unmute_handler_from_file_no_starter_ep \
    #         -i "${test_audio}" \
    #         -o "${output_path}" \
    #         --instruction_type "${instruction_type}" \
            

    python3.12 -m ${handler} -i ${audio_path} -o ${save_path} --instruction_type ${instruction_type} --server_url "ws://localhost:${port}/v1/realtime"


    # if current_id > 10: then quit
    # if [ ${current_id} -ge 10 ]; then
        # echo "Processed 10 files, exiting."
        # break
    # fi
    

done

