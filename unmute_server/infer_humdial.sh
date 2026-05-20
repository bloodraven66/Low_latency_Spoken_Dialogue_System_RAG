#!/bin/bash

set -er

root_path=/mnt/matylda4/udupa/exps/full_duplex/unmute

instruction_type=smalltalk_no_starter
# instruction_type=smalltalk
model_name=unmute_${instruction_type}

path="/mnt/matylda4/udupa/data/HumDial/HD-Track2/HD-Track2-dev/"
version="HD-Track2-dev-en"
#task="candor_turn_taking"

# task="Follow-up Questions"
# task="Pause Handling"
# task="Negation or Dissatisfaction"
# task="Repetition Requests"
# task="Silence or Termination"
# task="Third-party Speech_after"
task="Topic Switching"

skip_if_present=true

output_path="${root_path}/results/humdial-${version}/${model_name}/${task}/"
echo "Output path: ${output_path}"

mkdir -p "${output_path}" || exit

input_dir="${path}/${version}/${task}/"
current_id=0
# total_files=$(ls ${input_dir} | wc -l)
total_files=$(find "${input_dir}" -maxdepth 1 -name "*.wav" | wc -l)
echo "Total files to process: ${total_files}"

for audio_path in "${input_dir}"/*.wav; do
    current_id=$((current_id + 1))
    folder="${input_dir}/${id}/"
    id=$(basename "${audio_path}" .wav)

    if [ ! -f "${audio_path}" ]; then
        echo "File not found: ${audio_path}"
        exit
    fi
    save_folder="${output_path}/${id}/"
    mkdir -p "${save_folder}" || exit
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
    else
        handler="unmute.unmute_handler_from_file"
    fi
    echo "Using handler: ${handler}"

    python3.12 -m ${handler} -i "${audio_path}" -o "${save_path}" --instruction_type "${instruction_type}"
    # sleep 5

    echo ""


done

