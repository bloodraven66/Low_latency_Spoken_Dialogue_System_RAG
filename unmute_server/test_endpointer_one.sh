#!/bin/bash

set -e

# Test endpointer-based inference on audio files from a specific set
# Usage: ./test_endpointer_one.sh [set_number]
#   set_number: 0-9 (default: 0)
#   Each set processes 1/10 of the files in the folder

root_path=/mnt/matylda4/udupa/exps/full_duplex/unmute

# Configuration
instruction_type=smalltalk_no_starter
# audio_folder="/mnt/matylda4/udupa/data/HumDial/HD-Track2/HD-Track2-dev/HD-Track2-dev-en/Follow-up Questions"
audio_folder="/mnt/matylda4/udupa/data/HumDial/HD-Track2-Test/test/"
# audio_folder="/mnt/matylda4/udupa/data/HumDial/HD-Track2-Test/clean/"
output_dir="${root_path}/results/test_full_endpointer/"

# Get set number from argument (default to 0)
SET_NUM=${1:-0}

if [[ ! "$SET_NUM" =~ ^[0-9]$ ]]; then
    echo "Error: Set number must be 0-9"
    echo "Usage: $0 [set_number]"
    exit 1
fi

mkdir -p "${output_dir}"

# Map set number to port: 0-1 → 8010, 2-3 → 8012, 4-5 → 8014, 6-7 → 8016, 8-9 → 8020
if [ "$SET_NUM" -le 1 ]; then
    port=8010
elif [ "$SET_NUM" -le 3 ]; then
    port=8012
elif [ "$SET_NUM" -le 5 ]; then
    port=8014
elif [ "$SET_NUM" -le 7 ]; then
    port=8016
else
    port=8020
fi

echo "============================================"
echo "Processing Set $SET_NUM (1/10 of files) on port $port"
echo "============================================"

# Collect all wav files into a sorted array
mapfile -d '' all_files < <(find "$audio_folder" -maxdepth 1 -name "*.wav" -print0 | sort -z)
total_files=${#all_files[@]}

if [ $total_files -eq 0 ]; then
    echo "No .wav files found in $audio_folder"
    exit 1
fi

# Calculate set boundaries (10 sets)
files_per_set=$(( (total_files + 9) / 10 ))  # Ceiling division
start_idx=$(( SET_NUM * files_per_set ))
end_idx=$(( start_idx + files_per_set ))

# Clamp end_idx to total_files
if [ $end_idx -gt $total_files ]; then
    end_idx=$total_files
fi

echo "Total files: $total_files"
echo "Set $SET_NUM: Processing files $((start_idx + 1)) to $end_idx"
echo "============================================"

# Process files in this set
count=0
processed=0
for test_audio in "${all_files[@]}"; do
    if [ $count -ge $start_idx ] && [ $count -lt $end_idx ]; then
        processed=$((processed + 1))
        
        filename=$(basename "${test_audio}" .wav)
        output_path="${output_dir}/${filename}_output.wav"

        if [ -f "${output_path}" ]; then
            echo "✓ Skipping ${filename}, output already exists."
            count=$((count + 1)) 
            continue
        fi
        
        echo ""
        echo "File $processed (global: $((count + 1))/$total_files): ${filename}"
        echo "--------------------------------------------"
        
        python3.12 -m unmute.unmute_handler_from_file_no_starter_ep \
            -i "${test_audio}" \
            -o "${output_path}" \
            --instruction_type "${instruction_type}" \
            --server_url "ws://localhost:${port}/v1/realtime"
        
        echo "✓ Done"
    fi
    
    count=$((count + 1))
done

echo ""
echo "============================================"
echo "✓ Set $SET_NUM complete: Processed $processed files"
echo "============================================"
