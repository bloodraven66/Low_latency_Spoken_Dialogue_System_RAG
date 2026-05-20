#!/bin/bash
# start_sge.sh - universal SGE launcher

set -e

# -------------------
# Defaults
# -------------------
node="supergpu4"
gpu=1
gpu_ram="16G"
ram_free="1G"
mem_free="1G"
disk=1
name=""  # job name, optional
script_path=""

# -------------------
# Parse CLI arguments
# -------------------
for arg in "$@"; do
    case $arg in
        node=*) node="${arg#*=}" ;;
        gpu=*) gpu="${arg#*=}" ;;
        gpu_ram=*) gpu_ram="${arg#*=}" ;;
        name=*) name="${arg#*=}" ;;
        run=*) script_path="${arg#*=}" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# -------------------
# Must provide a script
# -------------------
if [[ -z "$script_path" ]]; then
    echo "ERROR: Must provide run=<script_name>"
    exit 1
fi

# Convert relative script path to absolute
if [[ "$script_path" != /* ]]; then
    script_path="$(pwd)/$script_path"
fi

# Must exist
if [[ ! -f "$script_path" ]]; then
    echo "ERROR: Script file does not exist: $script_path"
    exit 1
fi

# -------------------
# Job name
# -------------------
if [[ -z "$name" ]]; then
    name="$(basename "$script_path" .sh)_$(date +%Y%m%d_%H%M%S)"
fi

# -------------------
# Log path in the script directory
# -------------------
script_dir=$(dirname "$script_path")
log_dir="$script_dir/logs"
mkdir -p "$log_dir"
log_path="$log_dir/$name.log"

# -------------------
# Root launcher
# -------------------
root_run="/mnt/matylda4/udupa/common/root_run.sh"

echo "Submitting job '$name' to node '$node' with gpu=$gpu"
echo "Log file will be at: $log_path"
echo "Running per-job script: $script_path"

# -------------------
# Submit job to SGE
# -------------------
qsub -N "$name" \
     -q long.q \
     -l h="$node",gpu="$gpu",gpu_ram="$gpu_ram",ram_free="$ram_free",matylda4="$disk" \
     -j yes \
     -o "$log_path" \
     "$root_run" "$script_path"