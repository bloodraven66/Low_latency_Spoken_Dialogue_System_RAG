#!/bin/bash
set -ex
cd "$(dirname "$0")/.."

#python3.10 -m venv .llm-server-venv
#curl -LsSf https://astral.sh/uv/install.sh | sh
#echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
#source ~/.bashrc

export CUDA_VISIBLE_DEVICES=$(/mnt/matylda4/udupa/exps/archive/NLP-project-whisper/sge_utils/free-gpus.sh 1) || {
  echo "Could not obtain GPU."
  exit 1
}


export CUDA_HOME=/usr/local/share/cuda-12.1

export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"
export PATH="$CARGO_HOME/bin:$PATH"

export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=$CUDA_HOME/bin:$PATH

ulimit -f unlimited
ulimit -v unlimited
ulimit -u 4096
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

source .vllm-venv/bin/activate

# pip install vllm==0.9.1

# vllm serve google/gemma-3-1b-it \
#   --max-model-len=4096 \
#   --dtype=bfloat16 \
#   --gpu-memory-utilization=0.7 \
#   --port=8091


vllm serve google/gemma-3-12b-it \
  --max-model-len=8192 \
  --dtype=bfloat16 \
  --gpu-memory-utilization=0.7 \
  --host=127.0.0.1 \
  --port=8091

# python3 -m vllm.entrypoints.cli.main serve \
#   --model=google/gemma-3-4b-it \
#   --max-model-len=4096 \
#   --dtype=bfloat16 \
#   --gpu-memory-utilization=0.7 \
#   --port=8091

# vllm serve \
#   --model=google/gemma-3-1b-it \
#   --max-model-len=8192 \
#   --dtype=bfloat16 \
#   --gpu-memory-utilization=0.3 \
#   --port=8091

# uv tool run vllm@v0.9.1 serve \
#   --model=google/gemma-3-1b-it \
#   --max-model-len=8192 \
#   --dtype=bfloat16 \
#   --gpu-memory-utilization=0.3 \
#   --port=8091
