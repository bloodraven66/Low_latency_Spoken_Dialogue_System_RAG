#!/bin/bash

export CUDA_VISIBLE_DEVICES=""
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export CUDA_LAUNCH_BLOCKING=1

# This tells PyTorch to skip JIT compilation
export PYTORCH_JIT=0


python scripts/download_nemo.py