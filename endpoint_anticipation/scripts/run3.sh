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

env=/mnt/matylda4/udupa/exps/endpointing/smart-endpointing/ep-venv/bin/activate
run_folder="/mnt/matylda4/udupa/exps/endpointing/NAC-LD-Endpointer"
cd $run_folder
source $env

export CUDA_VISIBLE_DEVICES=$(/mnt/matylda4/udupa/exps/archive/NLP-project-whisper/sge_utils/free-gpus.sh 1) || {
  echo "Could not obtain GPU."
  exit 1
}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "hostname: $(hostname)"
# ------------------------------------- train -------------------------------------
# python3 run.py --config configs/forecasting/nemo/fc_lstm_nemo_12.5hz_loss1-1.yaml
# python3 run.py --config configs/forecasting/nemo/fc_lstm_nemo_12.5hz_loss1-01.yaml
# python3 run.py --config configs/forecasting/nemo/fc_lstm_nemo_12.5hz_loss-asym2-1.yaml


# python3 run.py --config configs/forecasting/nemo/fc_transformer_tuning/fc_transformer_nemo_12.5hz_loss-asym2-1_m1.yaml
# python3 run.py --config configs/forecasting/nemo/fc_transformer_tuning/fc_transformer_nemo_12.5hz_loss-asym2-1_m2.yaml
# python3 run.py --config configs/forecasting/nemo/fc_transformer_tuning/fc_transformer_nemo_12.5hz_loss-asym2-1_m3.yaml
# python3 run.py --config configs/forecasting/nemo/fc_transformer_tuning/fc_transformer_nemo_12.5hz_loss-asym2-1_m4.yaml
# python3 run.py --config configs/forecasting/nemo/fc_transformer_tuning/fc_transformer_nemo_12.5hz_loss-asym2-1_m5.yaml
# python3 run.py --config configs/forecasting/nemo/fc_transformer_tuning/fc_transformer_nemo_12.5hz_loss-asym2-1_m6.yaml
# python3 run.py --config configs/forecasting/nemo/fc_transformer_tuning/fc_transformer_nemo_12.5hz_loss-asym2-1_m7.yaml
# python3 run.py --config configs/forecasting/nemo/fc_transformer_tuning/fc_transformer_nemo_12.5hz_loss-asym2-1_m8.yaml

# ------------------------------------- infer -------------------------------------
# python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_AS_lstm_nemo_12.5hz_loss1-1
# python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_lstm_nemo_12.5hz_loss1-1
# python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_lstm_nemo_12.5hz_loss1-01
# python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_lstm_nemo_12.5hz_loss-asym2-1
# python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_lstm_nemo_12.5hz_loss-asym2-1
# python3 run.py --config configs/infer.yaml --infer fc_lstm_nemo_12.5hz

python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_transformer_nemo_12.5hz_loss1-1_m1
python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_transformer_nemo_12.5hz_loss1-1_m2
python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_transformer_nemo_12.5hz_loss1-1_m3
python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_transformer_nemo_12.5hz_loss1-1_m4

# python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_transformer_nemo_12.5hz_loss-asym2-1_m1
# python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_transformer_nemo_12.5hz_loss-asym2-1_m2
# python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_transformer_nemo_12.5hz_loss-asym2-1_m3
# python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_transformer_nemo_12.5hz_loss-asym2-1_m4
# python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_transformer_nemo_12.5hz_loss-asym2-1_m5
# python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_transformer_nemo_12.5hz_loss-asym2-1_m6
# python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_transformer_nemo_12.5hz_loss-asym2-1_m7
# python3 run.py --config configs/infer.yaml --infer data_mix_2__fc_transformer_nemo_12.5hz_loss-asym2-1_m8



echo "Job finished at: $(date)"
end_time=$(date +%s)
time_taken_minutes=$(echo "scale=2; ($end_time - $start_time) / 60" | bc)
echo "Time taken: $time_taken_minutes minutes"

