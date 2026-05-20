export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HOME="/mnt/matylda4/udupa/hugging-face"
export WANDB_MODE="offline"

export CUDA_VISIBLE_DEVICES=$(/mnt/matylda4/udupa/exps/archive/NLP-project-whisper/sge_utils/free-gpus.sh 1) || {
  echo "Could not obtain GPU."
  exit 1
}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


python3 NeMo/tools/nemo_forced_aligner/align.py \
        pretrained_name="stt_en_fastconformer_hybrid_large_pc" \
        manifest_filepath=tmp/manifest.json \
        output_dir=tmp/forced_alignments \
        align_using_pred_text=True \

