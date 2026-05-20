import os
import json
import argparse
from glob import glob
import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr
from tqdm import tqdm

MODEL_NAME = ""


def get_time_aligned_transcription(data_path, task, audio_filename="output.wav"):
    # Collect all matching audio files under the root directory
    # print(data_path)
    audio_paths = sorted(glob(f"{data_path}/*/{MODEL_NAME}{audio_filename}"))
    # print(audio_paths)
    # exit()

    # Load the pretrained NeMo ASR model and move to GPU
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # asr_model = asr_model.cuda()
    asr_model = asr_model.to(device)

    
    for audio_path in tqdm(audio_paths):
        print(audio_path)
        result_path = audio_path.replace(f"{MODEL_NAME}{audio_filename}", audio_filename.replace(".wav", ".json"))
        print(result_path)

        if os.path.exists(result_path):
            print(f"  Skipping (transcript exists): {result_path}")
            continue

        # Read the audio file (waveform and sample rate)
        waveform, sr = sf.read(audio_path)
        # If multichannel audio, convert to mono by averaging channels
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Default offset is zero (no cropping)
        offset = 0.0

        if task == "user_interruption":
            # Load the interrupt metadata to get [start, end] timestamps
            meta_path = audio_path.replace(f"{MODEL_NAME}output.wav", "interrupt.json")
            with open(meta_path, "r") as f:
                interrupt_meta = json.load(f)

            # We only care about the end of the interruption
            _, end_interrupt = interrupt_meta[0]["timestamp"]
            offset = end_interrupt

            # Compute the sample index to start from, and crop the waveform
            start_idx = int(end_interrupt * sr)
            waveform = waveform[start_idx:]

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, waveform, sr)
            # original file‐based API (this accepts timestamps=True)
            asr_outputs = asr_model.transcribe([tmp.name], timestamps=True)
        # remove the temp file so you don't leak disk
        os.unlink(tmp.name)

        # Take the first (and only) result
        result = asr_outputs[0]
        word_timestamps = result.timestamp["word"]

        # Build the output dict, adjusting each timestamp by the offset
        chunks = []
        text = ""
        for w in word_timestamps:
            start_time = w["start"] + offset
            end_time = w["end"] + offset
            word = w["word"]

            text += word + " "
            chunks.append(
                {
                    "text": word,
                    "timestamp": [start_time, end_time],
                }
            )

        output_dict = {
            "text": text.strip(),
            "chunks": chunks,
        }

        # Write the JSON result next to the WAV file
        
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(output_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe full audio or only after a user interruption"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root folder containing subfolders with output.wav (and interrupt.json)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="full",
        choices=["full", "user_interruption"],
        help="Choose 'full' for entire transcript or 'user_interruption' to crop before ASR",
    )
    parser.add_argument(
        "--audio_filename",
        type=str,
        default="output.wav",
        help="Audio filename to transcribe within each subfolder (default: output.wav)",
    )
    args = parser.parse_args()

    get_time_aligned_transcription(args.root_dir, args.task, args.audio_filename)