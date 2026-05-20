from dataclasses import dataclass

import os, sys, json

from infer_asr import score
os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import re
import time
import jiwer
import sentencepiece
import sphn
import textwrap
import numpy as np
import torch
from tqdm import tqdm
from moshi.models import loaders, MimiModel, LMModel, LMGen
import argparse
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--audio_folder', type=str, default="/mnt/matylda4/udupa/data/HumDial/HD-Track2/HD-Track2-dev/HD-Track2-dev-zh/", help='Path to the folder containing audio files.')
parser.add_argument('--save_folder', type=str, default="results/HD-Track2-dev-zh/", help='Path to the folder to save results.')
parser.add_argument('--hf_model', type=str, default="kyutai/stt-1b-en_fr", help='Huggingface model name or path.')
parser.add_argument('--root_dir', type=str, default="/mnt/matylda4/udupa/exps/full_duplex/moshi-finetune/", help='Root directory for experiments.')
parser.add_argument('--score', action='store_true', help='Whether to metrics only without saving detailed results.')
args = parser.parse_args()

class LSTM_Model(torch.nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super(LSTM_Model, self).__init__()
        self.model = nn.LSTM(
            input_size=kwargs["input_size"],
            hidden_size=kwargs["hidden_size"],
            num_layers=kwargs["num_layers"],
            batch_first=True,
            dropout=kwargs["dropout"],
            bidirectional=kwargs["bidirectional"],
        )
        self.linear = nn.Linear(kwargs["hidden_size"], kwargs["output_size"])
        self.system_embed = nn.Embedding(2, kwargs["input_size"])
        self.h = None
        self.c = None
        
    def init_hidden(self, batch_size, device):
        h_o = torch.zeros(self.model.num_layers, batch_size, self.model.hidden_size)
        c_0 = torch.zeros(self.model.num_layers, batch_size, self.model.hidden_size)
        return h_o.to(device), c_0.to(device)
    
    def infer(self, x, h=None, c=None, system_ids=None, init_hidden=False):
        assert x.size(0) == 1, "Inference only supports batch size of 1"
        if self.h is None:
            if init_hidden:
                print("Initializing hidden states")
                h, c = self.init_hidden(x.size(0), x.device)
        # else:
            # print("Using previous hidden states")
            # h, c = self.h, self.c
        # h, c = self.init_hidden(x.size(0), x.device)
        if system_ids is None:
            system_ids = torch.zeros(x.size(2), dtype=torch.long, device=x.device)
        x = x.permute(0, 2, 1)
        # x = self.mel_embed(x)
        x = x + self.system_embed(system_ids)
        x, (h_, c_) = self.model(x, (h, c))
        if self.h is None:
            self.h, self.c = h_, c_
        x = self.linear(x)
        return x, (h_, c_)

def load_model():
    # model_checkpoint = "/mnt/matylda4/udupa/exps/endpointing/smart-endpointing/checkpoints/lstm_mimi-12.5hz-nq8_delay3f/best_val_acc.pt"
    # model_checkpoint = "/mnt/matylda4/udupa/exps/endpointing/smart-endpointing/checkpoints/humdial_lstm_mimi-12.5hz-nq8_delay2f/best_val_acc.pt"
    model_checkpoint = "/mnt/matylda4/udupa/exps/endpointing/smart-endpointing/checkpoints/humdial_lstm_mimi-12.5hz-nq8_delay2f_load_spokenwoz/best_val_acc.pt"
    kwargs = {
        "input_size": 512,
        "hidden_size": 324,
        "num_layers": 3,
        "output_size": 5,
        "dropout": 0.1,
        "bidirectional": False,
    }

    model = LSTM_Model(
        **kwargs
    )
    model_state_dict = torch.load(model_checkpoint, map_location=device)["model_state_dict"]
    model.load_state_dict(model_state_dict)
    model.to(device) # Ensure model is loaded onto the correct device
    model.eval()
    return model
    

@dataclass
class InferenceState:
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    delay_frames: int = 0  # Number of initial frames to skip

    def __init__(
        self,
        mimi: MimiModel,
        batch_size: int,
        device: str | torch.device,
        delay_frames: int = 0,
        endpointer: LSTM_Model = None,
    ):
        self.mimi = mimi
        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.batch_size = batch_size
        self.delay_frames = delay_frames
        self.mimi.streaming_forever(batch_size)
        self.endpointer = endpointer
        self.labels = ["bos", "system_end", "user_end", "system",  "user"]

    def run(self, in_pcms: torch.Tensor):
        ntokens = 0
        first_frame = True
        chunks = [
            c
            for c in in_pcms.split(self.frame_size, dim=1   )
            if c.shape[-1] == self.frame_size
        ]
        start_time = time.time()
        all_text = []
        all_outs = []
        h, c = self.endpointer.init_hidden(1, device)
        data = []
        data = torch.zeros(len(chunks), len(self.labels), device=device)
        with torch.no_grad():
            for idx, chunk in enumerate(tqdm(chunks)):
                codes = self.mimi.encode(chunk.unsqueeze(0).to(self.device))
                embeddings = self.mimi.quantizer.decode(codes)                
                output, (h, c) = self.endpointer.infer(embeddings, h, c)
                data[idx] = output.squeeze().detach()
            data = torch.nn.functional.softmax(data, dim=-1).detach().cpu().numpy()
        return data

def plot_out_and_audio(out, wav, labels, seg_begs, seg_ends, all_trigger_locations):
    import matplotlib.pyplot as plt
    t_wav = np.arange(wav.shape[-1]) / 24000          # seconds
    t_out = np.arange(out.shape[0]) / 12.5
    fig, ax = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    # print(wav.shape, out.shape)
    ax[0].plot(t_wav, wav.squeeze())
    ax[0].set_title("Audio Waveform")
    for label_idx in range(out.shape[1]):
        ax[1].plot(t_out, out[:, label_idx], label=labels[label_idx])
    ax[1].set_title("Endpointer Outputs")
    for beg, end in zip(seg_begs, seg_ends):
        ax[1].axvline(x=beg/12.5, color='g', linestyle='--')
        ax[1].axvline(x=end/12.5, color='r', linestyle='--')
        ax[0].axvline(x=beg/12.5, color='g', linestyle='--')
        ax[0].axvline(x=end/12.5, color='r', linestyle='--')
    for trigger_loc in all_trigger_locations[0]:
        ax[1].scatter(trigger_loc/12.5, out[trigger_loc, 2], color='k', marker='x', s=100)
    ax[1].legend()
    # print(all_trigger_locations)
    plt.xlabel("Frames")
    plt.show()
    plt.savefig("ep_output.png")
    exit()
     

device = "cuda" if torch.cuda.is_available() else "cpu"
# Use the en+fr low latency model, an alternative is kyutai/stt-2.6b-en

def infer(audio_path, json_path, save_name, mimi, state, ep_thresh=0.8, late_trigger_collars=[0.2, 0.4, 0.6, 0.8, 1.0]):
    in_pcms_full, _ = sphn.read(audio_path, sample_rate=mimi.sample_rate)
    with open(json_path, "r") as f:
        segment_data = json.load(f)["speech_segments"]
    save_data = []
    in_pcms = torch.from_numpy(in_pcms_full).to(device=device)
    out = state.run(in_pcms)
    user_end_probs = out[:, 2]
    early_triggers, late_triggers = [], {}
    seg_begs, seg_ends = [], []
    for segment in segment_data:
        beg, end = segment["xmin"], segment["xmax"]
        beg_frame = int(beg * mimi.frame_rate)
        end_frame = int(end * mimi.frame_rate) + 2 #adding 2 frames because the timestamp cuts off a bit early
        seg_begs.append(beg_frame)
        seg_ends.append(end_frame)
        ##check if user_end_probs cross threshold in this region
        segment_user_end_probs = user_end_probs[beg_frame:end_frame]
        endpointer_triggered = np.any(segment_user_end_probs[5:] >= ep_thresh) ##ignore first 5 frames to avoid the initial delay
        early_triggers.append(endpointer_triggered)

        ##check for late trigger - should be triggered within late_trigger_collar seconds after end
        for late_trigger_collar in late_trigger_collars:
            collar_frames = int(late_trigger_collar * mimi.frame_rate)
            late_region_user_end_probs = user_end_probs[end_frame:end_frame + collar_frames]
            late_triggered = np.any(late_region_user_end_probs >= ep_thresh)
            if late_trigger_collar not in late_triggers:
                late_triggers[late_trigger_collar] = []
            late_triggers[late_trigger_collar].append(late_triggered)
        # late_triggers.append(late_triggered)

        # all_trigger_locations = np.where(user_end_probs >= ep_thresh)

    return early_triggers, late_triggers, out
    # plot_out_and_audio(out, in_pcms_full, state.labels, seg_begs, seg_ends, all_trigger_locations)

def run_inference():
    for folder in os.listdir(audio_folder):
        if folder == "readme.txt":
            continue
        save_folder_ = os.path.join(save_folder, folder)
        os.makedirs(save_folder_, exist_ok=True)
        audio_folder_ = os.path.join(audio_folder, folder)
        json_files = [f for f in os.listdir(audio_folder_) if f.endswith("_sentence.json")]
        valid_audio_files = [f.replace("_sentence.json", ".wav") for f in json_files]
            
        path = args.hf_model
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(path)
        mimi = checkpoint_info.get_mimi(device=device)
        endpointer = load_model()

        state = InferenceState(mimi, batch_size=1, device=device, endpointer=endpointer)
        all_early_triggers = []
        ep_thresh = 0.8
        trigger_collars = [0.2, 0.4, 0.6, 0.8, 1.0]
        all_late_triggers = {late_trigger: [] for late_trigger in trigger_collars}
        for json_file, audio_file in tqdm(zip(json_files, valid_audio_files)):
            audio_path = os.path.join(audio_folder_, audio_file)
            json_path = os.path.join(audio_folder_, json_file)
            save_name = os.path.join(save_folder_, json_file.replace("_sentence.json", "_results.json"))
            if os.path.exists(save_name):
                print(f"Skipping {save_name} as it already exists.")
                continue
            out = infer(audio_path, json_path, save_name, mimi, state, ep_thresh=ep_thresh, late_trigger_collars=trigger_collars)
            all_early_triggers.extend(out[0])
            # all_late_triggers.extend(out[1])
            for late_trigger in out[1]:
                all_late_triggers[late_trigger].extend(out[1][late_trigger])


            early_trigger_acc = 1 - (sum(all_early_triggers) / len(all_early_triggers))
            late_trigger_accs = {}
            for late_trigger in all_late_triggers:
                late_trigger_accs[late_trigger] = sum(all_late_triggers[late_trigger]) / len(all_late_triggers[late_trigger])
            print(f"Processed {len(all_early_triggers)} segments so far.")
            print(f"Pause Handling Accuracy: {early_trigger_acc*100:.2f}%")
            for late_trigger in late_trigger_accs:
                print(f"Late Trigger Accuracy within {late_trigger} sec: {late_trigger_accs[late_trigger]*100:.2f}%")
            early_triggers_int = [int(x) for x in out[0]]
            late_triggers_int = {k: [int(x) for x in v] for k, v in out[1].items()}
            results = {
                "raw_probs": out[2].tolist(),
                "labels": state.labels,
                "early_triggers": early_triggers_int,
                "late_triggers": late_triggers_int,
                "ep_thresh": ep_thresh,
                "trigger_collars": trigger_collars,
            }
            with open(save_name, "w") as f:
                json.dump(results, f, indent=4)

def score():
    id2json = {}
    late_trigger_collar = 0.4
    threshold = 0.2
    for folder in os.listdir(audio_folder):
        if folder == "readme.txt":
            continue
        audio_folder_ = os.path.join(audio_folder, folder)
        for json_file in os.listdir(audio_folder_):
            if json_file.endswith("_sentence.json"):
                id = json_file.replace("_sentence.json", "")
                id = os.path.join(folder.replace(' ', '__'), id)
                id2json[id] = os.path.join(audio_folder_, json_file)

    # vad_path = "/mnt/matylda4/udupa/data/HumDial/text_5700_train_dev/dev_vad_processed_v2"
        
    print(len(id2json), exp_folder, save_folder)
    example_json = list(id2json.values())[0]
    with open(example_json, "r") as f:
        example_data = json.load(f)
    print(example_data)
    overall_early_triggers = []
    overall_late_triggers = []
    for subfolder in os.listdir(save_folder):
        early_triggers = []
        late_triggers = []
        json_files = [f for f in os.listdir(os.path.join(save_folder, subfolder)) if f.endswith("_results.json")]
        for json_file in json_files:
            json_id = json_file.replace("_results.json", "")
            json_id = os.path.join(subfolder.replace(' ', '__'), json_id)
            with open(os.path.join(save_folder, subfolder, json_file), "r") as f:
                results = json.load(f)
            with open(id2json[json_id], "r") as f:
                segment_data = json.load(f)
            
            user_end_probs = results["raw_probs"]
            # if len(segment_data["speech_segments"]) == 1:
            #     continue
            for sidx, segment in enumerate(segment_data["speech_segments"]):
                ##if last sgement
                # print(subfolder, json_file, segment_data["speech_segments"])
                if "Pause" not in subfolder:
                    if sidx == len(segment_data["speech_segments"]) - 1:
                        break

                 ##check early trigger
                beg, end = segment["xmin"], segment["xmax"]
                beg_frame = int(beg * 12.5)
                end_frame = int(end * 12.5) 
                segment_user_end_probs = [user_end_probs[i][2] for i in range(beg_frame, end_frame)]
                endpoint_triggered = any([p >= threshold for p in segment_user_end_probs[5:]])
                early_triggers.append(int(endpoint_triggered))

                collar_frames = int(late_trigger_collar * 12.5)
                late_region_user_end_probs = [user_end_probs[i][2] for i in range(end_frame, end_frame + collar_frames)]
                late_triggered = any([p >= threshold for p in late_region_user_end_probs])
                late_triggers.append(int(late_triggered))
                # late_trigger_idx = [p >= threshold for i, p in enumerate(late_region_user_end_probs)]
                # print(late_trigger_idx)
            # print('--')
        if len(early_triggers) == 0:
            continue
        early_trigger_acc = 1 - (sum(early_triggers) / len(early_triggers))
        late_trigger_acc = sum(late_triggers) / len(late_triggers)
        print('----------------- Results -----------------')
        print(f"Folder: {subfolder}, num segments: {len(early_triggers)}")
        print(f"Pause Handling Accuracy: {early_trigger_acc*100:.2f}%")
        print(f"Late Trigger Accuracy within {late_trigger_collar} sec: {late_trigger_acc*100:.2f}%")
        overall_early_triggers.extend(early_triggers)
        overall_late_triggers.extend(late_triggers)

    overall_early_trigger_acc = 1 - (sum(overall_early_triggers) / len(overall_early_triggers))
    overall_late_trigger_acc = sum(overall_late_triggers) / len(overall_late_triggers)
    print('================= Overall Results ================')
    print(f"Overall num segments: {len(overall_early_triggers)}")
    print(f"Overall Pause Handling Accuracy: {overall_early_trigger_acc*100:.2f}%")
    print(f"Overall Late Trigger Accuracy within {late_trigger_collar} sec: {overall_late_trigger_acc*100:.2f}%")
                                                           
            # print(results)
            # print(segment_data)
            # exit()
    # print(save_folder) 

if __name__ == "__main__":
    audio_folder = args.audio_folder
    save_folder = args.save_folder
    exp_folder = "lstm-endpointer-humdial-spokenwoz-d2f"
    # exp_folder = "lstm-endpointer-humdial-d2f"
    # exp_folder = "lstm-endpointer"
    save_folder = os.path.join(args.root_dir, save_folder, exp_folder)
    os.makedirs(save_folder, exist_ok=True)
    if args.score:
        score()
        exit()
    
    run_inference()
    