import wandb, os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()  
parser.add_argument("--wandb_project", type=str, default="forecasting")
parser.add_argument("--root_path", type=str, default="/mnt/matylda4/udupa/exps/endpointing/NAC-LD-Endpointer/checkpoints")
parser.add_argument("--wandb_run_name", type=str, default=None)
parser.add_argument("--add_audio", type=str, default=None)


def add_img(run, folder):
    for filename in os.listdir(os.path.join(args.root_path, folder)):
        if filename.endswith(".png"):
            run.log({filename: wandb.Image(os.path.join(args.root_path, folder, filename))})

def add_audio(run, fname):
    run.log({Path(fname).stem: wandb.Audio(fname)})

if __name__ == "__main__":
    args = parser.parse_args()
    wandb.login()
    if args.add_audio is not None:
        run = wandb.init(project=args.wandb_project, name="Audio")
        add_audio(run, args.add_audio)
        exit()
    if args.wandb_run_name is None:
        done_flag = ".wandb_update_done"
        for folder in os.listdir(args.root_path):
            done_flag_path = os.path.join(args.root_path, folder, done_flag)
            if not os.path.exists(done_flag_path):
                run = wandb.init(project=args.wandb_project, name=folder)
                add_img(run, folder)
                run.finish()
            with open(done_flag_path, "w") as f:
                f.write(" ")
    else:
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        add_img(run, args.wandb_run_name)
        run.finish()