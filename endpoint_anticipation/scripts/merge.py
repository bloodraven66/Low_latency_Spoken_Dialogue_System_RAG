import os, json
import wandb
import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 28})

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", required=True, nargs='+')
parser.add_argument("--groups", required=True, nargs='+')
parser.add_argument("--chk_path", default="/mnt/matylda4/udupa/exps/endpointing/NAC-LD-Endpointer/checkpoints")
parser.add_argument("--save_folder", default="/mnt/matylda4/udupa/exps/endpointing/NAC-LD-Endpointer/results")
parser.add_argument("--group_json", default="/mnt/matylda4/udupa/exps/endpointing/NAC-LD-Endpointer/scripts/merge.json")
parser.add_argument("--add_to_wandb", default=True)
parser.add_argument("--wandb_project", default="forecasting")
parser.add_argument("--print_for_fc", default=None, nargs="+")
parser.add_argument("--ncd", default=33, type=float)
parser.add_argument("--hea", default=None, type=float)
parser.add_argument("--par", default=None, type=float)


styles = {
    0: 'solid',
    1: 'dashed',
    2: 'dashdot',
    3: 'dotted',
}

colors = [
    "black",
    "red",
    "blue",
    "pink",
    "gray"
]

def extract_results(chk_folder, ax, idx, name, color, style, linewdith, label, mc, data_idx):
    results_pt = os.path.join(chk_folder, "infer_results.pt")
    if not os.path.exists(results_pt):
        print(f"WARNING: file '{results_pt}' does not exist")
        return {}, {}
    
    metrics = torch.load(results_pt, weights_only=False)
    # print(name, idx, metrics.keys())
    
    intervals = metrics["forecast_intervals"]
    ep_cutoff = metrics["ep_cutoff"]
    median_forecast = metrics["median_forecast"]
    total_ep_cutoffs_mean = metrics["total_ep_cutoffs_mean"]
    total_cutoff_proportions_mean = metrics["total_cutoff_proportions_mean"]
    accuracies_with_collar = metrics["accuracies_with_collar"]
    thresholds = list(ep_cutoff.keys())
    vals, m_vals = {}, {}
    new_colors = None
    if name in ["fconfsh_XX", "fconf_XX"]:
        new_colors = cmap(np.linspace(0, 0.8, len(intervals)))
        new_color_iter = iter(new_colors)

    elif name in ["VAP_XX"]:
        new_colors = cmap(np.linspace(0, 0.8, 3))
        new_color_iter = iter(new_colors)
        
    
    if mc is not None:
        color=mc

    for interval_idx, interval in enumerate(intervals):
        if name in ["mimi_all"]:
            # if interval not in [320, 640]:
            # if interval not in [640, 1280]:
                # continue
            if interval == 1280:
                color = "#0d0787"
        if name in ["VAP_XX"]:
            if interval == 960:
                continue
            # interval_ = f"h={str(interval)}"
            # # label = rf'$VAP_{{{interval_}}}$'
            # label = rf'${{{interval_}}}$'
            # color = "black"
            # new_colors = None
            style = "--"
            linewdith = 3
            
            if mc is not None:
                color=mc[interval_idx]
        else:
            interval_ = f"h={str(interval)}"
            # label = rf'$VAP_{{{interval_}}}$'
            label = rf'${{{interval_}}}$'
            # if data_idx > 0:
                # label = None
            # print(mc_, mc, idx, color, interval, label, interval_idx)
        interval_ep_cutoff = [ep_cutoff[threshold][interval_idx] for threshold in thresholds]
        interval_median_forecast = [median_forecast[threshold][interval_idx] for threshold in thresholds]
        interval_ep_cutoff_means = [total_ep_cutoffs_mean[threshold][interval_idx] for threshold in thresholds]
        interval_ep_cutoff_proportions = [total_cutoff_proportions_mean[threshold][interval_idx]*100 for threshold in thresholds]
        interval_accuracies_with_collar = [accuracies_with_collar[threshold][interval_idx]*100 for threshold in thresholds]
        if args.ncd is not None:
            closest_idx = np.argmin(np.array([a - args.ncd if a - args.ncd > 0 else 100 for a in interval_ep_cutoff_proportions]))
        if args.hea is not None:
            closest_idx = np.argmin(np.array([a - args.hea if (a - args.hea) > 0 else 100 for a in interval_accuracies_with_collar]))
        if args.par is not None:
            closest_idx = np.argmin(np.array([a - args.par if a - args.par > 0 else 100 for a in interval_ep_cutoff_means]))
        ncd = round(interval_ep_cutoff_proportions[closest_idx], 2)
        tcr = round(interval_ep_cutoff[closest_idx], 2)
        mra = round(interval_median_forecast[closest_idx], 2)
        hea = round(interval_accuracies_with_collar[closest_idx], 2)
        print(data_idx, thresholds[closest_idx], name, label, interval, "&", mra, "&", hea, "&", tcr, "&", ncd)
        label = None
        # exit()
        if new_colors is not None and mc is None:
            color = next(new_color_iter)
        # print(name, color, interval)
        # ax[0].plot(interval_ep_cutoff_proportions, interval_median_forecast, label=label, linestyle=style, color=color, linewidth=linewdith)
        l = ax[0].plot(interval_ep_cutoff, interval_median_forecast, label=label, linestyle=style, color=color, linewidth=linewdith)
        # ax[1].plot(interval_ep_cutoff_proportions, interval_median_forecast, label=label, linestyle=style, color=color, linewidth=linewdith)
        # ax[2].plot(interval_ep_cutoff, interval_accuracies_with_collar, label=label, linestyle=style, color=color, linewidth=linewdith)
        ax[1].plot(interval_ep_cutoff_proportions, interval_accuracies_with_collar, label=label, linestyle=style, color=color, linewidth=linewdith)
        # ax[3].plot(interval_ep_cutoff, interval_accuracies_with_collar, label=label, linestyle=style, color=color, linewidth=linewdith)
        # print(l[0].get_color())
def add_to_wandb(filename, ax):
    if not args.add_to_wandb:
        return
    run = wandb.init(project=args.wandb_project, name=filename)
    run.log({filename: wandb.Image(ax)})
    run.finish()

if __name__ == "__main__":
    args = parser.parse_args()
    print_for_fc = []
    args.print_for_fc = [int(a) for a in args.print_for_fc]
    possible_intervals = [i*80 for i in range(50)]

    print(args.datasets)
    print(args.groups)
    print(args.print_for_fc)

    with open(args.group_json, "r") as f:
        json_data = json.load(f)
    models, model_ids, model_names, model_colours = [], [], [], []
    for idx, group in enumerate(args.groups):
        if group not in json_data:
            print(f"WARNING: group '{group}' does not exist")
        models_ = [a[0] for a in json_data[group]]
        names_ = [a[1] for a in json_data[group]]
        colors_ = [None for a in json_data[group]]
        try:
            colors_ = [a[2] for a in json_data[group]]
        except:
            pass
        model_colours.extend(colors_)
        models.extend(models_)
        model_names.extend(names_)
        model_ids.extend([idx]*len(json_data[group]))
    
    cmap = plt.get_cmap('plasma')
    colors = cmap(np.linspace(0, 0.8, len(model_ids)))
    styles_ = list(styles.values())
    styles = iter(list(styles.values()))
    linewdith=4

    # fig, ax = plt.subplots(4, 1, figsize=(20, 24))
    fig, ax = plt.subplots(2, 1, figsize=(20, 12))
    # if len(args.groups) > 1:
        # s_ = next(styles)
    s_ = None
    for data_idx, dataset in enumerate(args.datasets):
        # print(dataset)
        if len(args.datasets) > 1:
            s_ = next(styles)
        color_iter = iter(colors)
        for idx, folder in enumerate(models):
            mc_ = None
            chk_folder = os.path.join(args.chk_path, folder + '__' + dataset)
            if not os.path.exists(chk_folder):
                print(f"WARNING: folder '{chk_folder}' does not exist")
            c_ = next(color_iter)
            
            if model_names[idx] not in ["mimi_320", "mimi_960", "mimi_640", "fconf_640"]:
                # label = f'{model_names[idx]}'
                label = None
            else:
                label = f'{model_names[idx]}'
                # print("changing name")
                
                # a = f"h={str(model_names[idx].split("_")[1])}"
                # label = rf'${model_names[idx].split("_")[0]}_{{{a}}}$'
            # print("Name:", label)
            
            label = label if data_idx == 0 else None
            if len(args.groups) > 1:
                s_ = styles_[model_ids[idx]]
            # print(c_, s_, [model_names[idx]])
            if model_colours[idx] is not None:
                mc_ = model_colours[idx]
            extract_results(chk_folder, ax, model_ids[idx], name=model_names[idx], color=c_, style=s_, linewdith=linewdith, label=label, mc=mc_, data_idx=data_idx)

        ax[0].set_xlabel("Premature Anticipation Rate (%) (PAR)")
        ax[1].set_xlabel("Expected Redundant Computation (%) (ERC)")
        # ax[2].set_xlabel("Normalized Cutoff Density")
        ax[0].set_ylabel("Median Realized\nAnticipation (ms)\n(MRA)")
        # ax[1].set_ylabel("Median Realized Anticipation (MRA) (ms)")
        ax[1].set_ylabel("Horizon Entry\nAccuracy (%)\n(HEA)")
        ax[0].set_xlim((30, 70))
        ax[1].set_xlim((10, 60))
        # ax[1].set_xlim((0, 70))
        ax[1].legend(ncol=1, loc="upper left") #frameon=False)
        # horizon_leg = ax[1].legend()
        # ax[1].add_artist(horizon_leg) #
        from matplotlib.lines import Line2D
        # style_elements = [
        #     Line2D([0], [0], color='black', lw=2, linestyle='-', label='spokenwoz'),
        #     Line2D([0], [0], color='black', lw=2, linestyle='--', label='switchboard')
        # ]
        # interval_ = f"MHL\n(h=1280)"
        
        # label1 = rf'${{{interval_}}}$'
        if len(args.datasets) == 1:
            label1 = r"$\mathrm{EPA-M}$" + "\n" + r"$(h=1280)$"
            label2 = r"$\mathrm{EPA-M}$" + "\n" + r"$(h=640)$"
            label3 = r"$\mathrm{VAP}$" + "\n" + r"$(h=1280)$"
            label4 = r"$\mathrm{VAP}$" + "\n" + r"$(h=640)$"
            style_elements = [
                Line2D([0], [0], color='#0d0787', lw=4, linestyle='-', label=label1),
                Line2D([0], [0], color='orange', lw=4, linestyle='-', label=label2),
                Line2D([0], [0], color='#0d0787', lw=3, linestyle='--', label=label3),
                Line2D([0], [0], color='orange', lw=3, linestyle='--', label=label4)
            ]
        else:
            label1 = r"$\mathrm{SpokenWOZ}$" + "\n" + r"$(h=2560)$"
            label2 = r"$\mathrm{SpokenWOZ}$" + "\n" + r"$(h=960)$"
            label3 = r"$\mathrm{Switchboard}$" + "\n" + r"$(h=2560)$"
            label4 = r"$\mathrm{Switchboard}$" + "\n" + r"$(h=960)$"
            style_elements = [
                Line2D([0], [0], color='#0d0787', lw=4, linestyle='-', label=label1),
                Line2D([0], [0], color='orange', lw=4, linestyle='-', label=label2),
                Line2D([0], [0], color='#0d0787', lw=3, linestyle='--', label=label3),
                Line2D([0], [0], color='orange', lw=3, linestyle='--', label=label4)
            ]
        # if len(args.datasets) > 1:
        ax[0].legend(
            handles=style_elements, 
            ncol=4, 
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),  # ...is placed just outside the right edge (1.05) at the top (1)
            borderaxespad=0.
        )   

    # plt.tight_layout()
    fig.savefig(os.path.join(args.save_folder, f"{'_'.join(args.datasets)}_ep_cutoff_vs_median_forecast.png"), bbox_inches='tight', pad_inches=0.06)
    # add_to_wandb(f"{'_'.join(args.datasets)}_ep_cutoff_vs_median_forecast.png", fig)
    print(f"{'_'.join(args.datasets)}_ep_cutoff_vs_median_forecast.png")
    plt.clf()
    plt.close()
    
    
