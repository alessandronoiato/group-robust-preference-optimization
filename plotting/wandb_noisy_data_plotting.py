import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import multiprocessing
from tqdm import tqdm
from functools import partial
import logging

ENTITY = "anushkini"
PROJECT = "simple_noisy"
DPI = 100
METHODS = None
NOISE_LEVELS = None
PARALLELIZE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_wandb():
    return wandb.Api(timeout=30)

def get_runs(api, filter_dict=None):
    if filter_dict is None:
        filter_dict = {}
    if PROJECT == "simple_noisy":
        filter_dict["config.feature_type"] = "same"
    return api.runs(f"{ENTITY}/{PROJECT}", filter_dict)

def process_run_data(run, methods, noise_levels):
    data = {method: {str(noise): {} for noise in noise_levels} for method in methods}
    
    try:
        history = run.history()
        noise_level = run.config["deterministic_ratio_list"][1]
        
        if noise_level not in noise_levels:
            return None

        method = "cgd" if 'cgd' in run.name else \
                 "is_dpo" if "IS=True" in run.name else \
                 "rdpo" if "rdpo" in run.name else "dpo"
        
        if run.config["dpo_type"] == "rdpo" and run.config["ipo_grad_type"] == "noisy_dpo":
            method="noisy_rdpo"

        if "max_val_grp_loss" not in history:
            print(f"Skipping {run.name} because key is not in history")
            return None

        data[method][str(noise_level)][run.name] = history
        
        return data, method, str(noise_level)
    except Exception as e:
        print(f"Error processing run {run.name}: {str(e)}")
        return None

def plot_metric(ax, data_dict, noise_level, title, xlabel, ylabel, metric=None):
    map_ = {"dpo": "DPO", "rdpo": "GR-DPO", "is_dpo": "IS-DPO", "cgd": "CGD", "noisy_rdpo": "NOISY-GR-DPO"}
    colors = matplotlib.colormaps.get_cmap('Set3')(np.linspace(0, 1, len(map_)))
    
    for i, (method, method_name) in enumerate(map_.items()):
        all_dfs = [df[metric] if metric else df for df in data_dict[method][noise_level].values() if metric in df.columns or not metric]
        if not all_dfs:
            continue

        # Ensure all dataframes have the same index
        max_length = max(df.shape[0] for df in all_dfs)
        aligned_dfs = [df.reindex(range(max_length)) for df in all_dfs]

        # Calculate the mean across all runs
        mean_df = pd.concat(aligned_dfs, axis=1).mean(axis=1)

        # Calculate the standard error
        std_error = pd.concat(aligned_dfs, axis=1).std(axis=1) / np.sqrt(len(aligned_dfs))

        # Create x-axis values
        x_values = 100 * np.arange(len(mean_df))

        # Plot the mean line
        ax.plot(x_values, mean_df.values, label=method_name, color=colors[i])

        # Add shaded area for standard error
        ax.fill_between(x_values, mean_df.values - std_error.values, mean_df.values + std_error.values, alpha=0.3, color=colors[i])

    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    ax.set_title(title, fontsize=24)
    ax.legend(fontsize=24)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.tick_params(axis="both", which="major", labelsize=24)

def plot_and_save(data_dict, noise_level, plot_type, filename, metric=None):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    title_map = {
        "max_train_grp_loss": f"Max Train Group Loss (Noise Level: {1-float(noise_level):.1f})",
        "max_val_grp_loss": f"Max Val Group Loss (Noise Level: {1-float(noise_level):.1f})",
        "val_loss": f"Validation Loss (Noise Level: {1-float(noise_level):.1f})",
        "train_loss": f"Train Loss (Noise Level: {1-float(noise_level):.1f})",
        "train_group_loss": f"Train Group Loss - Group {{group}} (Noise Level: {1-float(noise_level):.1f})",
        "val_group_loss": f"Validation Group Loss - Group {{group}} (Noise Level: {1-float(noise_level):.1f})",
        "reward_err": f"Reward Error - Group {{group}} (Noise Level: {1-float(noise_level):.1f})",
        "group_weight": f"Group Weight - Group {{group}} (Noise Level: {1-float(noise_level):.1f})",
    }
    
    title = title_map.get(plot_type, f"{plot_type.capitalize()} (Noise Level: {1-float(noise_level):.1f})")
    if metric and metric[-1].isdigit():
        title = title.format(group=metric[-1])
    
    plot_metric(ax, data_dict, noise_level, title,
                "Iteration", plot_type.replace('_', ' ').capitalize(),
                metric)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close()

def plot_final_max_reward_error(data_dict, noise_level, filename):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    map_ = {"dpo": "DPO", "rdpo": "GR-DPO", "is_dpo": "IS-DPO", "cgd": "CGD", "noisy_rdpo": "NOISY-GR-DPO"}
    colors = matplotlib.colormaps.get_cmap('Set3')(np.linspace(0, 1, len(map_)))
    
    final_errors = []
    method_names = []
    
    for method, method_name in map_.items():
        all_dfs = [df['max_reward_err'] for df in data_dict[method][noise_level].values() if 'max_reward_err' in df.columns]
        if not all_dfs:
            continue
        
        final_values = [df.dropna().iloc[-1] if not df.dropna().empty else np.nan for df in all_dfs]
        final_error = np.nanmean(final_values)
        final_errors.append(final_error)
        method_names.append(method_name)
    
    ax.bar(method_names, final_errors, color=colors[:len(method_names)])
    
    ax.set_xlabel('Method', fontsize=30)
    ax.set_ylabel('Final Max Reward Error', fontsize=30)
    ax.set_title(f'Final Converged Max Reward Error (Noise Level: {1-float(noise_level):.1f})', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close()

def plot_group_weights(data_dict, noise_level, method, filename):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    metrics = ['group_weight_1', 'group_weight_2', 'group_weight_3']
    colors = ['r', 'g', 'b']
    
    has_data = False
    for metric, color in zip(metrics, colors):
        all_dfs = [df[metric] for df in data_dict[method][noise_level].values() if metric in df.columns]
        if not all_dfs:
            continue

        has_data = True
        max_length = max(df.shape[0] for df in all_dfs)
        aligned_dfs = [df.reindex(range(max_length)) for df in all_dfs]
        mean_df = pd.concat(aligned_dfs, axis=1).mean(axis=1)
        std_error = pd.concat(aligned_dfs, axis=1).std(axis=1) / np.sqrt(len(aligned_dfs))
        x_values = 100 * np.arange(len(mean_df))

        ax.plot(x_values, mean_df.values, label=f'Group {metric[-1]}', color=color)
        ax.fill_between(x_values, mean_df.values - std_error.values, mean_df.values + std_error.values, alpha=0.3, color=color)

    if has_data:
        ax.set_xlabel('Iteration', fontsize=30)
        ax.set_ylabel('Group Weight', fontsize=30)
        ax.set_title(f'{method.upper()} Group Weights (Noise Level: {1-float(noise_level):.1f})', fontsize=24)
        ax.legend(fontsize=24)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=24)

        plt.tight_layout()
        plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    else:
        print(f"No group weight data available for method {method} at noise level {noise_level}")
    
    plt.close()

def plot_cgd_rtg(data_dict, noise_level, filename):
    fig, axs = plt.subplots(3, 1, figsize=(14, 24))
    
    colors = ['r', 'g', 'b']
    
    for i in range(3):
        metrics = [f'rtg_{i+1}_1', f'rtg_{i+1}_2', f'rtg_{i+1}_3']
        
        for j, (metric, color) in enumerate(zip(metrics, colors)):
            all_dfs = [df[metric] for df in data_dict['cgd'][noise_level].values() if metric in df.columns]
            if not all_dfs:
                continue

            max_length = max(df.shape[0] for df in all_dfs)
            aligned_dfs = [df.reindex(range(max_length)) for df in all_dfs]
            mean_df = pd.concat(aligned_dfs, axis=1).mean(axis=1)
            std_error = pd.concat(aligned_dfs, axis=1).std(axis=1) / np.sqrt(len(aligned_dfs))
            x_values = 100 * np.arange(len(mean_df))

            axs[i].plot(x_values, mean_df.values, label=f'RTG {i+1}_{j+1}', color=color)
            axs[i].fill_between(x_values, mean_df.values - std_error.values, mean_df.values + std_error.values, alpha=0.3, color=color)

        axs[i].set_xlabel('Iteration', fontsize=20)
        axs[i].set_ylabel('RTG Values', fontsize=20)
        axs[i].set_title(f'CGD RTG for Group {i+1} (Noise Level: {1-float(noise_level):.1f})', fontsize=22)
        axs[i].legend(fontsize=16)
        axs[i].grid(True, linestyle='--', alpha=0.7)
        axs[i].tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close()

def main():
    api = setup_wandb()
    runs = get_runs(api)
    
    methods = ("dpo", "rdpo", "is_dpo", "cgd", "noisy_rdpo")
    noise_levels = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
    
    data_dict = {method: {str(noise): {} for noise in noise_levels} for method in methods}
    
    process_func = partial(process_run_data, methods=methods, noise_levels=noise_levels)

    if PARALLELIZE:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = list(tqdm(pool.imap(process_func, runs), total=len(runs), desc="Processing runs"))
    else:
        results = list(tqdm(map(process_func, runs), total=len(runs), desc="Processing runs"))

    for result in results:
        if result is not None:
            run_data, method, noise_level = result
            data_dict[method][noise_level].update(run_data[method][noise_level])

    os.makedirs("plot", exist_ok=True)

    for noise_level in noise_levels:
        noise_level_str = str(noise_level)
        plot_and_save(data_dict, noise_level_str, "max_train_grp_loss", 
                      f"plot/max_train_grp_loss_noise_{1-float(noise_level):.1f}.png", metric="max_train_grp_loss")
        plot_and_save(data_dict, noise_level_str, "max_val_grp_loss", 
                      f"plot/max_val_grp_loss_noise_{1-float(noise_level):.1f}.png", metric="max_val_grp_loss")
 
        plot_and_save(data_dict, noise_level_str, "val_loss", 
                      f"plot/val_loss_noise_{1-float(noise_level):.1f}.png", metric="val_loss")
        plot_and_save(data_dict, noise_level_str, "train_loss", 
                      f"plot/train_loss_noise_{1-float(noise_level):.1f}.png", metric="train_loss")
        
        # Separate plots for each group's train and validation losses
        for i in range(1, 4):  # Assuming 3 groups
            plot_and_save(data_dict, noise_level_str, "train_group_loss", 
                          f"plot/train_group_loss_{i}_noise_{1-float(noise_level):.1f}.png",
                          metric=f"train_group_loss_{i}")
            plot_and_save(data_dict, noise_level_str, "val_group_loss", 
                          f"plot/val_group_loss_{i}_noise_{1-float(noise_level):.1f}.png",
                          metric=f"val_group_loss_{i}")
            plot_and_save(data_dict, noise_level_str, "reward_err", 
                          f"plot/reward_err_{i}_noise_{1-float(noise_level):.1f}.png",
                          metric=f"reward_err_{i}")
 
        # Method-specific group weight plots
        for method in methods:
            if method not in ("dpo", "is_dpo"):  # Skip DPO and IS_DPO
                plot_group_weights(data_dict, noise_level_str, method,
                                   f"plot/{method}_group_weights_noise_{1-float(noise_level):.1f}.png")     
        
        # CGD-specific plot
        plot_cgd_rtg(data_dict, noise_level_str, 
                     f"plot/cgd_rtg_noise_{1-float(noise_level):.1f}.png")

        plot_final_max_reward_error(data_dict, noise_level_str,
                                    f"plot/final_max_reward_error_noise_{1-float(noise_level):.1f}.png")

if __name__ == "__main__":
    main()
