import os
import wandb

import matplotlib.pyplot as plt
import neatplot
import numpy as np
from scipy.stats import sem
import pandas as pd
from typing import Dict, Any

neatplot.set_style()

# Constants and configurations
ENTITY="group-robustness-noisy-labels"
PROJECT="common-good-ipo-even"

plt.rcParams.update({'font.size': 16})  # Increase the default font size

def plot_loss(data_dict, deterministic_ratio, exp_step_size, title, xlabel, ylabel, filename):
    all_dfs = list(data_dict[deterministic_ratio][exp_step_size].values())
    
    if not all_dfs:
        print(f"No data available for deterministic ratio {deterministic_ratio} and exp_step_size {exp_step_size}")
        return  # Exit the function if there's no data to plot

    # Ensure all dataframes have the same index
    max_length = max(df.shape[0] for df in all_dfs)
    aligned_dfs = [df.reindex(range(max_length)) for df in all_dfs]
    
    # Calculate the mean across all runs
    mean_df = pd.concat(aligned_dfs, axis=1).mean(axis=1)
    
    # Calculate the standard error
    std_error = pd.concat(aligned_dfs, axis=1).std(axis=1) / np.sqrt(len(aligned_dfs))
    
    # Create x-axis values (100 iterations)
    x_values = np.arange(len(mean_df))
    
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the mean line
    ax.plot(x_values, mean_df.values, label='RDPO')
    
    # Add shaded area for standard error
    ax.fill_between(x_values, 
                    mean_df.values - std_error.values,
                    mean_df.values + std_error.values,
                    alpha=0.3)

    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    ax.set_title(title, fontsize=24)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(fontsize=20)

    # Set x-axis ticks to show every 25 iterations
    ax.set_xticks(np.arange(0, 101, 25))
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory

if __name__ == "__main__":
    api = wandb.Api(timeout=60)

    runs = api.runs(f"{ENTITY}/{PROJECT}")#, filter_dict#)
    deterministic_ratios = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
    rdpo_exp_step_sizes = (0.1,)  # Changed variable name

    max_val_group_loss_dict = {str(ratio): {str(step): {} for step in rdpo_exp_step_sizes} for ratio in deterministic_ratios}

    for run in runs:
        if 'rdpo' not in run.name:
            continue

        deterministic_ratio = run.config["deterministic_ratio_list"][1]
        rdpo_exp_step_size = run.config.get("rdpo_exp_step_size", None)  # Changed to rdpo_exp_step_size

        if rdpo_exp_step_size is None:
            continue

        history = run.history()
        print(f"Using {run.name}")

        max_val_group_loss_dict[str(deterministic_ratio)][str(rdpo_exp_step_size)][run.name] = history["max_val_grp_loss"]
        
    # Create plot directory if it doesn't exist
    os.makedirs('plot', exist_ok=True)

    # Plot max validation group loss for each deterministic ratio and rdpo_exp_step_size
    for deterministic_ratio in deterministic_ratios:
        for rdpo_exp_step_size in rdpo_exp_step_sizes:
            if not max_val_group_loss_dict[str(deterministic_ratio)][str(rdpo_exp_step_size)]:
                print(f"Skipping plot for deterministic ratio {deterministic_ratio} and rdpo_exp_step_size {rdpo_exp_step_size} due to lack of data")
                continue

            plot_loss(
                data_dict=max_val_group_loss_dict,
                deterministic_ratio=str(deterministic_ratio),
                exp_step_size=str(rdpo_exp_step_size),
                title=f'Max Validation Group Loss (Det. Ratio: {float(deterministic_ratio):.1f}, RDPO Exp Step Size: {rdpo_exp_step_size})',
                xlabel='Iteration',
                ylabel='Max Validation Group Loss',
                filename=f'plot/max_val_group_loss_det_ratio_{float(deterministic_ratio):.1f}_rdpo_step_{rdpo_exp_step_size}.png',
            )

    # After the loop, print a summary of available data
    print("\nSummary of available data:")
    for deterministic_ratio in deterministic_ratios:
        for rdpo_exp_step_size in rdpo_exp_step_sizes:
            count = len(max_val_group_loss_dict[str(deterministic_ratio)][str(rdpo_exp_step_size)])
            print(f"Deterministic ratio {deterministic_ratio}, RDPO Exp step size {rdpo_exp_step_size}: {count} runs")