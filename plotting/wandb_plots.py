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
PROJECT="common-good-ipo"

plt.rcParams.update({'font.size': 16})  # Increase the default font size

def plot_loss(data_dict, noise_level, title, xlabel, ylabel, filename):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    map_ = {'dpo': 'DPO', 'rdpo': 'GR-DPO'}
    for method in ['dpo', 'rdpo']:
        # Combine all dataframes for this method and noise level
        all_dfs = list(data_dict[method][noise_level].values())
        
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
        ax.plot(x_values, mean_df.values, label=map_[method])
        
        # Add shaded area for standard error
        ax.fill_between(x_values, 
                        mean_df.values - std_error.values,
                        mean_df.values + std_error.values,
                        alpha=0.3)

    ax.set_xlabel(xlabel, fontsize=30) # ax.set_xlabel('Iteration', fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30) # ax.set_ylabel('Max Validation Group Loss', fontsize=30)
    ax.set_title(title, fontsize=24) # ax.set_title(f'DPO vs GR-DPO: Max Validation Group Loss (Noise Level: {1-float(noise_level):.1f})', fontsize=24)
    ax.legend(fontsize=24)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=24)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight') # plt.savefig(f'plot/max_val_group_error_noise_{1-float(noise_level):.1f}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    api = wandb.Api(timeout=30)

    runs = api.runs(f"{ENTITY}/{PROJECT}")#, filter_dict#)
    methods = ('dpo', 'rdpo')
    noise_levels = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
    
    max_val_group_loss_dict = {method: {str(noise): {} for noise in noise_levels} for method in methods}

    for run in runs:
        noise_level = run.config["deterministic_ratio_list"][1] # This is how you can filter runs

        history = run.history()
        print(f"Using {run.name}")

        method = 'rdpo' if 'rdpo' in run.name else 'dpo'

        max_val_group_loss_dict[method][str(noise_level)][run.name] = history["max_val_grp_loss"]
        
    # Create plot directory if it doesn't exist
    os.makedirs('plot', exist_ok=True)

    # Plot max validation group loss and max reward error (existing plots)
    for noise_level in noise_levels:
        plot_loss(
            data_dict=max_val_group_loss_dict,
            noise_level=str(noise_level),
            title=f'Max Validation Group Loss (Noise Level: {1-float(noise_level):.1f})',
            xlabel='Iteration',
            ylabel='Max Validation Group Loss',
            filename=f'plot/max_val_group_loss_noise_{1-float(noise_level):.1f}.png',
        )