import os
import wandb

import matplotlib.pyplot as plt
import neatplot
import numpy as np
from scipy.stats import sem
import pandas as pd
from typing import Dict, Any

ENTITY="group-robustness-noisy-labels"
PROJECT="common-good-ipo"

def plot_max_reward_error(data_dict, noise_level):
    fig, ax = plt.subplots(figsize=(10, 8))

    map_ = {'dpo': 'IPO', 'rdpo': 'GR-IPO'}
    methods = ['dpo', 'rdpo']
    x = np.arange(len(methods))
    width = 0.35

    for i, method in enumerate(methods):
        values = []
        for data in data_dict[method][noise_level].values():
            if isinstance(data, pd.DataFrame):
                last_non_nan = data.iloc[::-1].dropna().iloc[0].values[0]
            elif isinstance(data, pd.Series):
                last_non_nan = data.dropna().iloc[-1]
            elif isinstance(data, (np.ndarray, list)):
                last_non_nan = pd.Series(data).dropna().iloc[-1]
            elif isinstance(data, (float, np.float64)):
                last_non_nan = data
            else:
                print(f"Unexpected data type for {method} at noise level {noise_level}: {type(data)}")
                continue
            values.append(last_non_nan)
        
        if not values:
            print(f"No valid data for {method} at noise level {noise_level}")
            continue
        
        # Calculate mean and standard error
        mean = np.mean(values)
        std_error = np.std(values) / np.sqrt(len(values))
        
        # Plot bar
        ax.bar(i, mean, width, label=map_[method], yerr=std_error, capsize=5)

    ax.set_xlabel('Method', fontsize=30)
    ax.set_ylabel('Converged Max Reward Error', fontsize=30)
    ax.set_title(f'IPO vs GR-IPO: Converged Max Reward Error (Noise Level: {1-float(noise_level):.1f})', fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(map_.values(), fontsize=24)
    ax.legend(fontsize=24)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=24)

    plt.tight_layout()
    plt.savefig(f'plot/max_reward_error_noise_{1-float(noise_level):.1f}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    api = wandb.Api(timeout=30)
    filter_dict = {}

    runs = api.runs(f"{ENTITY}/{PROJECT}", filter_dict)
    methods = ('dpo', 'rdpo')
    noise_levels = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
    group_num = 3
    
    max_reward_err_dict = {method: {str(noise): {} for noise in noise_levels} for method in methods}

    for run in runs:
        noise_level = run.config["deterministic_ratio_list"][1]
        if noise_level not in noise_levels: 
            print(f"Skipping {run.name}")
            continue

        history = run.history()
        print(f"Using {run.name}")

        method = 'rdpo' if 'rdpo' in run.name else 'dpo'
        max_reward_err_dict[method][str(noise_level)][run.name] = history["max_reward_err"]

    # Create plot directory if it doesn't exist
    os.makedirs('plot', exist_ok=True)

    # Plot max validation group loss and max reward error (existing plots)
    for noise_level in noise_levels:
        plot_max_reward_error(max_reward_err_dict, str(noise_level))