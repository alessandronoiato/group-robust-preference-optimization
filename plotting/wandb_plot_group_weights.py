import matplotlib
import matplotlib.pyplot as plt
import wandb
import numpy as np
from scipy.stats import sem
import neatplot
import os
import logging

logging.basicConfig(level=logging.INFO)

ENTITY = "group-robustness-noisy-labels"
PROJECT = "common-good-ipo"
METHOD = "is"  # Change this to the desired method: "ipo", "is", or "gripo"
SAVE_DIR = os.path.join("plots", f"{METHOD}_group_weights")

def plot_group_weights(data, deterministic_ratio, save_path):
    neatplot.set_style()
    
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    logging.debug(f"Entering plot_group_weights function with deterministic_ratio: {deterministic_ratio}")
    plt.figure(figsize=(10, 6))
    
    all_means = []
    all_errors = []
    
    group_weights = {
        'group_weight_1': ('Group 1', 'blue'),
        'group_weight_2': ('Group 2', 'orange'),
        'group_weight_3': ('Group 3', 'green')
    }
    
    for weight, (label, color) in group_weights.items():
        if weight in data and deterministic_ratio in data[weight]:
            mean = data[weight][deterministic_ratio]['mean']
            std_error = data[weight][deterministic_ratio]['std_error']
            iterations = range(1, len(mean) + 1)
            
            plt.plot(iterations, mean, label=label, color=color, linewidth=0.5)
            plt.fill_between(iterations, mean - std_error, mean + std_error, alpha=0.2, color=color)
            
            all_means.extend(mean)
            all_errors.extend(std_error)

    if plt.gca().get_lines():
        plt.xlabel('Iteration')
        plt.ylabel('Group Weight')
        plt.title(f'Group Weights Comparison for {METHOD.upper()}\n(Deterministic Ratio: {deterministic_ratio})')
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        # Adjust y-axis limits
        if all_means and all_errors:
            min_val = min(all_means) - max(all_errors)
            max_val = max(all_means) + max(all_errors)
            y_range = max_val - min_val
            plt.ylim(min_val - 0.1 * y_range, max_val + 0.1 * y_range)
        
        neatplot.save_figure(save_path)
        plt.close()
        logging.info(f"Plot saved: {save_path}")
    else:
        logging.warning(f"No data to plot for deterministic ratio: {deterministic_ratio}")

# Main execution
api = wandb.Api()

logging.info(f"Fetching runs from {ENTITY}/{PROJECT}")
runs = api.runs(f"{ENTITY}/{PROJECT}")
logging.info(f"Total number of runs fetched: {len(runs)}")

data = {
    'group_weight_1': {},
    'group_weight_2': {},
    'group_weight_3': {}
}

for run in runs:
    logging.info(f"Processing run: {run.name}")
    
    dpo_type = run.config.get('dpo_type')
    is_enabled = run.config.get('importance_sampling', False)
    is_weights = run.config.get('importance_sampling_weights', '')
    
    # Determine if this run matches the METHOD we want
    if METHOD == 'ipo' and dpo_type == 'rdpo' and is_enabled and is_weights == '0.333,0.333,0.333':
        pass
    elif METHOD == 'is' and dpo_type == 'rdpo' and is_enabled and is_weights == 'None':
        pass
    elif METHOD == 'gripo' and dpo_type == 'rdpo' and not is_enabled:
        pass
    else:
        logging.info(f"Skipping run that doesn't match METHOD: {METHOD}")
        continue

    deterministic_ratio_list = run.config.get('deterministic_ratio_list', [])
    deterministic_ratio = ','.join(map(str, deterministic_ratio_list))
    
    logging.info(f"Deterministic ratio: {deterministic_ratio}")
    
    history = run.history(keys=['group_weight_1', 'group_weight_2', 'group_weight_3'])
    
    for group in ['group_weight_1', 'group_weight_2', 'group_weight_3']:
        if group not in history.columns:
            logging.warning(f"'{group}' not found in history for run: {run.name}")
            continue
        
        group_weight = history[group].tolist()
        
        if deterministic_ratio not in data[group]:
            data[group][deterministic_ratio] = []
        data[group][deterministic_ratio].append(group_weight)

logging.info(f"Processed data: {data.keys()}")
for group in data:
    logging.info(f"{group}: {data[group].keys()}")

# Calculate mean and standard error
for group in data:
    for deterministic_ratio in data[group]:
        weights = np.array(data[group][deterministic_ratio])
        mean = np.mean(weights, axis=0)
        std_error = sem(weights, axis=0)
        data[group][deterministic_ratio] = {
            'mean': mean,
            'std_error': std_error
        }

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)
logging.info(f"Save directory created: {os.path.abspath(SAVE_DIR)}")

# Create plots
all_ratios = set()
for group in data:
    all_ratios.update(data[group].keys())

for deterministic_ratio in all_ratios:
    save_path = os.path.join(SAVE_DIR, f'group_weights_comparison_ratio_{deterministic_ratio}.png')
    plot_group_weights(data, deterministic_ratio, save_path)

logging.info(f"Plotting complete. Plots should be saved in {os.path.abspath(SAVE_DIR)}")
