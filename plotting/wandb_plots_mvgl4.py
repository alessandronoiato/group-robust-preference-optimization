import matplotlib
import matplotlib.pyplot as plt
import wandb
import numpy as np
from scipy.stats import sem
import neatplot
import os
import logging
import csv

logging.basicConfig(level=logging.INFO)

ENTITY = "group-robustness-noisy-labels"
PROJECT = "common-good-ipo-even"
SAVE_DIR = "plots/val_grp_loss"

def get_actual_iterations(method, start_iteration, num_points):
    if method in ['rdpo_with_is', 'rdpo_with_is_none']:
        return range(start_iteration, start_iteration + num_points * 100, 100)
    else:  # 'rdpo_without_is' and 'cgd'
        return range(start_iteration, start_iteration + num_points * 10, 10)

def get_start_index(method, start_iteration):
    if method in ['rdpo_with_is', 'rdpo_with_is_none']:
        return (start_iteration // 100) - 1
    else:  # 'rdpo_without_is' and 'cgd'
        return (start_iteration // 10) - 1

def plot_loss(data, deterministic_ratio, save_path, start_iteration=100):
    neatplot.set_style()
    
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    logging.debug(f"Entering plot_loss function with deterministic_ratio: {deterministic_ratio}")
    plt.figure(figsize=(10, 6))
    
    all_means = []
    all_errors = []
    
    methods = {
        'rdpo_with_is': ('IPO', 'blue'),
        'rdpo_without_is': ('GR-IPO', 'orange'),
        'rdpo_with_is_none': ('IS', 'green'),
        'cgd': ('CGD', 'red')
    }
    
    for method, (label, color) in methods.items():
        if method in data and deterministic_ratio in data[method]:
            start_index = get_start_index(method, start_iteration)
            mean = data[method][deterministic_ratio]['mean'][start_index:]
            std_error = data[method][deterministic_ratio]['std_error'][start_index:]
            actual_iterations = get_actual_iterations(method, start_iteration, len(mean))
            
            plt.plot(actual_iterations, mean, label=label, color=color, linewidth=0.5)
            plt.fill_between(actual_iterations, mean - std_error, mean + std_error, alpha=0.2, color=color)
            
            all_means.extend(mean)
            all_errors.extend(std_error)

    if plt.gca().get_lines():
        plt.xlabel('Iteration')
        plt.ylabel('Max Validation Group Loss')
        plt.title(f'Max Validation Group Loss Comparison\n(Deterministic Ratio: {deterministic_ratio})')
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

def write_data_to_csv(data, save_dir, start_iteration=100):
    csv_path = os.path.join(save_dir, 'plot_data.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['Method', 'Deterministic Ratio', 'Actual Iteration', 'Mean', 'Std Error']
        writer.writerow(header)
        
        for method in data:
            for det_ratio in data[method]:
                start_index = get_start_index(method, start_iteration)
                mean_values = data[method][det_ratio]['mean'][start_index:]
                std_error_values = data[method][det_ratio]['std_error'][start_index:]
                actual_iterations = get_actual_iterations(method, start_iteration, len(mean_values))
                
                for actual_iteration, mean, std_error in zip(actual_iterations, mean_values, std_error_values):
                    row = [method, det_ratio, actual_iteration, mean, std_error]
                    writer.writerow(row)
    
    logging.info(f"CSV file with plot data saved: {csv_path}")

# Main execution
api = wandb.Api()

logging.info(f"Fetching runs from {ENTITY}/{PROJECT}")
runs = api.runs(f"{ENTITY}/{PROJECT}")
logging.info(f"Total number of runs fetched: {len(runs)}")

data = {
    'rdpo_with_is': {},
    'rdpo_without_is': {},
    'rdpo_with_is_none': {},
    'cgd': {}  # Added CGD method
}

for run in runs:
    logging.info(f"Processing run: {run.name}")
    
    dpo_type = run.config.get('dpo_type')
    
    if dpo_type == 'rdpo':
        is_enabled = run.config.get('importance_sampling', False)
        is_weights = run.config.get('importance_sampling_weights', '')
        
        logging.info(f"IS enabled: {is_enabled}, IS weights: {is_weights}")
        
        if is_enabled and is_weights == '0.333,0.333,0.333':
            method_key = 'rdpo_with_is'
        elif is_enabled and is_weights == 'None':
            method_key = 'rdpo_with_is_none'
        elif not is_enabled:
            method_key = 'rdpo_without_is'
        else:
            logging.info(f"Skipping run with IS enabled: {is_enabled} and weights: {is_weights}")
            continue
    elif dpo_type == 'cgd':
        method_key = 'cgd'
    else:
        logging.info(f"Skipping run with dpo_type: {dpo_type}")
        continue

    deterministic_ratio_list = run.config.get('deterministic_ratio_list', [])
    deterministic_ratio = ','.join(map(str, deterministic_ratio_list))
    
    logging.info(f"Deterministic ratio: {deterministic_ratio}")
    
    history = run.history(keys=['max_val_grp_loss'])
    
    if history.empty or 'max_val_grp_loss' not in history.columns:
        logging.warning(f"'max_val_grp_loss' not found in history for run: {run.name}")
        continue
    
    max_val_grp_loss = history['max_val_grp_loss'].tolist()
    
    if deterministic_ratio not in data[method_key]:
        data[method_key][deterministic_ratio] = []
    data[method_key][deterministic_ratio].append(max_val_grp_loss)

logging.info(f"Processed data: {data.keys()}")
for method in data:
    logging.info(f"{method}: {data[method].keys()}")

# Calculate mean and standard error
for method in data:
    for deterministic_ratio in data[method]:
        losses = np.array(data[method][deterministic_ratio])
        mean = np.mean(losses, axis=0)
        std_error = sem(losses, axis=0)
        data[method][deterministic_ratio] = {
            'mean': mean,
            'std_error': std_error
        }

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)
logging.info(f"Save directory created: {os.path.abspath(SAVE_DIR)}")

# Write data to CSV
write_data_to_csv(data, SAVE_DIR, start_iteration=100)

# Create plots
all_ratios = set(data['rdpo_with_is'].keys()) | set(data['rdpo_without_is'].keys())
for deterministic_ratio in all_ratios:
    save_path = os.path.join(SAVE_DIR, f'loss_comparison_ratio_{deterministic_ratio}_even.png')
    plot_loss(data, deterministic_ratio, save_path, start_iteration=100)

logging.info(f"Plotting complete. Plots should be saved in {os.path.abspath(SAVE_DIR)}")
