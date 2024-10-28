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
PROJECT = "common-good-ipo-even"
# Set the fixed deterministic_ratio

DETERMINISTIC_RATIO = '1,0.3,1'  # Modify this value as needed
# Update SAVE_DIR to include the deterministic ratio
SAVE_DIR = os.path.join('plots', f'deterministic_ratio_{DETERMINISTIC_RATIO.replace(",", "_")}')
os.makedirs(SAVE_DIR, exist_ok=True)
logging.info(f"Save directory created: {os.path.abspath(SAVE_DIR)}")

def plot_loss(data, epsilon, save_path, deterministic_ratio):
    neatplot.set_style()
    
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    logging.debug(f"Entering plot_loss function with epsilon: {epsilon}")
    plt.figure(figsize=(10, 6))
    
    all_means = []
    all_errors = []
    
    for method, color in zip(['rdpo_with_is', 'rdpo_without_is'], ['blue', 'orange']):
        if method in data and epsilon in data[method]:
            mean = data[method][epsilon]['mean']
            std_error = data[method][epsilon]['std_error']
            iterations = range(1, len(mean) + 1)
            
            plt.plot(iterations, mean, label=f"{'IS' if method == 'rdpo_with_is' else 'GR-IPO'}", color=color, linewidth=0.5)
            plt.fill_between(iterations, mean - std_error, mean + std_error, alpha=0.2, color=color)
            
            all_means.extend(mean)
            all_errors.extend(std_error)

    if plt.gca().get_lines():
        plt.xlabel('Iteration')
        plt.ylabel('Max Validation Group Loss')
        plt.title(f'Max Validation Group Loss Comparison\n(Epsilon: {epsilon}, Deterministic Ratio: {deterministic_ratio})')
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        # Adjust y-axis limits
        if all_means and all_errors:
            min_val = min(all_means) - max(all_errors)
            max_val = max(all_means) + max(all_errors)
            y_range = max_val - min_val
            plt.ylim(min_val - 0.9 * y_range, max_val + 0.9 * y_range)
        
        neatplot.save_figure(save_path)
        plt.close()
        logging.info(f"Plot saved: {save_path}")
    else:
        logging.warning(f"No data to plot for epsilon: {epsilon}")

# Main execution
api = wandb.Api()

logging.info(f"Fetching runs from {ENTITY}/{PROJECT}")
runs = api.runs(f"{ENTITY}/{PROJECT}")
logging.info(f"Total number of runs fetched: {len(runs)}")

data = {
    'rdpo_with_is': {},
    'rdpo_without_is': {}
}

# Set the fixed deterministic_ratio

for run in runs:
    logging.info(f"Processing run: {run.name}")
    
    dpo_type = run.config.get('dpo_type')
    if dpo_type != 'rdpo':
        logging.info(f"Skipping run with dpo_type: {dpo_type}")
        continue

    is_enabled = run.config.get('importance_sampling', False)
    is_weights = run.config.get('importance_sampling_weights', '')
    
    logging.info(f"IS enabled: {is_enabled}, IS weights: {is_weights}")
    
    if is_enabled:
        method_key = 'rdpo_with_is'
    elif not is_enabled:
        method_key = 'rdpo_without_is'
    else:
        logging.info(f"Skipping run with IS enabled: {is_enabled} and weights: {is_weights}")
        continue

    deterministic_ratio_list = run.config.get('deterministic_ratio_list', [])
    deterministic_ratio = ','.join(map(str, deterministic_ratio_list))
    
    if deterministic_ratio != DETERMINISTIC_RATIO:
        logging.info(f"Skipping run with deterministic ratio: {deterministic_ratio}")
        continue

    epsilon = run.config.get('epsilon')
    if epsilon is None:
        logging.warning(f"Epsilon not found in config for run: {run.name}")
        continue
    
    logging.info(f"Epsilon: {epsilon}")
    
    history = run.history(keys=['max_val_grp_loss'])
    
    if history.empty or 'max_val_grp_loss' not in history.columns:
        logging.warning(f"'max_val_grp_loss' not found in history for run: {run.name}")
        continue
    
    max_val_grp_loss = history['max_val_grp_loss'].tolist()
    
    if epsilon not in data[method_key]:
        data[method_key][epsilon] = []
    data[method_key][epsilon].append(max_val_grp_loss)

logging.info(f"Processed data: {data.keys()}")
for method in data:
    logging.info(f"{method}: {data[method].keys()}")

# Calculate mean and standard error
for method in data:
    for epsilon in data[method]:
        losses = np.array(data[method][epsilon])
        mean = np.mean(losses, axis=0)
        std_error = sem(losses, axis=0)
        data[method][epsilon] = {
            'mean': mean,
            'std_error': std_error
        }

# Create plots
all_epsilons = set(data['rdpo_with_is'].keys()) | set(data['rdpo_without_is'].keys())
for epsilon in all_epsilons:
    save_path = os.path.join(SAVE_DIR, f'loss_comparison_epsilon_{epsilon}.png')
    plot_loss(data, epsilon, save_path, DETERMINISTIC_RATIO)

logging.info(f"Plotting complete. Plots saved in {os.path.abspath(SAVE_DIR)}")
