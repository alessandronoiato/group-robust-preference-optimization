import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
project_name = "group-robustness-noisy-labels/common-good-ipo"
def plot_max_reward_err(data, deterministic_ratio):
    methods = ['IPO', 'GR-IPO', 'IS']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    means = []
    errors = []
    for method in methods:
        values = data[method]
        if values:
            means.append(np.mean(values))
            errors.append(np.std(values) / np.sqrt(len(values)))
        else:
            print(f"Warning: No data for method {method} at deterministic ratio {deterministic_ratio}")
            means.append(0)
            errors.append(0)

    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.arange(len(methods))
    width = 0.6

    bars = ax.bar(x, means, width, color=colors, yerr=errors, capsize=7)
    ax.set_ylabel('Converged Maximum Reward Error')
    ax.set_title(f'Comparison of Methods\n(Deterministic Ratio: {deterministic_ratio})')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0)

    # Add value labels above the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        error = errors[i]
        ax.text(bar.get_x() + bar.get_width() / 2, height + error + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # Adjust y-axis to make room for labels
    y_max = max(mean + error for mean, error in zip(means, errors))
    ax.set_ylim(0, y_max * 1.15)  # Add 15% padding above the highest point

    plt.tight_layout()
    
    # Get the current working directory
    current_dir = os.getcwd()
    plots_dir = os.path.join(current_dir, 'plots', 'max_reward_err')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_path = os.path.join(plots_dir, f'max_reward_err_comparison_det_ratio_{deterministic_ratio}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved at: {plot_path}")

if __name__ == "__main__":
    try:
        api = wandb.Api(timeout=60)
        project_name = "group-robustness-noisy-labels/common-good-ipo"
        runs = api.runs(project_name)

        print(f"Attempting to access project: {project_name}")
        print(f"Total number of runs found: {len(runs)}")

        max_reward_err_dict = defaultdict(lambda: defaultdict(list))
        deterministic_ratios = set()
        run_count = 0

        for run in runs:
            run_count += 1
            print(f"\nAnalyzing run {run_count}: {run.name}")
            
            deterministic_ratio = run.config.get('deterministic_ratio_list')
            dpo_type = run.config.get('dpo_type')
            importance_sampling = run.config.get('importance_sampling')
            importance_sampling_weights = run.config.get('importance_sampling_weights')
            
            print(f"  deterministic_ratio_list: {deterministic_ratio}")
            print(f"  dpo_type: {dpo_type}")
            print(f"  importance_sampling: {importance_sampling}")
            print(f"  importance_sampling_weights: {importance_sampling_weights}")

            if deterministic_ratio is not None:
                deterministic_ratios.add(tuple(deterministic_ratio))  # Convert list to tuple for set
                
                if dpo_type == 'rdpo':
                    if importance_sampling == True and importance_sampling_weights == '0.333,0.333,0.333':
                        method = 'IPO'
                    elif importance_sampling == False and importance_sampling_weights == 'None':
                        method = 'GR-IPO'
                    elif importance_sampling == True and importance_sampling_weights == 'None':
                        method = 'IS'
                    else:
                        print("  Skipping: Doesn't match any specified method")
                        continue
                    
                    print(f"  Matched method: {method}")
                    
                    # Get the last (converged) max_reward_err value
                    history = run.history(keys=['max_reward_err'])
                    if not history.empty:
                        converged_max_reward_err = history['max_reward_err'].iloc[-1]
                        max_reward_err_dict[tuple(deterministic_ratio)][method].append(converged_max_reward_err)
                        print(f"  Added max_reward_err: {converged_max_reward_err}")
                    else:
                        print(f"  Warning: No history data for run {run.name}")
                else:
                    print("  Skipping: dpo_type is not 'rdpo'")
            else:
                print("  Skipping: deterministic_ratio_list is None")

        print(f"\nDeterministic ratios found: {deterministic_ratios}")
        print(f"Data collected: {dict(max_reward_err_dict)}")

        for deterministic_ratio in sorted(deterministic_ratios):
            data = {method: max_reward_err_dict[deterministic_ratio][method] for method in ['IPO', 'GR-IPO', 'IS']}
            plot_max_reward_err(data, deterministic_ratio)

        print("Plots have been generated and saved in the 'plots/max_reward_err' directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")