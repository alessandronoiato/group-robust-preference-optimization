import argparse
import yaml
import os
from datetime import datetime

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def generate_command(base_config, method_config, feature_type, noise_level, seed):
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_dir = f"log_{method_config['dpo_type']}/{timestamp}/{feature_type}_noise{noise_level}_[1,{noise_level},1]_{seed}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/log", exist_ok=True)

    wandb_name = f"{method_config['dpo_type']},{base_config['step_size']},{base_config['reg_coef']},{base_config['exp_step_size']},[1,{noise_level},1],{seed},IS={method_config['importance_sampling']}"

    cmd = [
        "python", "-m", "experiments.run_glb_noisy",
        "--mle_adaptive",
        f"--state_dim={base_config['state_dim']}",
        f"--action_num={base_config['action_num']}",
        f"--group_num={base_config['group_num']}",
        f"--pref_data_num={base_config['pref_data_num']}",
        f"--dpo_num_iters={base_config['dpo_num_iters']}",
        f"--wandb_use",
        f"--wandb_group={base_config['wandb']['group']}",
        f"--wandb_project={base_config['wandb']['project']}",
        f"--reg_coef={base_config['reg_coef']}",
        f"--seed={seed}",
        f"--weights={base_config['weights']}",
        f"--val_weights={base_config['val_weights']}",
        f"--test_weights={base_config['test_weights']}",
        f"--log_dir={log_dir}",
        f"--dpo_type={method_config['dpo_type']}",
        f"--dpo_step_size={base_config['step_size']}",
        f"--rdpo_exp_step_size={base_config['exp_step_size']}",
        f"--rdpo_batch_size={base_config['batch_size']}",
        f"--feature_type={feature_type}",
        f"--rdpo_weighted_batches={base_config.get('weighted_batches', 'False')}",
        f"--exp_adaptive={base_config.get('exp_adaptive', '0')}",
        f"--rdpo_adj={base_config.get('rdpo_adj', '0')}",
        f"--eval_metric={base_config['eval_metric']}",
        f"--importance_sampling={method_config['importance_sampling']}",
        f"--importance_sampling_weights={base_config.get('importance_sampling_weights', 'None')}",
        f"--ipo_grad_type={method_config.get('ipo_grad_type', '')}",
        f"--param_limit={base_config['param_limit']}",
        f"--use_closed_form={base_config.get('use_closed_form', 'False')}",
        f"--val_deterministic={base_config.get('val_deterministic', 'false')}",
        f"--deterministic_ratio_list=[1,{noise_level},1]",
        f"--val_deterministic_ratio_list=[1,1,1]",
        f"--lamba={base_config.get('lamba', '0')}",
        f"--l2_reg_rdpo={base_config.get('l2_reg_rdpo', '0')}",
        f"--use_uneven_grp={base_config.get('use_uneven_grp', 'False')}",
        f"--use_uneven_grp_val={base_config.get('use_uneven_grp_val', 'False')}",
        f"--use_theory={base_config.get('use_theory', 'False')}",
        f"--wandb_logdir={log_dir}/log",
        f"--wandb_name={wandb_name}",
        f"--C={base_config['C']}",
        f"--chi={base_config['chi']}",
        f"--reward_param={base_config['reward_param']}"
    ]
    return ' '.join(cmd)

def main(args):
    base_config = load_config(args.base_config)
    method_configs = [load_config(f) for f in args.method_configs]
    
    with open('commands.txt', 'w') as f:
        for method_config in method_configs:
            for feature_type in base_config['feature_types']:
                for noise_level in base_config['noise_levels']:
                    for seed in base_config['seeds']:
                        cmd = generate_command(base_config, method_config, feature_type, noise_level, seed)
                        f.write(f"{cmd}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate commands for experiments")
    parser.add_argument('--base_config', type=str, required=True, help='Path to base configuration file')
    parser.add_argument('--method_configs', type=str, nargs='+', required=True, help='Paths to method-specific configuration files')
    args = parser.parse_args()
    main(args)
