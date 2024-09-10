import argparse
import ast
import copy
import os

import numpy as np
import yaml

from algos.linear_bandit.group_dpo_vectorised import GroupDirectPolicyOptimizationVectorised as GDPO
from algos.linear_bandit.group_robust_dpo_vectorised_gradfix import (
    GroupRobustDirectPolicyOptimizationVectorised as GRDPO,
)
from algos.linear_bandit.mle import MLERewardLearning
from envs.group_linear_bandit import GroupLinearBandit as GLB
from envs.group_linear_bandit import GroupLinearBanditSep as GLBS
from envs.group_linear_bandit import ret_feature_func
from utils.collect_data import (
    collect_group_preference_data,
    collect_group_preference_data_partial_deterministic,
    collect_group_preference_data_partial_deterministic_list,
    collect_group_preference_data_wth_deterministic_list,
    collect_preference_data,
    collect_rl_data,
    collect_uneven_group_preference_data_partial_deterministic_list,
    merge_datasets,
    pref_to_rl,
    ret_uniform_policy_group,
)
from utils.io_utils import create_log_dir, save_code, save_config
from utils.logger import Logger
from utils.utils import return_apt_weights, softmax


def float_list(arg):
    try:
        # Using ast.literal_eval to safely evaluate the string as a Python expression
        # This will handle '[0.5,0.3]' correctly
        return [float(item) for item in ast.literal_eval(arg)]
    except (ValueError, SyntaxError) as e:
        print("ERROR: ", arg)
        print(e)
        raise argparse.ArgumentTypeError("Invalid list format or elements are not floats")


def set_reward_params(feature_dim: int):
    assert feature_dim in (4,)
    if feature_dim == 4:
        rparams = np.array(
            [
                [1.0, 3.0, 1.0, 3.0],
                [3.0, 1.0, 3.0, 1.0],
                [1.5, 2.5, 1.5, 2.5],
            ],
            np.float32,
        )
    return rparams


def ret_policy(action_num: int, feature_func, param):
    action_num = action_num
    feature_func = copy.deepcopy(feature_func)
    param = param

    def policy(state: np.ndarray, group_id: int) -> np.ndarray:
        arr = np.zeros(action_num, np.float32)
        for action_idx in range(action_num):
            feature = feature_func(state, action_idx, group_id)
            arr[action_idx] = np.dot(feature, param)
        prob = softmax(arr)

        return prob

    return policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="group_linear_bandit")
    parser.add_argument("--state_dim", type=int, default=1)
    parser.add_argument("--action_num", type=int, default=4)
    parser.add_argument("--group_num", type=int, default=2)
    parser.add_argument("--feature_type", type=str, default="same")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--agent", type=str, default="pg")
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--eval_metric", type=str, default="expectation")
    parser.add_argument("--eval_metric_prob", type=str, default="KL")

    parser.add_argument(
        "--deterministic_ratio_list",
        type=float_list,
        help="A list of determinisitic ratios as a string",
        default="[0,0]",
    )

    parser.add_argument("--val_deterministic", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument(
        "--val_deterministic_ratio_list",
        type=float_list,
        help="A list of determinisitic ratios as a string",
        default="[0,0]",
    )

    parser.add_argument("--pref_data_num", type=int, default=500)
    parser.add_argument("--weights", type=str, default="equal")

    parser.add_argument("--val_data_num", type=int, default=50)
    parser.add_argument("--val_weights", type=str, default="equal")

    parser.add_argument("--num_trials_for_eval", type=int, default=1000)
    parser.add_argument("--test_weights", type=str, default="equal")

    parser.add_argument("--mle_num_iters", type=int, default=100)
    parser.add_argument("--mle_adaptive", action="store_true")
    parser.add_argument("--mle_ada_coef", type=float, default=1.0)
    parser.add_argument("--mle_step_size", type=float, default=0.1)

    parser.add_argument("--reg_coef", type=float, default=1.0)

    parser.add_argument("--dpo_type", type=str, default="dpo")
    parser.add_argument("--dpo_num_iters", type=int, default=200)
    parser.add_argument("--exp_adaptive", type=float, default=0.0)
    parser.add_argument("--dpo_adaptive", action="store_true")
    parser.add_argument("--dpo_ada_coef", type=float, default=1.0)
    parser.add_argument("--dpo_step_size", type=float, default=0.1)
    parser.add_argument("--rdpo_batch_size", type=int, default=5)
    parser.add_argument("--rdpo_exp_step_size", type=float, default=0.01)
    parser.add_argument(
        "--rdpo_weighted_batches",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
    )
    parser.add_argument("--rdpo_adj", type=str, default="0")
    parser.add_argument(
        "--importance_sampling",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
    )
    parser.add_argument("--importance_sampling_weights", type=str, default="None")
    parser.add_argument("--ipo_grad_type", type=str, default="justdpo")
    parser.add_argument("--param_limit", type=int, default=1)
    parser.add_argument("--use_closed_form", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--lamba", type=float, default=0)
    parser.add_argument("--l2_reg_rdpo", type=float, default=0)
    parser.add_argument("--use_weight_val", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--use_uneven_grp", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--use_uneven_grp_val", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--use_theory", type=lambda x: (str(x).lower() == "true"), default=False)

    parser.add_argument("--wandb_use", action="store_true")
    parser.add_argument("--wandb_key", type=str, default="cd7b1433bad4eb38b457b881d01b17040a0b2432")
    parser.add_argument("--wandb_entity", type=str, default="robust-rl-project")
    parser.add_argument("--wandb_project", type=str, default="bandits_dpo")
    parser.add_argument("--wandb_group", type=str, default="group1")
    parser.add_argument("--wandb_name", type=str, default="linear_bandits")

    parser.add_argument("--chi", type=float, default=1.0)

    return parser.parse_args()


def main(args):
    np.random.seed(args.seed)

    # Logging
    log_dir = create_log_dir(args)
    save_code(log_dir)
    save_config(args.__dict__, log_dir)
    logger = Logger(log_dir)

    print(f"Logging to {log_dir}")
    print(f"(IB) Seed: {args.seed}")
    print(f"(IB) Data: {args.pref_data_num}")

    state_dim = args.state_dim
    action_num = args.action_num
    group_num = args.group_num
    num_trials_for_eval = args.num_trials_for_eval

    feature_dim = 2 * state_dim
    feature_func = ret_feature_func(
        num_action=action_num,
        state_dim=state_dim,
        group_num=group_num,
        feature_type=args.feature_type,
    )

    reward_param = set_reward_params(feature_dim=feature_dim)

    # WANDB setup
    print(f"{args.wandb_use=}")
    if args.wandb_use:
        print("USING WANDB")
        wandb.login(key=args.wandb_key)

        if args.dpo_adaptive:
            tags = [
                args.dpo_num_iters,
                f"adaptive_{args.dpo_adaptive}",
                args.ada_coef,
                args.reg_coef,
            ]
        else:
            tags = [
                f"num_iters_{args.dpo_num_iters}",
                f"adaptive_{args.dpo_adaptive}",
                f"step_size_{args.dpo_step_size}",
                f"beta_{args.reg_coef}",
            ]

        if args.dpo_type == "dpo":
            exp_name = f"{args.wandb_name}_{args.dpo_type}_{args.seed}"
        else:
            exp_name = f"{args.wandb_name}_{args.dpo_type}_{args.rdpo_exp_step_size}_{args.rdpo_batch_size}_{args.rdpo_weighted_batches}_{args.rdpo_adj}_{args.seed}"

        wandb_group = f"state_dim={args.state_dim},action_num={args.action_num},group_num{args.group_num},pref_data_num={args.pref_data_num},weights={args.weights},feature_type={args.feature_type},eval_metric={args.eval_metric},args.wandb_group"
        wandb.init(
            group=wandb_group,
            project=args.wandb_project,
            config=args.__dict__,
            dir=log_dir,
            name=exp_name,
            tags=tags,
        )

        wandb.config["true_reward_params"] = reward_param

    env = GLBS(
        state_dim=state_dim,
        action_num=action_num,
        group_num=group_num,
        reward_param=reward_param,
        feature_func=feature_func,
        num_trials_for_eval=num_trials_for_eval,
        eval_metric=args.eval_metric,
        eval_metric_prob=args.eval_metric_prob,
    )

    weights = return_apt_weights(args.weights, group_num)
    val_weights = return_apt_weights(args.val_weights, group_num)
    test_weights = return_apt_weights(args.test_weights, group_num)

    opt_policy = env.get_opt_policy()
    uniform_policy = ret_uniform_policy_group(action_num)

    if args.use_uneven_grp:
        pref_data = collect_uneven_group_preference_data_partial_deterministic_list(
            args.pref_data_num,
            env,
            weights,
            uniform_policy,
            deterministic_ratio_list=args.deterministic_ratio_list,
        )
    else:
        pref_data = collect_group_preference_data_partial_deterministic_list(
            args.pref_data_num,
            env,
            weights,
            uniform_policy,
            deterministic_ratio_list=args.deterministic_ratio_list,
        )

    if args.use_uneven_grp_val:
        val_pref = collect_uneven_group_preference_data_partial_deterministic_list(
            args.num_trials_for_eval,
            env,
            val_weights,
            uniform_policy,
            deterministic_ratio_list=args.val_deterministic_ratio_list,
        )
    else:
        val_pref = collect_group_preference_data_partial_deterministic_list(
            args.num_trials_for_eval,
            env,
            val_weights,
            uniform_policy,
            deterministic_ratio_list=args.val_deterministic_ratio_list,
        )

    test_pref = collect_group_preference_data(
        num=num_trials_for_eval,
        env=env,
        weights=test_weights,
        policy_func=uniform_policy,
        deterministic=True,
    )

    opt_reward = env.evaluate_reward_group_wise(policy=opt_policy, states=test_pref)
    uni_reward = env.evaluate_reward_group_wise(policy=uniform_policy, states=test_pref)

    formatted_opt_reward = ", ".join([f"{reward:.4f}" for reward in opt_reward])
    formatted_uni_reward = ", ".join([f"{reward:.4f}" for reward in uni_reward])
    logger.info(f"optimal policy reward: {formatted_opt_reward}, uniform policy reward: {uni_reward}.")

    # learn the reward function
    reward_model = MLERewardLearning(
        feature_func,
        feature_dim,
        args.mle_step_size,
        args.mle_num_iters,
        args.mle_adaptive,
        args.mle_ada_coef,
    )
    loss, l2_dist, acc = reward_model.train_by_cvxpy_group(dataset=pref_data, true_reward_param=reward_param)
    logger.info(f"Reward loss: {loss:.4f}, l2 distance: {l2_dist:.4f}, acc: {acc:.2f}.")

    learned_reward_param = reward_model.get_reward_param
    logger.info("True reward parameter: {}".format(reward_param))
    logger.info("Learned reward parameter: {}".format(learned_reward_param))

    # Oracle test
    learned_env = GLB(
        state_dim,
        action_num,
        group_num,
        learned_reward_param,
        feature_func,
        num_trials_for_eval=num_trials_for_eval,
    )
    learned_oracle_opt_policy = learned_env.get_opt_policy()
    learned_oracle_opt_reward = env.evaluate_reward_group_wise(policy=learned_oracle_opt_policy, states=test_pref)

    formatted_learned_oracle_opt_reward = ", ".join([f"{reward:.4f}" for reward in learned_oracle_opt_reward])
    logger.info(f"Learned oracle reward: {formatted_learned_oracle_opt_reward}")

    # Train the RL on the preference data
    logger.info(f"Train a policy solely on the preference data (DPO).")

    # learn the policy
    policy_feature_func = ret_feature_func(
        num_action=action_num,
        state_dim=state_dim,
        group_num=group_num,
        feature_type=args.feature_type,
    )
    if args.dpo_type == "dpo":
        agent = GDPO(
            state_dim=state_dim,
            action_num=action_num,
            group_num=group_num,
            feature_dim=feature_dim,
            feature_func=policy_feature_func,
            ref_policy=uniform_policy,
            reg_coef=args.reg_coef,
            step_size=args.dpo_step_size,
            num_iters=args.dpo_num_iters,
            is_adaptive=args.dpo_adaptive,
            ada_coef=args.dpo_ada_coef,
            logger=logger,
            wandb_use=args.wandb_use,
            ipo_grad_type=args.ipo_grad_type,
            param_limit=args.param_limit,
            lamba=args.lamba,
            report_iter=100,
        )
    elif args.dpo_type == "rdpo":
        agent = GRDPO(
            state_dim=state_dim,
            action_num=action_num,
            group_num=group_num,
            feature_dim=feature_dim,
            feature_func=policy_feature_func,
            ref_policy=uniform_policy,
            reg_coef=args.reg_coef,
            step_size=args.dpo_step_size,
            num_iters=args.dpo_num_iters,
            exp_adaptive=args.exp_adaptive,
            is_adaptive=args.dpo_adaptive,
            ada_coef=args.dpo_ada_coef,
            batch_size=args.rdpo_batch_size,
            exp_step_size=args.rdpo_exp_step_size,
            logger=logger,
            wandb_use=args.wandb_use,
            weighted_batches=args.rdpo_weighted_batches,
            adj=args.rdpo_adj,
            importance_sampling=args.importance_sampling,
            importance_sampling_weights=args.importance_sampling_weights,
            ipo_grad_type=args.ipo_grad_type,
            param_limit=args.param_limit,
            use_closed_form=args.use_closed_form,
            l2_reg_rdpo=args.l2_reg_rdpo,
            reg_by_group_weights=0,
            lamba=args.lamba,
            chi=args.chi,
            report_iter=100,
        )
    else:
        agent = GDPO(
            state_dim=state_dim,
            action_num=action_num,
            group_num=group_num,
            feature_dim=feature_dim,
            feature_func=policy_feature_func,
            ref_policy=uniform_policy,
            reg_coef=args.reg_coef,
            step_size=args.dpo_step_size,
            num_iters=args.dpo_num_iters,
            exp_adaptive=args.exp_adaptive,
            is_adaptive=args.dpo_adaptive,
            ada_coef=args.dpo_ada_coef,
            logger=logger,
            wandb_use=args.wandb_use,
            ipo_grad_type=args.ipo_grad_type,
            param_limit=args.param_limit,
            lamba=args.lamba,
            train_agent=False,  # random_train() func called instead of train()
            report_iter=100,
        )

    if agent.train_agent == True:
        if args.use_weight_val == False:
            reward = agent.train(
                dataset=pref_data,
                val_dataset=val_pref,
                test_dataset=test_pref,
                env=env,
                optimal_reward=opt_reward,
            )
        else:
            reward = agent.alternate_train(
                dataset=pref_data,
                weight_val_dataset=weight_val_pref,
                val_dataset=val_pref,
                test_dataset=test_pref,
                env=env,
                optimal_reward=opt_reward,
            )
    else:
        reward = agent.random_train(
            dataset=pref_data,
            val_dataset=val_pref,
            test_dataset=test_pref,
            env=env,
            optimal_reward=opt_reward,
        )
    formatted_reward = ", ".join([f"{reward:.4f}" for reward in reward])
    rew_error = [float((a - b) / a) for a, b in zip(opt_reward, reward)]
    formatted_rew_error = ", ".join([f"{reward:.4f}" for reward in rew_error])
    policy_param = agent.get_param
    logger.info(f"Policy parameter learned solely on the preference data {args.dpo_type}: {policy_param}.")
    logger.info(
        f"Training solely on the preference data {args.dpo_type}, dataset size: {len(pref_data): d}, optimal reward: {formatted_opt_reward}, reward: {formatted_reward}, reward error: {formatted_rew_error}."
    )

    rew_err_dict, rew_dict = dict(), dict()
    rew_err_dict[args.pref_data_num] = rew_error
    rew_dict[args.pref_data_num] = reward

    save_path = os.path.join(log_dir, f"reward_error_{args.dpo_type}.yml")
    yaml.dump(rew_err_dict, open(save_path, "w"), default_flow_style=False)
    save_path = os.path.join(log_dir, f"reward_{args.dpo_type}.yml")
    yaml.dump(rew_dict, open(save_path, "w"), default_flow_style=False)

    # calculating errors if param is known
    known_param_rewards = []
    known_param_rew_err = []
    for i in range(group_num):
        reward = env.evaluate_reward_group_wise(
            policy=ret_policy(action_num, policy_feature_func, reward_param[i]),
            states=test_pref,
        )
        reward_err = [float((a - b) / a) for a, b in zip(opt_reward, reward)]
        known_param_rewards.append(reward)
        known_param_rew_err.append(reward_err)
    logger.info(
        f"optimal reward: {formatted_opt_reward}, known_param_reward: {known_param_rewards}, Known param reward error: {known_param_rew_err}."
    )

    if args.wandb_use:
        d_wandb = {}
        # Assuming rew_err is a list
        for i, err in enumerate(rew_error):
            key = f"final/reward_err_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
            d_wandb[key] = err
        for i, param in enumerate(policy_param):
            key = f"final/reward_param_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
            d_wandb[key] = param
        for i, opt_r in enumerate(opt_reward):
            key = f"optimal_reward_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
            d_wandb[key] = opt_r
        for i, rew in enumerate(reward):
            key = f"final/reward_{i + 1}"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
            d_wandb[key] = rew
        for i, rew in enumerate(known_param_rewards):
            for j, r in enumerate(rew):
                key = f"reward_{j}_when_{i + 1}_group_param_known"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = r
        for i, err in enumerate(known_param_rew_err):
            for j, e in enumerate(err):
                key = f"reward_error_{j}_when_{i + 1}_group_param_known"  # Creating dynamic key, e.g., "reward_err_1", "reward_err_2", ...
                d_wandb[key] = e

        wandb.log(d_wandb)
        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
