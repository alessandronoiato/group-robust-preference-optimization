import copy
from collections import defaultdict
import random
from typing import List, Union

import numpy as np
import torch as t
import wandb
t.set_grad_enabled(True)
from envs.group_linear_bandit import GroupLinearBandit
from utils.collect_data import GroupTransition
from utils.logger import Logger
from utils.utils import sigmoid, softmax


class CommonGradientDescent:
    def __init__(
        self,
        state_dim: int,  # state s drawn as a vector of `state_dim` elements from Uniform(0,1)
        action_num: int,  ## number of actions in Action Space
        group_num: int,  ## number of groups
        feature_dim: int,  ## feature_dim = 2 * state_dim (num elements in vector φ(s,a,g) )
        feature_func,  ## φ(s,a,g)
        ref_policy,  ## π_ref(a|s)
        reg_coef: float,  ## β scaling in the DPO gradient & loss -- controls KL Divergence from π_ref
        step_size: float,  ## η_θ step size for Gradient Descent on the DPO/IPO loss (if not is_adaptive)
        C: float,  ## Group adjustment hyperparameter, Section 2 of paper
        ipo_grad_type: str,  ## `justdpo`, `linear` (IPO), 
        num_iters: int,  ## number of update steps on Training dataset
        batch_size: int,  ## batch computation instead of for-loop over each datapoint in D_pref
        logger: Logger = None,  ## logger
        wandb_use: bool = False,  ## recording results in WandB
        use_closed_form: bool = False,
        param_limit: int = 1,  ## elements of vector θ range in [0, param_limit]
        report_iter: int = 10,  ## log metrics after these iters
        seed: int = None,  ## Seed
    ) -> None:
        print(f"RUNNING CGD STEP_SIZE={step_size} C={C} REG_COEG={reg_coef} SEED={seed}")
        self.state_dim = state_dim
        self.action_num = action_num
        self.group_num = group_num
        self.feature_dim = feature_dim
        self.feature_func = feature_func
        self.ref_policy = ref_policy
        self.reg_coef = reg_coef
        self.step_size = step_size
        self.use_closed_form = use_closed_form
        self.C = C
        self.ipo_grad_type = ipo_grad_type
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.logger = logger
        self.wandb_use = wandb_use
        self.param_limit = param_limit
        self.report_iter = report_iter
        self.seed = seed

        if seed is not None:
            t.manual_seed(seed)

        self.np_float = np.float32
        self.t_float = t.float32

        # Initialize the policy parameter
        self.param = t.FloatTensor(feature_dim).uniform_(0, self.param_limit)
        self.param.requires_grad = True

        # Initialize the optimizer
        self.optimizer = t.optim.SGD([self.param], lr=self.step_size)

    def ret_action_prob(self, state: np.ndarray, group_id: int) -> np.ndarray:
        arr = np.zeros(self.action_num, self.np_float)
        param = self.param.detach().numpy().astype(self.np_float)

        for action_idx in range(self.action_num):
            feature = self.feature_func(state, action_idx, group_id)
            arr[action_idx] = np.dot(feature, self.param)

        prob = softmax(arr)
        return prob

    def ret_policy(self):
        action_num = self.action_num
        feature_func = copy.deepcopy(self.feature_func)
        param = self.param.detach().numpy().astype(self.np_float)

        def policy(state: np.ndarray, group_id: int) -> np.ndarray:
            arr = np.zeros(action_num, self.np_float)
            for action_idx in range(action_num):
                feature = feature_func(state, action_idx, group_id)
                arr[action_idx] = np.dot(feature, param)
            prob = softmax(arr)

            return prob

        return policy

    def sample_action(self, state: np.ndarray, group_id: int) -> int:
        prob = self.action_prob(state, group_id)
        sampled_act = np.random.choice(a=self.action_num, size=1, replace=True, p=prob)
        return sampled_act

    def evaluate_grp_loss(self, dataset: List[GroupTransition]) -> float:
        if self.ipo_grad_type == "justdpo":
            print("running cgd with DPO loss")
            loss = t.zeros(self.group_num)
            counts = t.zeros(self.group_num)

            group_id_idx_all = defaultdict(list)
            feature_diff_all = t.zeros((len(dataset), self.feature_dim))

            for idx, transition in enumerate(dataset):
                state, action_one, action_two, group_id, pref = (
                    transition.state,
                    transition.action_0,
                    transition.action_1,
                    transition.group_id,
                    transition.pref,
                )
                pref_act = action_two if pref == 1 else action_one
                non_pref_act = action_two if pref == 0 else action_one

                feat_pref_act, feat_non_pref_act = (
                    t.tensor(self.feature_func(state, pref_act, group_id), dtype=self.t_float),
                    t.tensor(self.feature_func(state, non_pref_act, group_id), dtype=self.t_float),
                )

                feature_diff_all[idx, :] = feat_pref_act - feat_non_pref_act

                group_id_idx_all[group_id].append(idx)  # get dataset indices for each group
                counts[group_id] += 1

            log_ratio_diff = self.reg_coef * feature_diff_all @ self.param.reshape(self.feature_dim, 1)

            for group_id in range(self.group_num):
                group_indices = group_id_idx_all[group_id]
                loss[group_id] = t.sum(-t.log(t.sigmoid(log_ratio_diff[group_indices])))

        
        elif self.ipo_grad_type == "linear":
            print("running cgd with IPO loss")
            #if policy is None:
            #    policy = self.ret_policy()

            loss = t.zeros(self.group_num, dtype=self.t_float)
            counts = t.zeros(self.group_num, dtype=t.float32)

            group_id_idx_all = {i: [] for i in range(self.group_num)}
            feature_diff_all = t.zeros((len(dataset), self.feature_dim), dtype=t.float32)
            pref_act_all = []
            non_pref_act_all = []
            #eval_policy_act_prob_all = t.zeros((len(dataset), self.action_num), dtype=t.float32)
            #ref_policy_act_prob_all = t.zeros((len(dataset), self.action_num), dtype=t.float32)

            for idx, transition in enumerate(dataset):
                state, action_one, action_two, group_id, pref = (
                    transition.state,
                    transition.action_0,
                    transition.action_1,
                    transition.group_id,
                    transition.pref,
                )
                pref_act = action_two if pref == 1 else action_one
                non_pref_act = action_two if pref == 0 else action_one

                #pref_act_all.append(pref_act)
                #non_pref_act_all.append(non_pref_act)

                feat_pref_act, feat_non_pref_act = (
                    t.tensor(self.feature_func(state, pref_act, group_id), dtype=t.float32),
                    t.tensor(self.feature_func(state, non_pref_act, group_id), dtype=t.float32),
                )
                feature_diff_all[idx, :] = feat_pref_act - feat_non_pref_act

                group_id_idx_all[group_id].append(idx)
                counts[group_id] += 1
            
            #param = t.tensor(self.param, dtype=t.float32)

            #lin_diff = t.matmul(feature_diff_all, param.unsqueeze(1)) - 0.5 * (1 / self.reg_coef)
            #coef = lin_diff
            lin_diff = feature_diff_all @ self.param.reshape(self.feature_dim, 1) - 0.5 * (1 / self.reg_coef)
            coef = lin_diff

            for group_id in range(self.group_num):
                group_indices = t.tensor(group_id_idx_all[group_id], dtype=t.long)
                loss[group_id] = t.sum(t.square(coef[group_indices])) #+ self.adj[group_id] / t.sqrt(self.group_counts[group_id])
        else:
            raise ValueError(f"Invalid ipo_grad_type: {self.ipo_grad_type}")
        
        loss = loss / counts

        return loss

    def evaluate_loss(self, dataset: List[GroupTransition]) -> float:
        feature_diff_all = t.zeros((len(dataset), self.feature_dim))

        for idx, transition in enumerate(dataset):
            state, action_one, action_two, group_id, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.group_id,
                transition.pref,
            )
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one

            feat_pref_act, feat_non_pref_act = (
                t.tensor(self.feature_func(state, pref_act, group_id), dtype=self.t_float),
                t.tensor(self.feature_func(state, non_pref_act, group_id), dtype=self.t_float),
            )
            feature_diff_all[idx, :] = feat_pref_act - feat_non_pref_act

        log_ratio_diff = self.reg_coef * feature_diff_all @ self.param.reshape(self.feature_dim, 1)
        loss = t.sum(-t.log(t.sigmoid(log_ratio_diff))) / len(dataset)

        return loss

    def evaluate_weighted_loss(self, dataset: List[GroupTransition]) -> float:
        loss = 0.0

        group_id_idx_all = defaultdict(list)
        feature_diff_all = t.zeros((len(dataset), self.feature_dim))

        for idx, transition in enumerate(dataset):
            state, action_one, action_two, group_id, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.group_id,
                transition.pref,
            )
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one

            feat_pref_act, feat_non_pref_act = (
                t.tensor(self.feature_func(state, pref_act, group_id), dtype=self.t_float),
                t.tensor(self.feature_func(state, non_pref_act, group_id), dtype=self.t_float),
            )

            feature_diff_all[idx, :] = feat_pref_act - feat_non_pref_act

            group_id_idx_all[group_id].append(idx)  # get dataset indices for each group

        log_ratio_diff = self.reg_coef * feature_diff_all @ self.param.reshape(self.feature_dim, 1)

        for group_id in range(self.group_num):
            group_indices = group_id_idx_all[group_id]
            loss += t.sum(-self.group_weights[group_id] * t.log(t.sigmoid(log_ratio_diff[group_indices])))

        loss /= len(dataset)
        loss = loss * self.group_num
        return loss

    def train(
        self,
        dataset: List[GroupTransition],
        val_dataset: List[GroupTransition],
        test_dataset: List[GroupTransition],
        env: GroupLinearBandit,
        optimal_reward: List[float],
    ) -> float:
        ratio = int(len(dataset) / self.batch_size)

        group_counts = [0 for i in range(self.group_num)]
        for transition in dataset:
            group_counts[transition.group_id] += 1

        group_weights = t.exp(self.C / t.sqrt(t.tensor(group_counts, dtype=self.t_float)))
        self.initial_group_weights = group_weights / group_weights.sum()

        self.group_weights = t.autograd.Variable(self.initial_group_weights.clone(), requires_grad=False)
        self.alpha = t.autograd.Variable(t.ones(self.group_num) / self.group_num, requires_grad=True)

        print(f"INITIAL GROUP WEIGHTS: {self.initial_group_weights}")

        for step in range(ratio * self.num_iters):
            self.batch_update_once(dataset=dataset, batch_size=self.batch_size)

            if step % self.report_iter == 0:
                self.report_metrics(step, dataset, val_dataset, test_dataset, env, optimal_reward)

        self.report_metrics(step, dataset, val_dataset, test_dataset, env, optimal_reward)

        return self.evaluate_reward(env, test_dataset)

    def report_metrics(self, step, dataset, val_dataset, test_dataset, env, optimal_reward):
        with t.no_grad():
            grad = self.param.grad.clone()
            grad_norm = t.norm(grad).item()
            live_grad = 0.0

            param = self.param.detach().tolist()
            group_weights = self.group_weights.detach().tolist()

            # Calculate losses
            train_loss = self.evaluate_loss(dataset).item()
            val_loss = self.evaluate_loss(val_dataset).item()

            train_wt_loss = self.evaluate_weighted_loss(dataset).item()
            val_wt_loss = self.evaluate_weighted_loss(val_dataset).item()

            train_grp_loss = self.evaluate_grp_loss(dataset).tolist()
            val_grp_loss = self.evaluate_grp_loss(val_dataset).tolist()

            kl_dist = self.evaluate_KL(env, test_dataset)
            formatted_kl = ", ".join([f"{kld:.4f}" for kld in kl_dist])

            test_reward = self.evaluate_reward(env=env, states=test_dataset)
            rew_err = [float(a - b) / a for a, b in zip(optimal_reward, test_reward)]
            formatted_rew = ", ".join([f"{reward:.4f}" for reward in rew_err])

            max_reward_err = max(rew_err)
            max_reward_err_index = rew_err.index(max_reward_err)

            max_kl_dist = max(kl_dist)
            max_kl_dist_index = kl_dist.index(max_kl_dist)

            max_train_grp_loss = max(train_grp_loss)
            max_val_grp_loss = max(val_grp_loss)
            max_train_grp_loss_index = train_grp_loss.index(max_train_grp_loss)
            max_val_grp_loss_index = val_grp_loss.index(max_val_grp_loss)

            logging_str = (
                f"Iteration: {step}, train_loss: {train_loss:.4f}, "
                f"val_loss: {val_loss:.4f}, grad_norm: {grad_norm: .4f}, live_grad: {live_grad:.4f}, "
                f"reward_err: {', '.join([f'{r:.4f}' for r in rew_err])}, "
                f"KL_dist: {', '.join([f'{k:.4f}' for k in kl_dist])}, "
                f"param: {param}, "
                f"weights: {group_weights}, "
                f"train_wt_loss: {train_wt_loss: .4f}, "
                f"val_wt_loss: {val_wt_loss:.4f}, "
                f"train_grp_loss: {train_grp_loss}, "
                f"val_grp_loss: {val_grp_loss}, "
                f"max_reward_err: {max_reward_err:.4f}, max_reward_err_index: {max_reward_err_index}, "
                f"max_kl_dist: {max_kl_dist:.4f}, max_kl_dist_index: {max_kl_dist_index}, "
                f"max_train_grp_loss: {max_train_grp_loss:.4f}, max_train_grp_loss_index: {max_train_grp_loss_index}, "
                f"max_val_grp_loss: {max_val_grp_loss:.4f}, max_val_grp_loss_index: {max_val_grp_loss_index}"
            )

            if self.wandb_use:
                wandb_dict = {
                    "Iteration": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "max_reward_err": max_reward_err,
                    "max_reward_err_index": max_reward_err_index,
                    "max_kl_dist": max_kl_dist,
                    "max_kl_dist_index": max_kl_dist_index,
                    "max_train_grp_loss": max_train_grp_loss,
                    "max_train_grp_loss_index": max_train_grp_loss_index,
                    "max_val_grp_loss": max_val_grp_loss,
                    "max_val_grp_loss_index": max_val_grp_loss_index,
                    "grad_norm": grad_norm,
                }

                for i, err in enumerate(rew_err):
                    wandb_dict[f"reward_err_{i+1}"] = err

                for i, kl in enumerate(kl_dist):
                    wandb_dict[f"KL_distance_{i+1}"] = kl

                for i, param in enumerate(self.param.tolist()):
                    wandb_dict[f"param_{i+1}"] = param

                for i, grp_wt in enumerate(self.group_weights.detach().tolist()):
                    wandb_dict[f"group_weight_{i + 1}"] = grp_wt

                for i, loss in enumerate(train_grp_loss):
                    wandb_dict[f"train_group_loss_{i+1}"] = loss

                for i, loss in enumerate(val_grp_loss):
                    wandb_dict[f"val_group_loss_{i+1}"] = loss

                # Log individual gradient components
                for i, g in enumerate(grad.tolist()):
                    wandb_dict[f"grad_{i+1}"] = g

                # Log RTG for each group
                rtg_sum = t.sum(self.RTG, dim=1)
                for grp_i in range(self.group_num):
                    for grp_j in range(self.group_num):
                        wandb_dict[f"rtg_{grp_i+1}_{grp_j+1}"] = self.RTG[grp_i][grp_j].item()
                    wandb_dict[f"relative_transfer_gain_{grp_i+1}"] = rtg_sum[grp_i].item()

                wandb.log(wandb_dict)

            if self.logger:
                self.logger.info(logging_str)
            else:
                print(logging_str)

    def batch_update_once(self, dataset: List[GroupTransition], batch_size: int) -> float:
        def sample_group_transition(group_id):
            group_transitions_with_id = [transition for transition in dataset if transition.group_id == group_id]
            return random.choice(group_transitions_with_id)

        if batch_size < len(dataset):
            sampled_group_transitions = random.choices(dataset, k=batch_size)
        else:
            sampled_group_transitions = dataset

        self.optimizer.zero_grad()

        group_losses = self.evaluate_grp_loss(dataset)
        #group_losses.requires_grad = True
        print("group_losses:", group_losses)
        all_grads = [
            t.autograd.grad(group_losses[li], self.param, retain_graph=True, allow_unused=True)[0] for li in range(self.group_num)
        ]
        print("all_grads:", all_grads)
        # Relative Transfer Gain Matrix
        RTG = t.zeros((self.group_num, self.group_num))
        for li in range(self.group_num):
            for lj in range(self.group_num):
                cos_sim = (all_grads[lj] @ all_grads[li]) / t.clamp(
                    (t.norm(all_grads[lj]) * t.norm(all_grads[li])), min=1e-3
                )
                RTG[li][lj] = cos_sim

        # Gradient scaling - Appendix A
        _gl = t.sqrt(group_losses.detach().unsqueeze(-1))
        RTG = t.mm(_gl, _gl.t()) * RTG
        self.RTG = RTG

        _exp = self.step_size * (RTG @ self.initial_group_weights)
        _exp -= _exp.max()  # To avoid overflow

        # Equation 4
        self.alpha.data = t.exp(_exp)
        self.group_weights *= self.alpha.data
        self.group_weights = self.group_weights / self.group_weights.sum()
        self.group_weights = t.clamp(self.group_weights, min=1e-5)

        #insert WeightedRegression here

        # compute objective
        if self.use_closed_form:
            self.WeightedRegression(sampled_group_transitions, self.reg_coef)
        else:
            objective = self.evaluate_grp_loss(sampled_group_transitions)
            loss = t.sum(objective @ self.group_weights)
            loss.backward()

            self.optimizer.step()

    def evaluate_reward(self, env: GroupLinearBandit, states: Union[list, None]) -> float:
        policy = self.ret_policy()
        rew = env.evaluate_reward_group_wise(policy, states)

        return rew

    def evaluate_KL(self, env: GroupLinearBandit, states: Union[list, None]) -> float:
        policy = self.ret_policy()
        kl_dist = env.evaluate_KL_group_wise(policy, states)

        return kl_dist

    @property
    def get_param(self) -> np.ndarray:
        return self.param

    def WeightedRegression(self, dataset: List[GroupTransition], lamba: float) -> float:
        """Compute closed-form solution for the optimization problem"""
        print("WEIGHTED REGRESSION\n\n")
        Y = []
        w = []
        for transition in dataset:
            state, action_one, action_two, group_id, pref = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.group_id,
                transition.pref,
            )
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one
            feat_pref_act, feat_non_pref_act = (
                self.feature_func(state, pref_act, group_id),
                self.feature_func(state, non_pref_act, group_id),
            )
            Y.append(feat_pref_act - feat_non_pref_act)
            #print(f"self.group_counts within the weighted regression: {self.group_counts}")
            #print(f"self.group_weights within the weighted regression: {self.group_weights}")
            w.append((1 - self.chi) / self.group_num + self.chi * self.group_weights[group_id]/self.group_counts[group_id])

        Y = np.array(Y)
        w = np.array(w)
        #print("w vector within the weighted regression: ", w)
        # print(Y.shape,np.diag(w).shape,(Y@self.param).T.shape,((Y@self.param).T-1/(2*self.reg_coef)).dot(Y).shape)
        coef = np.linalg.inv(Y.transpose() @ np.diag(w) @ Y + lamba * np.eye(Y.shape[1]))
        # print(np.linalg.det(np.matmul(Y.transpose(),Y)))
        variate = np.matmul(np.matmul(Y.transpose(), np.diag(w)), np.ones([len(dataset), 1]))
        self.param = np.matmul(coef, variate).ravel() / (2 * self.reg_coef)
        live_grad = (np.diag(w).dot((Y @ self.param).T - 1 / (2 * self.reg_coef))).dot(Y) + lamba * self.param
        print("live_grad within the weighted regression: ", live_grad)
        print(np.sqrt(np.sum(np.square(live_grad))))
        return np.sqrt(np.sum(np.square(live_grad)))
