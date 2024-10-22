import collections
import math
from typing import List

import matplotlib.pyplot as plt
import neatplot
import numpy as np

neatplot.set_style()
Transition = collections.namedtuple("Transition", ["state", "action_0", "action_1", "reward_0", "reward_1", "pref"])

GroupTransition = collections.namedtuple(
    "GroupTransition", ["state", "action_0", "action_1", "group_id", "reward_0", "reward_1", "pref"]
)


def sigmoid(x: float):
    return 1.0 / (1.0 + math.exp(-x))


def ret_uniform_policy(action_num: int = 0):
    assert action_num > 0, "The number of actions should be positive."

    def uniform_policy(state: np.ndarray = None):
        action_prob = np.full(shape=action_num, fill_value=1.0 / action_num)
        return action_prob

    return uniform_policy


def ret_uniform_policy_group(action_num: int = 0):
    assert action_num > 0, "The number of actions should be positive."

    def uniform_policy_group(state: np.ndarray = None, group_id: int = None):
        action_prob = np.full(shape=action_num, fill_value=1.0 / action_num)
        return action_prob

    return uniform_policy_group


def collect_preference_data(num: int, env, policy_func) -> List[Transition]:
    pref_dataset = []
    action_num = env.action_num
    for _ in range(num):
        state = env.reset()
        action_prob = policy_func(state)
        sampled_actions = np.random.choice(a=action_num, size=2, replace=False, p=action_prob)  # replace=True
        action_one, action_two = sampled_actions[0], sampled_actions[1]
        reward_one, reward_two = env.sample(action_one), env.sample(action_two)
        # print(state, reward_one, reward_two, reward_two - reward_one)

        bernoulli_param = sigmoid(reward_two - reward_one)
        # pref=1 means that the second action is preferred over the first one
        pref = np.random.binomial(1, bernoulli_param, 1)[0]
        transition = Transition(state, action_one, action_two, reward_one, reward_two, pref)
        pref_dataset.append(transition)
    return pref_dataset


def collect_group_preference_data(
    num: int, env, weights: List[float], policy_func, deterministic: bool = False
) -> List[GroupTransition]:
    pref_dataset = []
    action_num = env.action_num
    group_num = env.group_num

    group_counts = np.ceil(np.array(weights) * num).astype(int)
    group_ids = [i for i, count in enumerate(group_counts) for _ in range(count)]
    np.random.shuffle(group_ids)
    group_ids = group_ids[:num]

    for i in range(num):
        state = env.reset()
        group_id = group_ids[i]

        action_prob = policy_func(state, group_id)
        sampled_actions = np.random.choice(a=action_num, size=2, replace=False, p=action_prob)  # replace=True
        action_one, action_two = sampled_actions[0], sampled_actions[1]
        reward_one, reward_two = env.sample(action_one, group_id), env.sample(action_two, group_id)

        if deterministic == True:
            pref = 0 if reward_one > reward_two else 1
        else:
            bernoulli_param = sigmoid(reward_two - reward_one)
            pref = np.random.binomial(1, bernoulli_param, 1)[0]
        # pref=1 means that the second action is preferred over the first one

        # pref= 0 if reward_one>reward_two else 1
        group_transition = GroupTransition(state, action_one, action_two, group_id, reward_one, reward_two, pref)
        pref_dataset.append(group_transition)

    print(f"WEIGHTS: {weights}, COUNTS: {np.bincount(group_ids)}")
    return pref_dataset


def collect_group_preference_data_plot(
    num: int, env, weights: List[float], policy_func, feature_type: str, deterministic: bool = False
) -> List[GroupTransition]:
    pref_dataset = []
    action_num = env.action_num
    group_num = env.group_num
    print(weights)
    group_id_1 = 0
    group_id_2 = 0
    group_counts = np.round(np.array(weights) * num).astype(int)
    group_ids = [i for i, count in enumerate(group_counts) for _ in range(count)]
    np.random.shuffle(group_ids)
    group_ids = group_ids[:num]
    probs = []
    crt_data = 0
    for i in range(num):
        state = env.reset()
        # group_id= int(np.random.choice(np.arange(group_num),1,p=np.array(weights)))
        group_id = group_ids[i]
        # print(group_id)
        if group_id == 0:
            group_id_1 += 1
        else:
            group_id_2 += 1
        action_prob = policy_func(state, group_id)
        sampled_actions = np.random.choice(a=action_num, size=2, replace=False, p=action_prob)  # replace=True
        action_one, action_two = sampled_actions[0], sampled_actions[1]
        reward_one, reward_two = env.sample(action_one, group_id), env.sample(action_two, group_id)
        # print(state, reward_one, reward_two, reward_two - reward_one)
        if deterministic == True:
            pref = 0 if reward_one > reward_two else 1
        else:
            bernoulli_param = sigmoid(reward_two - reward_one)
            probs.append(bernoulli_param)
            pref = np.random.binomial(1, bernoulli_param, 1)[0]
            if reward_two - reward_one > 0 and pref == 1:
                crt_data += 1
            elif reward_two - reward_one < 0 and pref == 0:
                crt_data += 1
        # pref=1 means that the second action is preferred over the first one

        # pref= 0 if reward_one>reward_two else 1
        group_transition = GroupTransition(state, action_one, action_two, group_id, reward_one, reward_two, pref)
        pref_dataset.append(group_transition)
    probs = np.array(probs)
    group_ids = np.array(group_ids)
    plt.figure(figsize=(12, 6))
    # Calculate histogram bins
    bins = np.arange(0, 1.2, 0.2)
    bins1 = np.arange(1, 2.2, 0.2)
    group1_indices = [i for i in range(num) if group_ids[i] == 1]
    group0_indices = [i for i in range(num) if group_ids[i] == 0]
    # plt.plot(group_ids[group0_indices], probs[group0_indices], label='prob_distribution_group_0')
    # plt.plot(group_ids[group1_indices], probs[group1_indices], label='prob_distribution_group_1')
    # Plot histograms for each group separately
    plt.hist(
        probs[group_ids == 0],
        bins=bins,
        label="Group 0",
        alpha=0.5,
        align="mid",
        histtype="barstacked",
        edgecolor="black",
    )
    plt.hist(
        probs[group_ids == 1],
        bins=bins,
        label="Group 1",
        alpha=0.5,
        align="mid",
        histtype="barstacked",
        edgecolor="black",
    )

    # Calculate positions for each group's histogram
    positions_group0 = np.arange(0, len(bins) - 1) * 2
    positions_group1 = np.arange(0, len(bins) - 1) * 2 + 1
    # Plot histograms for each group separately
    # plt.hist(probs[group_ids == 0], bins=bins+, label='Group 0', alpha=0.5, align='mid', histtype='barstacked', edgecolor='black', weights=np.ones(len(probs[group_ids == 0])) / len(probs[group_ids == 0]), color='blue', position=positions_group0)
    # plt.hist(probs[group_ids == 1], bins=bins, label='Group 1', alpha=0.5, align='mid', histtype='barstacked', edgecolor='black', weights=np.ones(len(probs[group_ids == 1])) / len(probs[group_ids == 1]), color='orange', position=positions_group1)

    plt.title(f"Prob_distributions_{feature_type}")
    plt.xlabel("Groups")
    plt.ylabel("Prob Values")
    plt.legend()
    neatplot.save_figure(f"Prob_distributions_{num}_{feature_type}")
    plt.close()
    complement_crt_data = num - crt_data

    # Plotting
    plt.figure(figsize=(8, 6))

    # Plotting crt_data
    plt.bar("crt_data", crt_data, color="blue", label="crt_labelled_data")

    # Plotting num - crt_data
    plt.bar("wrong_data", complement_crt_data, color="orange", label="mis_labelled_data")

    plt.title("Histogram of crt_data and wrong_data_{feature_type}")
    plt.ylabel("Value")
    plt.legend()
    neatplot.save_figure(f"crt_labels_{num}_{feature_type}")
    plt.close()
    print(group_id_1, group_id_2)
    return pref_dataset


def collect_group_preference_data_partial_deterministic(
    num: int, env, weights: List[float], policy_func, deterministic_ratio: int = 0
) -> List[GroupTransition]:
    pref_dataset = []
    action_num = env.action_num
    group_num = env.group_num
    print(weights)
    group_id_1 = 0
    group_id_2 = 0
    group_counts = np.round(np.array(weights) * num).astype(int)
    group_ids = [i for i, count in enumerate(group_counts) for _ in range(count)]
    np.random.shuffle(group_ids)
    group_ids = group_ids[:num]
    for i in range(num):
        state = env.reset()
        # group_id= int(np.random.choice(np.arange(group_num),1,p=np.array(weights)))
        group_id = group_ids[i]
        # print(group_id)
        if group_id == 0:
            group_id_1 += 1
        else:
            group_id_2 += 1
        action_prob = policy_func(state, group_id)
        sampled_actions = np.random.choice(a=action_num, size=2, replace=False, p=action_prob)  # replace=True
        action_one, action_two = sampled_actions[0], sampled_actions[1]
        reward_one, reward_two = env.sample(action_one, group_id), env.sample(action_two, group_id)
        # print(state, reward_one, reward_two, reward_two - reward_one)
        epsilon = np.random.uniform(0, 1)
        if deterministic_ratio >= epsilon:
            pref = 0 if reward_one > reward_two else 1
        else:
            bernoulli_param = sigmoid(reward_two - reward_one)
            pref = np.random.binomial(1, bernoulli_param, 1)[0]
        # pref=1 means that the second action is preferred over the first one

        # pref= 0 if reward_one>reward_two else 1
        group_transition = GroupTransition(state, action_one, action_two, group_id, reward_one, reward_two, pref)
        pref_dataset.append(group_transition)
    print(group_id_1, group_id_2)
    return pref_dataset


def collect_group_preference_data_partial_deterministic_list(
    num: int, env, weights: List[float], policy_func, deterministic_ratio_list: List[float]
) -> List[GroupTransition]:
    pref_dataset = []
    action_num = env.action_num
    group_num = env.group_num
    group_counts = np.ceil(np.array(weights) * num).astype(int)
    group_ids = [i for i, count in enumerate(group_counts) for _ in range(count)]
    np.random.shuffle(group_ids)
    group_ids = group_ids[:num]
    crt_count = np.zeros(group_num)

    for i in range(num):
        state = env.reset()
        group_id = group_ids[i]
        action_prob = policy_func(state, group_id)
        sampled_actions = np.random.choice(a=action_num, size=2, replace=False, p=action_prob)  # replace=True
        action_one, action_two = sampled_actions[0], sampled_actions[1]
        reward_one, reward_two = env.sample(action_one, group_id), env.sample(action_two, group_id)
        # print(state, reward_one, reward_two, reward_two - reward_one)
        epsilon = np.random.uniform(0, 1)
        if deterministic_ratio_list[group_id] >= epsilon:
            pref = 0 if reward_one > reward_two else 1
            crt_count[group_id] += 1
        else:
            bernoulli_param = sigmoid(reward_two - reward_one)
            pref = np.random.binomial(1, bernoulli_param, 1)[0]
            if reward_two - reward_one >= 0 and pref == 1:
                crt_count[group_id] += 1
            elif reward_two - reward_one <= 0 and pref == 0:
                crt_count[group_id] += 1
        # pref=1 means that the second action is preferred over the first one

        # pref= 0 if reward_one>reward_two else 1
        group_transition = GroupTransition(state, action_one, action_two, group_id, reward_one, reward_two, pref)
        pref_dataset.append(group_transition)
    print(f"WEIGHTS: {weights}, COUNTS: {np.bincount(group_ids)}")
    print(crt_count / np.bincount(group_ids) * 100, "crt_data")
    return pref_dataset


def collect_uneven_group_preference_data_partial_deterministic_list(
    num: int, env, weights: List[float], policy_func, deterministic_ratio_list: List[float]
) -> List[GroupTransition]:
    pref_dataset = []
    action_num = env.action_num
    group_num = env.group_num
    group_id_1 = 0
    group_id_2 = 0

    group_counts = np.ceil(np.array(weights) * num).astype(int)
    group_ids = [i for i, count in enumerate(group_counts) for _ in range(count)]
    np.random.shuffle(group_ids)
    group_ids = group_ids[:num]
    crt_count = np.zeros(group_num)
    for i in range(num):
        state = env.reset()
        group_id = group_ids[i]

        if group_id == 0:
            group_id_1 += 1
        else:
            group_id_2 += 1
        action_prob = policy_func(state, group_id)
        sampled_action_1 = np.random.choice(a=action_num, size=1, replace=False, p=action_prob)  # replace=True
        if group_id % 2 == 0:
            sampled_action_2 = (
                sampled_action_1 + 2 * np.random.choice(a=2, size=1, replace=False, p=[0.5, 0.5]) - 1  # replace=True
            )
            # print(sampled_action_1,sampled_action_2)
            if sampled_action_2 >= action_num:
                sampled_action_2 = sampled_action_1 - 1
            elif sampled_action_2 < 0:
                sampled_action_2 = sampled_action_1 + 1
        else:
            sampled_action_2 = sampled_action_1 + int((action_num / 2))
            if sampled_action_2 >= action_num:
                sampled_action_2 = sampled_action_2 % action_num

        sampled_actions = [sampled_action_1, sampled_action_2]
        action_one, action_two = sampled_actions[0], sampled_actions[1]
        reward_one, reward_two = env.sample(action_one, group_id), env.sample(action_two, group_id)
        # print(state, reward_one, reward_two, reward_two - reward_one)
        epsilon = np.random.uniform(0, 1)
        if deterministic_ratio_list[group_id] >= epsilon:
            pref = 0 if reward_one > reward_two else 1
            crt_count[group_id] += 1
        else:
            bernoulli_param = sigmoid(reward_two - reward_one)
            pref = np.random.binomial(1, bernoulli_param, 1)[0]
            if reward_two - reward_one >= 0 and pref == 1:
                crt_count[group_id] += 1
            elif reward_two - reward_one <= 0 and pref == 0:
                crt_count[group_id] += 1
        # pref=1 means that the second action is preferred over the first one

        # pref= 0 if reward_one>reward_two else 1
        group_transition = GroupTransition(state, action_one, action_two, group_id, reward_one, reward_two, pref)
        pref_dataset.append(group_transition)
    # print(group_id_1, group_id_2)
    unique, counts = np.unique(group_ids, return_counts=True)
    print(f"WEIGHTS: {weights} COUNTS: {counts}")
    print(crt_count / group_counts * 100, "crt_data")
    return pref_dataset


def collect_group_preference_data_wth_deterministic_list(
    num: int, env, weights: List[float], policy_func, deterministic_list: List[bool]
) -> List[GroupTransition]:
    pref_dataset = []
    action_num = env.action_num
    group_num = env.group_num
    print(weights)
    group_id_1 = 0
    group_id_2 = 0
    group_counts = np.round(np.array(weights) * num).astype(int)
    group_ids = [i for i, count in enumerate(group_counts) for _ in range(count)]
    np.random.shuffle(group_ids)
    group_ids = group_ids[:num]
    for i in range(num):
        state = env.reset()
        # group_id= int(np.random.choice(np.arange(group_num),1,p=np.array(weights)))
        group_id = group_ids[i]
        # print(group_id)
        # if group_id==0:
        #    group_id_1+=1
        # else:
        #    group_id_2+=1
        action_prob = policy_func(state, group_id)
        sampled_actions = np.random.choice(a=action_num, size=2, replace=False, p=action_prob)  # replace=True
        action_one, action_two = sampled_actions[0], sampled_actions[1]
        reward_one, reward_two = env.sample(action_one, group_id), env.sample(action_two, group_id)
        # print(state, reward_one, reward_two, reward_two - reward_one)
        epsilon = np.random.uniform(0, 1)
        # print(group_id)
        # print(deterministic_list)
        if deterministic_list[group_id]:
            pref = 0 if reward_one > reward_two else 1
            group_id_1 += 1
        else:
            bernoulli_param = sigmoid(reward_two - reward_one)
            pref = np.random.binomial(1, bernoulli_param, 1)[0]
            group_id_2 += 1
        # pref=1 means that the second action is preferred over the first one

        # pref= 0 if reward_one>reward_two else 1
        group_transition = GroupTransition(state, action_one, action_two, group_id, reward_one, reward_two, pref)
        pref_dataset.append(group_transition)
    print(group_id_1, group_id_2, "part_deterministic")
    return pref_dataset


def double_group_preference_data(env, pref_data: List[GroupTransition]) -> List[GroupTransition]:
    double_pref_data = []
    group_num = env.group_num
    for transition in pref_data:
        state, action_one, action_two, group_id, pref = (
            transition.state,
            transition.action_0,
            transition.action_1,
            transition.group_id,
            transition.pref,
        )
        for i in range(group_num):
            if i == group_id:
                continue
            reward_one, reward_two = env.sample(action_one, i), env.sample(action_two, i)

    return double_pref_data


def collect_group_preference_data_debug(num: int, env, weights: List[float], policy_func) -> List[GroupTransition]:
    pref_dataset = []
    action_num = env.action_num
    group_num = env.group_num
    print(weights)
    group_id_1 = 0
    group_id_2 = 0
    group_counts = np.round(np.array(weights) * num).astype(int)
    group_ids = [i for i, count in enumerate(group_counts) for _ in range(count)]
    np.random.shuffle(group_ids)
    print(group_ids, "groups")
    group_ids = group_ids[:num]
    for i in range(num):
        state = env.reset()
        # group_id= int(np.random.choice(np.arange(group_num),1,p=np.array(weights)))
        group_id = group_ids[i]
        # print(group_id)
        if group_id == 0:
            group_id_1 += 1
        else:
            group_id_2 += 1
        action_prob = policy_func(state, group_id)
        sampled_actions = np.random.choice(a=action_num, size=2, replace=False, p=action_prob)  # replace=True
        action_one, action_two = sampled_actions[0], sampled_actions[1]
        reward_one, reward_two = env.sample(action_one, group_id), env.sample(action_two, group_id)
        # print(state, reward_one, reward_two, reward_two - reward_one)

        bernoulli_param = sigmoid(reward_two - reward_one)
        # pref=1 means that the second action is preferred over the first one
        pref = np.random.binomial(1, bernoulli_param, 1)[0]
        # pref= 0 if reward_one>reward_two else 1
        group_transition = GroupTransition(state, action_one, action_two, group_id, reward_one, reward_two, pref)
        pref_dataset.append(group_transition)
    print(group_id_1, group_id_2)
    return pref_dataset


def process_pref_data(pref_data: list[Transition]) -> list:
    states = list()
    pos_action = list()
    neg_action = list()

    for i, t in enumerate(pref_data):
        state, a0, a1, r0, r1, pref = t

        states.append(state)

        if r0 > r1:
            pos_action.append(a0)
            neg_action.append(a1)

        else:
            pos_action.append(a1)
            neg_action.append(a0)

    # Process into numpy arrays:
    states = np.array(states)
    pos_action = np.array(pos_action)
    neg_action = np.array(neg_action)

    return states, pos_action, neg_action


def process_pref_grp_data(pref_data: list[GroupTransition]) -> list:
    states = list()
    pos_action = list()
    neg_action = list()
    groups = list()

    for i, t in enumerate(pref_data):
        state, a0, a1, g, r0, r1, pref = t

        states.append(state)
        groups.append(g)

        if r0 > r1:
            pos_action.append(a0)
            neg_action.append(a1)

        else:
            pos_action.append(a1)
            neg_action.append(a0)

    # Process into numpy arrays:
    states = np.array(states)
    groups = np.array(groups)
    pos_action = np.array(pos_action)
    neg_action = np.array(neg_action)

    return states, pos_action, neg_action, groups


def collect_rl_data(num: int, env) -> List[float]:
    rl_dataset = []
    for _ in range(num):
        state = env.reset()
        rl_dataset.append(state)

    return rl_dataset


def merge_datasets(pref_dataset: List[Transition], rl_dataset: List[float]):
    merged_rl_dataset = rl_dataset
    for transition in pref_dataset:
        state = transition.state
        merged_rl_dataset.append(state)

    return merged_rl_dataset


def pref_to_rl(pref_dataset: List[Transition]):
    rl_dataset = []
    for transition in pref_dataset:
        state = transition.state
        rl_dataset.append(state)

    return rl_dataset
