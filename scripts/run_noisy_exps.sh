#!/bin/bash

set -e
set -x

STATE_DIM=2
ACTION_NUM=8
GROUP_NUM=3
PREF_DATA_NUM=300
BATCH_SIZE=300
DPO_NUM_ITERS=2

# Default values
VAL_DETERMINISTIC='True'
STEP_SIZE=0.1 
REG_COEF=1 
DPO_TYPE='rdpo'
EXP_STEP_SIZE=1
WEIGHTED_BATCHES='false'
C=0
EXP_ADAPTIVE=0
RDPO_ADJ='0'
EVAL_METRIC='argmax'
IMPORTANCE_SAMPLING='False'
IMPORTANCE_SAMPLING_WEIGHTS='None'
IPO_GRAD_TYPE='linear'
PARAM_LIMIT=5
USE_CLOSED_FORM='True'
LAMBA=0
L2_REG_RDPO=0
USE_UNEVEN_GRP='False' # 'False'
USE_UNEVEN_GRP_VAL='False'
USE_THEORY='False'
WEIGHTS="[1,1,0.4]" # "[1,1,1]"
VAL_WEIGHTS="[1,1,1]" # "[1,1,1]"
TEST_WEIGHTS="[1,1,1]" # "[1,1,1]"
WANDB_ENTITY="group-robustness-noisy-labels"
WANDB_PROJECT="hp-search"
WANDB_GROUP='hp-search'
CHI=1

# Timestamp for meta log directory
TIMESTAMP=$(date +'%Y_%m_%d_%H_%M_%S')

# Seeds
SEEDS=(2021) #(2021 2022 2023 2024 2025 2026 2027 2028 2029 2030) 

# Noise levels (deterministic_ratio_list)
NOISE_LEVELS=(0.9) #("1.0" "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3")

# Feature types
FEATURE_TYPES=("same") # ("same" "flipped" "swapped")

# DPO types
DPO_TYPES=("rdpo")

# EPSILONS
EPSILONS=("0.0")

# REWARD_PARAM
REWARD_PARAM='[[2.0, 2.0, 2.0, 2.0],[2.0, 2.0, 2.0, 2.0],[2.0, 2.0, 2.0, 2.0]]'

# Main loop
for DPO_TYPE in "${DPO_TYPES[@]}"; do
    # Create meta log directory for each DPO type
    META_LOG_DIR="log_${DPO_TYPE}/${TIMESTAMP}"
    mkdir -p "$META_LOG_DIR"

    for FEATURE_TYPE in "${FEATURE_TYPES[@]}"; do
        for NOISE_LEVEL in "${NOISE_LEVELS[@]}"; do
            for EPSILON in "${EPSILONS[@]}"; do
                DETERMINISTIC_RATIO_LIST="[1,$NOISE_LEVEL,1]" # "[1,1,1]" # Change when group_num changes
                VAL_DETERMINISTIC_RATIO_LIST="[1,1,1]" # "[1,1,1]" # Change when group_num changes
                        
                for SEED in "${SEEDS[@]}"; do
                        # Create subdirectory for each variation
                        SUB_LOG_DIR="${META_LOG_DIR}/${FEATURE_TYPE}_noise${NOISE_LEVEL}_${DETERMINISTIC_RATIO_LIST}_${SEED}_${EPSILON}"
                        mkdir -p "$SUB_LOG_DIR"
                        mkdir -p "$SUB_LOG_DIR/log"
                        
                        python -m experiments.run_glb_noisy \
                        --mle_adaptive \
                        --state_dim ${STATE_DIM} \
                        --action_num ${ACTION_NUM} \
                        --group_num ${GROUP_NUM} \
                        --pref_data_num ${PREF_DATA_NUM} \
                        --dpo_num_iters ${DPO_NUM_ITERS} \
                        --wandb_use \
                        --C ${C} \
                        --wandb_group "${WANDB_GROUP}" \
                        --reg_coef ${REG_COEF} \
                        --seed ${SEED} \
                        --epsilon ${EPSILON} \
                        --weights ${WEIGHTS}  \
                        --val_weights ${VAL_WEIGHTS} \
                        --test_weights ${TEST_WEIGHTS} \
                        --log_dir ${SUB_LOG_DIR} \
                        --dpo_type ${DPO_TYPE} \
                        --dpo_step_size ${STEP_SIZE} \
                        --rdpo_exp_step_size ${EXP_STEP_SIZE} \
                        --rdpo_batch_size ${BATCH_SIZE} \
                        --feature_type ${FEATURE_TYPE} \
                        --rdpo_weighted_batches ${WEIGHTED_BATCHES} \
                        --exp_adaptive ${EXP_ADAPTIVE} \
                        --rdpo_adj ${RDPO_ADJ} \
                        --reward_param "${REWARD_PARAM}" \
                        --eval_metric ${EVAL_METRIC} \
                        --importance_sampling ${IMPORTANCE_SAMPLING} \
                        --importance_sampling_weights ${IMPORTANCE_SAMPLING_WEIGHTS} \
                        --ipo_grad_type ${IPO_GRAD_TYPE} \
                        --param_limit ${PARAM_LIMIT} \
                        --use_closed_form ${USE_CLOSED_FORM} \
                        --val_deterministic ${VAL_DETERMINISTIC} \
                        --deterministic_ratio_list ${DETERMINISTIC_RATIO_LIST} \
                        --val_deterministic_ratio_list ${VAL_DETERMINISTIC_RATIO_LIST} \
                        --lamba ${LAMBA} \
                        --l2_reg_rdpo ${L2_REG_RDPO} \
                        --use_uneven_grp ${USE_UNEVEN_GRP} \
                        --use_uneven_grp_val ${USE_UNEVEN_GRP_VAL} \
                        --use_theory ${USE_THEORY} \
                        --wandb_entity ${WANDB_ENTITY} \
                        --wandb_project ${WANDB_PROJECT} \
                        --wandb_logdir "${SUB_LOG_DIR}/log" \
                        --wandb_name "${DPO_TYPE},${STEP_SIZE},${REG_COEF},${EXP_STEP_SIZE},${DETERMINISTIC_RATIO_LIST},${SEED}" \
                        --chi ${CHI}
                done
            done
        done
    done
done
