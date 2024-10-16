#!/bin/bash

set -e
set -x

# Default values
ACTION_NUM=8
GROUP_NUM=3
PREF_DATA_NUM=300
BATCH_SIZE=300
DPO_NUM_ITERS=100
STATE_DIM=2
WEIGHTED_BATCHES='false'
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
USE_WEIGHT_VAL='False'
USE_UNEVEN_GRP='False'
USE_UNEVEN_GRP_VAL='False'
USE_THEORY='False'
WEIGHTS="[1,1,0.4]"
WANDB_ENTITY='group-robustness-noisy-labels'
WANDB_GROUP='group1'
WANDB_PROJECT='hp-search'
CHI=1
FEATURE_TYPE='flipped'

# Create log directory with timestamp
LOG_DIR="log-hyperparam-search/$(date +'%Y_%m_%d_%H_%M_%S')"
mkdir -p "$LOG_DIR"

# Seeds to use
SEEDS=(2021 2022 2023)

# Hyperparameters to search
STEP_SIZES=(0.1 0.2)
REG_COEFS=(0.1)
EXP_STEP_SIZES=(0.01 0.1 1) # Doesn't matter for DPO

# DPO types
DPO_TYPES=('rdpo') # ('dpo' 'rdpo')


for DPO_TYPE in "${DPO_TYPES[@]}"; do
    for STEP_SIZE in "${STEP_SIZES[@]}"; do
        for REG_COEF in "${REG_COEFS[@]}"; do
            for EXP_STEP_SIZE in "${EXP_STEP_SIZES[@]}"; do
		for SEED in "${SEEDS[@]}"; do
                    DETERMINISTIC_RATIO_LIST="[1,0.8,1]"
                    VAL_DETERMINISTIC_RATIO_LIST="[1,1,1]"
                   
		    SUB_LOG_DIR="${LOG_DIR}/${DPO_TYPE},${STEP_SIZE},${REG_COEF},${EXP_STEP_SIZE},${SEED}"
		    mkdir -p $SUB_LOG_DIR
		    mkdir -p "$SUB_LOG_DIR/log"  
	     		    
                    python -m experiments.run_glb_noisy \
                    --mle_adaptive \
                    --state_dim ${STATE_DIM} \
                    --action_num ${ACTION_NUM} \
                    --group_num ${GROUP_NUM} \
                    --pref_data_num ${PREF_DATA_NUM} \
                    --dpo_num_iters ${DPO_NUM_ITERS} \
                    --wandb_use \
                    --wandb_group ${WANDB_GROUP} \
                    --reg_coef ${REG_COEF} \
                    --seed ${SEED} \
                    --weights ${WEIGHTS} \
                    --log_dir ${SUB_LOG_DIR} \
                    --dpo_type ${DPO_TYPE} \
                    --dpo_step_size ${STEP_SIZE} \
                    --rdpo_exp_step_size ${EXP_STEP_SIZE} \
                    --rdpo_batch_size ${BATCH_SIZE} \
                    --feature_type ${FEATURE_TYPE} \
                    --rdpo_weighted_batches ${WEIGHTED_BATCHES} \
                    --exp_adaptive ${EXP_ADAPTIVE} \
                    --rdpo_adj ${RDPO_ADJ} \
                    --eval_metric ${EVAL_METRIC} \
                    --importance_sampling ${IMPORTANCE_SAMPLING} \
                    --importance_sampling_weights ${IMPORTANCE_SAMPLING_WEIGHTS} \
                    --ipo_grad_type ${IPO_GRAD_TYPE} \
                    --param_limit ${PARAM_LIMIT} \
                    --use_closed_form ${USE_CLOSED_FORM} \
                    --val_deterministic 'True' \
                    --deterministic_ratio_list ${DETERMINISTIC_RATIO_LIST} \
                    --val_deterministic_ratio_list ${VAL_DETERMINISTIC_RATIO_LIST} \
                    --lamba ${LAMBA} \
                    --l2_reg_rdpo ${L2_REG_RDPO} \
                    --use_weight_val ${USE_WEIGHT_VAL} \
                    --use_uneven_grp ${USE_UNEVEN_GRP} \
                    --use_uneven_grp_val ${USE_UNEVEN_GRP_VAL} \
                    --use_theory ${USE_THEORY} \
		    --wandb_project ${WANDB_PROJECT} \
		    --wandb_logdir "${SUB_LOG_DIR}/log" \
		    --wandb_name "${DPO_TYPE},${STEP_SIZE},${REG_COEF},${EXP_STEP_SIZE}" \
                    --chi ${CHI}
                done
            done
        done
    done
done
