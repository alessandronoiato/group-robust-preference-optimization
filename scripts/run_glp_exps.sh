#bin/bash

STATE_DIM=2
ACTION_NUM=8
GROUP_NUM=3
FEATURE_TYPE='same'
SEED=4
DRL='[1,1,1]'
VDRL='[1,1,1]'
VAL_DETERMINISTIC="true"
PREF_DATA_NUM=500
WEIGHTS="[1,1,1]"
VAL_DATA_NUM=50
VAL_WEIGHTS="[1,1,1]"
NUM_TRIALS_FOR_EVAL=1000
DPO_TYPE="rdpo"

python -m experiments.run_glb \
	--state_dim=$STATE_DIM \
	--action_num=$ACTION_NUM \
	--group_num=$GROUP_NUM \
	--feature_type=$FEATURE_TYPE \
	--seed=$SEED \
	--deterministic_ratio_list=$DRL \
	--val_deterministic_ratio_list=$VDRL \
	--weights=$WEIGHTS \
	--val_weights=$VAL_WEIGHTS \
	--dpo_type=$DPO_TYPE
