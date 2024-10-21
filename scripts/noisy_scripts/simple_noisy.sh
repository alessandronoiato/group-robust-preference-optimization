#!/bin/bash

set -e
set -x

MAX_PARALLEL=8

run_experiment() {
    eval $1
}

SCRIPT_DIR="scripts/noisy_scripts"
CONFIG_DIR="scripts/noisy_scripts/configs"

# Generate commands for each method
python ${SCRIPT_DIR}/main_runner.py --base_config ${CONFIG_DIR}/simple_noisy_config.yaml --method_configs ${CONFIG_DIR}/dpo_config.yaml ${CONFIG_DIR}/rdpo_config.yaml ${CONFIG_DIR}/rdpo_is_config.yaml ${CONFIG_DIR}/cgd_config.yaml ${CONFIG_DIR}/rdpo_noisy_config.yaml

# Function to run a single experiment
run_experiment() {
    eval $1
}

# Run experiments in parallel
PARALLEL_JOBS=0
while IFS= read -r cmd
do
    run_experiment "$cmd" &
    PARALLEL_JOBS=$((PARALLEL_JOBS + 1))
    if [ $PARALLEL_JOBS -ge $MAX_PARALLEL ]; then
        wait -n
        PARALLEL_JOBS=$((PARALLEL_JOBS - 1))
    fi
done < commands.txt

# Wait for any remaining background jobs to finish
wait
