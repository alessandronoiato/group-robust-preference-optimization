sh scripts/run_glp_mult_uneven.sh --dpo_type 'rdpo' --feature_type 'swapped' --step_size 5 --reg_coef 2 --eval_metric 'argmax' --ipo_grad_type 'justdpo' --param_limit 5 --exp_step_size 0.01 --dpo_num_iters 5000 --use_closed_form False --deterministic_ratio_list '[1,1]' --weighted_batches False --lamba 0 --val_deterministic_ratio_list '[1,1]' --use_theory False --use_uneven_grp True --use_uneven_grp_val True --weight 0.2 --wandb_group 'uneven_imbal_v2_5k'