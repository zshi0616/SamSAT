# DATASET=../../dataset/LEC/all_case/
DATASET=../../dataset/SAT_Comp/
cd ./src

python3 solve_exp.py \
 --exp_id solve_large_feature \
 --Problem_AIG_Dir ${DATASET} \
 --customized_mapper ./mockturtle/build/examples/my_mapper \
 --baseline_mapper ./mockturtle/build/examples/my_baseline \
 --max_solve_time 1000 \
 --RL_mode test \
 --large_feature --mlp_layers 4 \
 --resume