# DATASET=../../dataset/LEC/all_case_recoveraig/
# DATASET=../../dataset/SamSAT
DATASET=../../dataset/SamSAT

cd ./src

python3 main.py \
 --exp_id train_1112 \
 --Problem_AIG_Dir ${DATASET} \
 --RL_mode train \
 --customized_mapper ./mockturtle/build/examples/my_mapper \
 --baseline_mapper ./mockturtle/build/examples/my_baseline \
 --min_solve_time 0 --max_solve_time 100 \
 --resume