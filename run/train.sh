# DATASET=../../dataset/LEC/all_case_recoveraig/
DATASET=../../dataset/SamSAT

cd ./src

python3 main.py \
 --exp_id train_new \
 --Problem_AIG_Dir ${DATASET} \
 --customized_mapper ./mockturtle/build/examples/my_mapper \
 --min_solve_time 0 --max_solve_time 100 \
 --resume
