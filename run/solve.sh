# DATASET=../../dataset/LEC/all_case_recoveraig/
DATASET=../../dataset/LEC/all_case/

cd ./src

python3 solve.py \
 --exp_id solve \
 --Problem_AIG_Dir ${DATASET} \
 --customized_mapper ./mockturtle/build/examples/my_mapper \
 --baseline_mapper ./mockturtle/build/examples/my_baseline \
 --max_solve_time 1000 \
 --resume