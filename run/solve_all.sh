# DATASET=../../dataset/LEC/all_case_recoveraig/
# DATASET=../../dataset/LEC/all_case/
DATASET=../../dataset/SAT_Comp/

cd ./src

python3 solve_all.py \
 --exp_id solve_all \
 --Problem_AIG_Dir ${DATASET} \
 --customized_mapper ./mockturtle/build/examples/my_lutmap4sat \
 --baseline_mapper ./mockturtle/build/examples/my_baseline \
 --max_solve_time 1000 \
 --resume