# DATASET=../../dataset/LEC/all_case_recoveraig/
DATASET=../../dataset/LEC/all_case/

cd ./src

python3 main.py \
 --exp_id train \
 --Problem_AIG_Dir ${DATASET} \
 --customized_mapper ./mockturtle/build/examples/my_lutmap4sat \
 --resume