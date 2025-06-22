#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate mlearning

# echo $1
# echo "hello world"
# python ./../main.py \
#     --dataset 'LA'

# MODEL_NAME=('lstm' 'gru')
# WINDOW_SIZE=(5)
# LSTM_LAYERS=(1 2 3 4)

# time=$(date "+%Y%m%d%H%M%S")
# # echo "${time}"
# for model in ${MODEL_NAME[@]}; do
#     python main.py
#         # --model ${model} \
#         # --time ${time}
# done

# a=1
# if [ ${a} == 1 ]; then
#     echo "hello world"
# elif [ ${a} == 2 ]; then
#     echo "hello world2"
# else
#     echo "hello world3"
# fi

# for i in $(seq $startNum $endNum)
# do
#     echo current is $i
# done

runingTimes=10
for i in $(seq 1 $runingTimes); do
    seed=`expr $RANDOM % 200 + 1`
    echo ${seed}
done

# function genRand(){
#     max=$1
#     seed=`expr $RANDOM % ${max} + 1`
#     echo ${seed}
#     # return
#     # return ${seed}
# }
# echo $?

# seed=${genRand}
# echo ${seed}
# # exec /bin/bash