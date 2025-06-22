#!/bin/bash
# 不同数据集分开来做
eval "$(conda shell.bash hook)"
conda activate mlearning

# whether have model from parent shell
if [ $2 == "none" ]; then
    model='AttGRU'
else
    model=$2
fi
# where have seed from parent shell
# if [ $3 == "none" ]; then
#     seed=42
# else
#     seed=$3
# fi


python main.py \
    --model_type ${model} \
    --dataset 'TE' \
    --target 'v10' \
    --rmcols "None" \
    --data_path "./data/te_data.csv" \
    --graph_type "mechanism" \
    --epoches 5 \
    --time $1 \
    # --seed ${seed} \

# exec /bin/bash
