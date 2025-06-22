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
if [ $3 == "none" ]; then
    seed=42
else
    seed=$3
fi
# seed=42

python main.py \
    --model_type ${model} \
    --dataset 'DC' \
    --target 'X08' \
    --rmcols "None" \
    --data_path "./data/DC_data.csv" \
    --graph_type "mechanism" \
    --epoches 80 \
    --time $1 \
    --seed ${seed} \

# exec /bin/bash
