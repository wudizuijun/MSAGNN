#!/bin/bash
# 不同数据集分开来做
eval "$(conda shell.bash hook)"
conda activate mlearning

# 先预测一个道路节点

python main.py \
    --model 'AttGRU' \
    --dataset 'LA' \ 
    --target 'v10' \
    --rmcols "None" \
    --data_path "./data/te_data.csv" \
    --graph_type "mechanism" \
    --epoches 20 \

exec /bin/bash