#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate mlearning

if [ $2 == "none" ]; then
    model='AttGRU'
else
    model=$2
fi
# model list [GRU, GAT, AttGRU]
# dataset list [FCC, LA, TE]
# data path : all_data (paper I) / save_df.csv (II催中所有数据)
python main.py \
    --model_type ${model} \
    --dataset 'FCC' \
    --target '柴油' \
    --rmcols "PHD_StartDate,油浆,柴油,汽油,液化气,干气,焦炭,mode" \
    --epoches 20 \
    --time $1 \

# exec /bin/bash