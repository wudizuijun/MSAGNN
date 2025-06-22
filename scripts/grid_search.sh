#!/bin/bash
# script description: 一个模型在不同参数范围上进行多次预测，绘制一个参数在不同值下的结果
# ./scripts/all_model_predict_once.sh
eval "$(conda shell.bash hook)"
conda activate mlearning

workDir=`pwd`
curPath=${workDir}/scripts

shellFile=('fcc.sh' 'te.sh', 'dc.sh')
# modelList=('GRU' 'GAT_v2' 'dcrnn' 'stgcn' 'gtnet' 'agcrn')
modelList=('AttGRU')
testModel=${modelList[0]}
runingTimes=1

# window size 8 10 12 14 16
paramerList=(8 10 12 14 16)

# batch size 16 24 32 40 48
# paramerList=(16 24 32 40 48)

# temporature 0.1 0.25 0.5 0.75 1
# paramerList=(0.1 0.25 0.5 0.75 1)

# Hidden size 32 64 96 128 256
# paramerList=(32 64 96 128 256)

#####################       Window size       #####################
# echo ${testModel}
# # run model at different parameter
# for param in ${paramerList[@]}; do
#     echo ${param}
#     timeStamp=$(date "+%Y%m%d%H%M%S")
#     timeRec[${#timeRec[@]}]=${timeStamp} # add timestamp to timeRec
#     python main.py \
#         --model_type ${testModel} \
#         --dataset 'DC' \
#         --target 'X08' \
#         --rmcols "None" \
#         --data_path "./data/DC_data.csv" \
#         --graph_type "mechanism" \
#         --epoches 80 \
#         --time ${timeStamp} \
#         --window_size ${param} \
#     # sh ${curPath}/${shellFile[2]} ${timeStamp} ${testModel} ${param}
# done
# # using python script to ploe
# strTimeRec=$(IFS=,; echo "${timeRec[*]}")
# echo ${strTimeRec}
# python ./plot/PlotRes.py --res_paths ${strTimeRec}


#####################      batch size       #####################
# paramerList=(16 24 32 40 48)
# echo ${testModel}
# # run model at different parameter
# for param in ${paramerList[@]}; do
#     echo ${param}
#     timeStamp=$(date "+%Y%m%d%H%M%S")
#     timeRec[${#timeRec[@]}]=${timeStamp} # add timestamp to timeRec
#     python main.py \
#         --model_type ${testModel} \
#         --dataset 'DC' \
#         --target 'X08' \
#         --rmcols "None" \
#         --data_path "./data/DC_data.csv" \
#         --graph_type "mechanism" \
#         --epoches 80 \
#         --time ${timeStamp} \
#         --batch_size ${param} \
#     # sh ${curPath}/${shellFile[2]} ${timeStamp} ${testModel} ${param}
# done
# # using python script to ploe
# strTimeRec=$(IFS=,; echo "${timeRec[*]}")
# echo ${strTimeRec}
# python ./plot/PlotRes.py --res_paths ${strTimeRec}


# #####################     WEIGHT_DECAY       #####################
# paramerList=(0.001 0.005 0.01 0.1 1)
# echo ${testModel}
# # run model at different parameter
# for param in ${paramerList[@]}; do
#     echo ${param}
#     timeStamp=$(date "+%Y%m%d%H%M%S")
#     timeRec[${#timeRec[@]}]=${timeStamp} # add timestamp to timeRec
#     python main.py \
#         --model_type ${testModel} \
#         --dataset 'DC' \
#         --target 'X08' \
#         --rmcols "None" \
#         --data_path "./data/DC_data.csv" \
#         --graph_type "mechanism" \
#         --epoches 80 \
#         --time ${timeStamp} \
#         --WEIGHT_DECAY ${param} \
#     # sh ${curPath}/${shellFile[2]} ${timeStamp} ${testModel} ${param}
# done
# # using python script to ploe
# strTimeRec=$(IFS=,; echo "${timeRec[*]}")
# echo ${strTimeRec}
# python ./plot/PlotRes.py --res_paths ${strTimeRec}

# #####################     Hidde size       #####################
paramerList=(32 64 96 128 256)
echo ${testModel}
# run model at different parameter
for param in ${paramerList[@]}; do
    echo ${param}
    timeStamp=$(date "+%Y%m%d%H%M%S")
    timeRec[${#timeRec[@]}]=${timeStamp} # add timestamp to timeRec
    python main.py \
        --model_type ${testModel} \
        --dataset 'DC' \
        --target 'X08' \
        --rmcols "None" \
        --data_path "./data/DC_data.csv" \
        --graph_type "mechanism" \
        --epoches 80 \
        --time ${timeStamp} \
        --attgru_hidsize ${param} \
    # sh ${curPath}/${shellFile[2]} ${timeStamp} ${testModel} ${param}
done
# using python script to ploe
strTimeRec=$(IFS=,; echo "${timeRec[*]}")
echo ${strTimeRec}
python ./plot/PlotRes.py --res_paths ${strTimeRec}