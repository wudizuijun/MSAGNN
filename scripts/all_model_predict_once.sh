#!/bin/bash
# script description: 用于多个模型对一个数据集进行预测的结果
# ./scripts/all_model_predict_once.sh
eval "$(conda shell.bash hook)"
conda activate mlearning

workDir=`pwd`
curPath=${workDir}/scripts

shellFile=('fcc.sh' 'te.sh' 'dc.sh')
# GAT_V2 all ones 准确度还行
# modelList=('GRU' 'GAT_v2' 'dcrnn' 'stgcn' 'gtnet' 'agcrn')
modelList=('GAT_v2' 'dcrnn' 'gtnet')

for model in ${modelList[@]}; do
    echo ${model}
    timeStamp=$(date "+%Y%m%d%H%M%S")
    timeRec[${#timeRec[@]}]=${timeStamp} # add timestamp to timeRec
    sh ${curPath}/${shellFile[2]} ${timeStamp} ${model}
done

strTimeRec=$(IFS=,; echo "${timeRec[*]}")
# python ./plot/PlotRes.py --res_paths ${strTimeRec}

# A=(AA BB CC)
# S=$(IFS=,; echo "${A[*]}")