#!/bin/bash
# script description: 所有模型多次预测的结果
# ./scripts/all_model_predict_once.sh
eval "$(conda shell.bash hook)"
conda activate mlearning

workDir=`pwd`
curPath=${workDir}/scripts

shellFile=('fcc.sh' 'te.sh', 'dc.sh')
# modelList=('GRU' 'GAT_v2' 'dcrnn' 'stgcn' 'gtnet' 'agcrn')
modelList=('GAT_v2')
# modelList=('AttGRU')

runingTimes=10
# gat nfeat

for testModel in ${modelList[@]}; do
    echo ${testModel}
    # run a model for many times
    for i in $(seq 1 $runingTimes); do
        # generate random seed
        seed=`expr $RANDOM % 200 + 1`
        # echo ${seed}
        # record time stamp
        timeStamp=$(date "+%Y%m%d%H%M%S")
        timeRec[${#timeRec[@]}]=${timeStamp} # add timestamp to timeRec
        sh ${curPath}/${shellFile[2]} ${timeStamp} ${testModel} ${seed}
    done
done

# using python script to plot
strTimeRec=$(IFS=,; echo "${timeRec[*]}")
echo ${strTimeRec}
python ./plot/PlotRes.py --res_paths ${strTimeRec}

