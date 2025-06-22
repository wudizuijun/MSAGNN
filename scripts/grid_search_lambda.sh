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
l1_list=(0 0.001 0.01 0.1 1)
l2_list=(0 0.001 0.01 0.1 1)

##### FCC
# for l1 in ${l1_list[@]}; do
#     for l2 in ${l2_list[@]}; do
#         echo ${l1} ${l2}
#         timeStamp=$(date "+%Y%m%d%H%M%S")
#         timeRec[${#timeRec[@]}]=${timeStamp} # add timestamp to timeRec
#         python main.py \
#             --model_type ${testModel} \
#             --dataset 'FCC' \
#             --target '柴油' \
#             --data_path "./data/all_data_caiyou.csv" \
#             --rmcols "PHD_StartDate,油浆,柴油,汽油,液化气,干气,焦炭,mode" \
#             --epoches 80 \
#             --time ${timeStamp} \
#             --lambda1 ${l1} \
#             --lambda2 ${l2}
#     done
# done

# using python script to ploe
# strTimeRec=$(IFS=,; echo "${timeRec[*]}")
# echo ${strTimeRec}
# python ./plot/PlotRes.py --res_paths ${strTimeRec}


### TE
for l1 in ${l1_list[@]}; do
    for l2 in ${l2_list[@]}; do
        echo ${l1} ${l2}
        timeStamp=$(date "+%Y%m%d%H%M%S")
        timeRec[${#timeRec[@]}]=${timeStamp} # add timestamp to timeRec
        python main.py \
            --model_type ${testModel} \
            --dataset 'TE' \
            --target 'v10' \
            --data_path "./data/te_data.csv" \
            --rmcols "None" \
            --epoches 50 \
            --graph_type "mechanism" \
            --time ${timeStamp} \
            --lambda1 ${l1} \
            --lambda2 ${l2}
    done
done

# using python script to ploe
strTimeRec=$(IFS=,; echo "${timeRec[*]}")
echo ${strTimeRec}
python ./plot/PlotRes.py --res_paths ${strTimeRec}


### DC  
for l1 in ${l1_list[@]}; do
    for l2 in ${l2_list[@]}; do
        echo ${l1} ${l2}
        timeStamp=$(date "+%Y%m%d%H%M%S")
        timeRec[${#timeRec[@]}]=${timeStamp} # add timestamp to timeRec
        python main.py \
            --model_type ${testModel} \
            --dataset 'DC' \
            --target 'X08' \
            --data_path "./data/DC_data.csv" \
            --rmcols "None" \
            --graph_type "mechanism" \
            --epoches 80 \
            --time ${timeStamp} \
            --lambda1 ${l1} \
            --lambda2 ${l2}
    done
done

# using python script to ploe
strTimeRec=$(IFS=,; echo "${timeRec[*]}")
echo ${strTimeRec}
python ./plot/PlotRes.py --res_paths ${strTimeRec}