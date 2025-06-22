#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate mlearning

MODEL_NAME=('lstm' 'gru')
WINDOW_SIZE=(5)
LSTM_LAYERS=(1 2 3 4)

time=$(date "+%Y%m%d%H%M%S")
# echo "${time}"
for model in ${MODEL_NAME[@]}; do
    python main.py \
        --model ${model} \
        --time ${time}
done






















































#     for window_size in ${WINDOW_SIZE[@]}; do
#         # LSTM
#         if [ ${model} == 'lstm' ]; then
#             for lstm_layer in ${LSTM_LAYERS[@]}; do
#                 for enc_hidden_size in ${ENC_HIDDEN_SIZE[@]}; do 
#                     python main.py \
#                         --model ${model} \
#                         # --window_size ${window_size} \
#                         --lstm_layers ${lstm_layer} \
#                         --enc_hidden_size ${enc_hidden_size}
#                 done
#             done
#         fi

#         # SA-LSTM
#         if [ ${model} == 'sa_lstm' ]; then
#             for enc_layer_num in ${ENC_LAYER_NUM[@]}; do
#                 for enc_hidden_size in ${ENC_HIDDEN_SIZE[@]}; do
#                     python main.py \
#                         --model ${model} \
#                         # --window_size ${window_size} \
#                         --enc_layer_num ${enc_layer_num} \
#                         --enc_hidden_size ${enc_hidden_size}
#                 done
#             done
#         fi

#         # TA-LSTM
#         if [ ${model} == 'ta_lstm' ]; then  
#             for dec_layer_num in ${DEC_LAYER_NUM[@]}; do
#                 for dec_hidden_size in ${ENC_HIDDEN_SIZE[@]}; do
#                     python main.py \
#                         --model ${model} \
#                         # --window_size ${window_size} \
#                         --dec_layer_num ${dec_layer_num} \
#                         --dec_hidden_size ${dec_hidden_size}
#                 done
#             done
#         fi

#         # SPA-LSTM
#         if [ ${model} == 'spa_lstm' ]; then  
#             for enc_layer_num in ${ENC_LAYER_NUM[@]}; do
#                 for dec_layer_num in ${DEC_LAYER_NUM[@]}; do
#                     for enc_hidden_size in ${ENC_HIDDEN_SIZE[@]}; do
#                         for dec_hidden_size in ${DEC_HIDDEN_SIZE[@]}; do
#                             python main.py \
#                                 --model ${model} \
#                                 # --window_size ${window_size} \
#                                 --enc_layer_num ${enc_layer_num} \
#                                 --enc_hidden_size ${enc_hidden_size} \
#                                 --dec_layer_num ${dec_layer_num} \
#                                 --dec_hidden_size ${dec_hidden_size}
#                         done
#                     done
#                 done
#             done
#         fi
#     done
# done
    
