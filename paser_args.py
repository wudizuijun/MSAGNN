# used in shell 
import argparse


''' FCC debug '''
def get_args():
    parser = argparse.ArgumentParser(description='Training parameters! ')
    ###########################################
    #############         Common parameters:
    ###########################################

    parser.add_argument('--results_path', type=str, required=False, default='./results/')
    parser.add_argument('--time', type=str, required=False, default='None', help='shell used to distinguish different shell results') # contain target 
    # # # TE
    parser.add_argument('--data_path', type=str, required=False, default='./data/te_data.csv', help='data path') 
    parser.add_argument('--dataset', type=str, required=False, default='TE',  help='dataset name')
    parser.add_argument('--target', type=str, required=False, default='v10', help='target name') 
    parser.add_argument('--rmcols', type=str, required=False, default='None', help='useless columns need to drop') # contain target 
    
    # # DC
    # parser.add_argument('--data_path', type=str, required=False, default='./data/DC_data.csv', help='data path') 
    # parser.add_argument('--dataset', type=str, required=False, default='DC',  help='dataset name')
    # parser.add_argument('--target', type=str, required=False, default='X08', help='target name') 
    # parser.add_argument('--rmcols', type=str, required=False, default='None', help='useless columns need to drop') # contain target 

    parser.add_argument('--rolling_window', type=int, required=False, default=1)
    parser.add_argument('--scaler', type=str, required=False, default='MinMaxScaler', choices=['MinMaxScaler', 'StandardScaler', 'None'],help='scaler name')
    parser.add_argument('--train_ratio', type=float, required=False, default=0.6, help='train ratio')
    parser.add_argument('--valid_ratio', type=float, required=False, default=0.1, help='valid ratio')
    parser.add_argument('--output_dim', type=int, required=False, default=1) 
    parser.add_argument('--in_dim', type=int, default=1, help='input data dimension')
    
    # parser.add_argument('--time', type=str, required=False, default='day', help='runing time')
    
    ###########################################
    #############         Training paramters:
    ###########################################
    
    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--optimizer', type=str, required=False, default='Adam')
    parser.add_argument('--epoches', type=int, required=False, default=1)
    parser.add_argument('--lr', type=float, required=False, default=0.0001)
    parser.add_argument('--WEIGHT_DECAY', type=float, required=False, default=0.001)
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    
    
    ###########################################
    #############         Model parameters:
    ###########################################
    parser.add_argument('--model_type', type=str, required=False, default='AttGRU', help='specify model name') 
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--window_size', type=int, required=False, default=16)
    parser.add_argument('--horizon', type=int, required=False, default=1) # 预测未来长度
    parser.add_argument("--autoregress", type=bool, required=False, default=False)
    parser.add_argument('--graph_type', type=str, required=False, default='mechanism', choices=['mechanism', 'corr', 'allone'])
    parser.add_argument('--graph_learning_method', type=str, required=False, default='new', choices=['new', 'old']) 
    parser.add_argument('--use_dagga', type=bool, required=False, default=True, help='use dagga or not')

    ###########################################
    #############         Custom model parameters:
    ###########################################
    
    # GSL
    parser.add_argument('--lambda1', type=float, required=False, default=0.0)
    parser.add_argument('--lambda2', type=float, required=False, default=0.0)
    
    # PaperII model(AttGRU)
    parser.add_argument('--attgru_hidsize', type=int, required=False, default=64)
    parser.add_argument('--attgru_hidslayer', type=int, required=False, default=32)
    # GGRU
    parser.add_argument('--gru_hid', type=int, required=False, default=16)
    
    # new graph_learn_method argument
    parser.add_argument('--ng_dim', type=int, default=32, help='embedding dimension')
    parser.add_argument('--ng_cout', type=int, default=16) # useless, ignore it
    parser.add_argument('--ng_heads', type=int, default=8)
    parser.add_argument('--ng_head_dim', type=int, default=32)
    parser.add_argument('--ng_eta', type=float, default=1) # useless, ignore it
    parser.add_argument('--ng_gamma', type=float, default=0.001, help='weight of f_norm')
    parser.add_argument('--ng_dropout', type=float, default=0.5)
    parser.add_argument('--ng_m', type=float, default=0.9, help='momentum update')
    parser.add_argument('--ng_is_add1', type=bool, default=False, help='whether to add 1 to time_step')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    a = get_args()
    # print(vars(a))
    # print(a.dec_layer_num)

