### 工具类
import paser_args
import os, random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import r2_score as R2
from datetime import datetime
from utils.cal_adj_mat import AdjMatCal
from plot.Plot import Plot

DPI = 600

class Utils:
    def __init__(self):
        pass
    
    @staticmethod
    def get_args():
        return paser_args.get_args()
    
    @staticmethod
    def get_global_adj(**args):
        adj_calculator = AdjMatCal(**args)
        if args['graph_type'] == 'mechanism':
            adj = adj_calculator.cal_by_mechanism()
        elif args['graph_type'] == 'corr':
            adj = adj_calculator.cal_by_correlation()
        elif args['graph_type'] == 'allone':
            adj = torch.ones(33,33)
        adj = adj.to(args['device'])
        return adj
    
    @staticmethod   
    def seed_torch(seed=42):
        ''' 固定种子
        '''
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        
    @staticmethod
    def get_device():
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @staticmethod
    def flattenBatch(batch):
        ''' 将batch展平
        batch type: [tensors] ,shape: (Batch_num * bathch_size)
        return: flatten_data(np.array), shape: (Batch_num * bathch_size )
        '''
        flatten_data = np.array([])
        for data in batch:
            flatten_data = np.append(flatten_data, data)
        return flatten_data
        
    @staticmethod
    def get_adj(path='./data/adj_pri.xlsx'):
        adj = pd.read_excel(path, index_col=0, sheet_name='Sheet1').values
        # fill nan with 0
        adj = np.nan_to_num(adj)    
        return adj
    
    @staticmethod
    def get_normalized_adj(A):
        """
        Input: A(troch.tensor): Adjacency matrix(不一定是对称的)
        output: A_wave(torch.tensor)
        Returns the degree normalized adjacency matrix.
        谱图卷积这类需要对称邻接矩阵的模型需要使用
        """
        # convert to numpy
        A = A.cpu().numpy()
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
        D = np.array(np.sum(A, axis=1)).reshape((-1,))
        D[D <= 10e-5] = 10e-5    # Prevent infs
        diag = np.reciprocal(np.sqrt(D))
        A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                            diag.reshape((1, -1)))
        device = Utils.get_device()
        A_wave = torch.from_numpy(A_wave).to(device)
        return A_wave


    @staticmethod
    def cal_prior_graph(data:pd.DataFrame, thresh=0.6):
        """ 按照变量相关性计算先验图
        """
        scalar = StandardScaler()
        data = scalar.fit_transform(data)
        
        corr = np.abs(np.corrcoef(data.T))
        graph = np.where(corr>thresh, 1, 0)
        # print(graph)
        return graph
        
    @staticmethod
    def train(model, optimizer, train_data, epoches, loss_fn, device, valid_data,adj=None):
        ''' Trainning model
        Attention: 传入的model为可变变量(python引用传递)
        
        return: total training loss
        '''
        pbar = tqdm(total=epoches, ncols=100, position=0)
        total_loss = []
        total_valid_loss = []
        for epoch in range(1, epoches+1):
            ave_loss = Utils.train_per_epoch(model, optimizer, train_data, epoches, loss_fn, device, adj=adj)
            total_loss.append(ave_loss)
            valid_loss, _, _ = Utils.val_test(model, valid_data, loss_fn, device, adj)
            total_valid_loss.append(valid_loss)
            pbar.set_description('Training. Epoch: %d, train_loss %.3f, validation_loss: %.3f' % (epoch + 1, ave_loss, valid_loss))
            pbar.update()
        pbar.close()
        return total_loss, total_valid_loss
    
    @staticmethod
    def train_per_epoch(model, optimizer, train_data, epoches, loss_fn, device, adj):
        ''' Trainning in one epoch
        Attention: 传入的model为可变变量(python引用传递)
        return: training loss
        '''
        loss_list = [] # record each batch loss in one epoch
        for _, (X, X_d, y) in enumerate(train_data): # in LA dataset, X_d -> temporal feature
            # x: (B * seq * N)
            # X_d: (B * N * N * seq)
            X, y = X.to(device).permute(0,2,1), y.to(device)#[:, 120]
            X_d = X_d.to(device)
            # do prediction and backpropagation
            pred = model(X, adj, X_d) # input_shape: x(B*seq*N)
            loss = loss_fn(pred, y) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            
            del X, y, pred, X_d
            torch.cuda.empty_cache()
        ave_loss = np.mean(loss_list)
        return ave_loss      
    
    @staticmethod
    def val_test(model, data, loss_fn, device, adj):
        ''' validation or test
        Attention: 传入的model为可变变量(python引用传递)
        return: total training loss
        '''
        total_pred = torch.Tensor().to(device)
        total_y = torch.Tensor().to(device)
        with torch.no_grad():
            for _, (X, X_d, y) in enumerate(data):
                X, y = X.to(device).permute(0,2,1), y.to(device)#[:, 120]
                X_d = X_d.to(device)
                pred = model(X, adj, X_d)
                total_pred = torch.cat((total_pred, pred), dim=0)
                total_y = torch.cat((total_y, y), dim=0)
                
                del X, X_d
                torch.cuda.empty_cache()
        ave_loss = loss_fn(total_pred, total_y).item()
        return ave_loss, total_y.squeeze(), total_pred.squeeze()
        # return loss.item(), y, pred
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式 
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)    
        else:
            pass
            
            
    @staticmethod
    def save_results(train_loss, valid_loss, pred, real, args, now=None, 
                     train_pred=None, train_real=None,
                     Ap_att=None, Ad=None):
        ''' save results
        Ap_att: 节点相关性图
        Ad: 学到的动态邻接矩阵
        '''

        # create folder
        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)
        res_root_folder = os.path.join(args.results_path, now)
        os.makedirs(res_root_folder, exist_ok=True)
        
        # temp last is nan
        real = real[:-1]
        pred = pred[:-1]

        # save pred & real data
        res = pd.DataFrame(np.array([real, pred]).T,
                            columns=['real', 'prediction'],
                            dtype=np.float32)
        res.to_csv(os.path.join(res_root_folder, 'prediction.csv'))
        
        # myplot = Plot(os.path.join(res_root_folder, 'prediction.csv'))
        myplot = Plot(pred=pred, real=real)
        if train_pred is not None:
            my_plot1 = Plot(pred=train_pred, real=train_real)
            fig = my_plot1.pred_line_plot()
            fig.savefig(os.path.join(res_root_folder, 'train_pred_line.png'), dpi=DPI)
            
        # save graph
        if Ap_att is not None:
            AP_att = np.array(Ap_att)
            Ad = np.array(Ad)
            np.save(os.path.join(res_root_folder, 'Ap_att.npy'), AP_att)
            np.save(os.path.join(res_root_folder, 'Ad.npy'), Ad)
        
        # save loss curve (后续可以考虑加上validataion loss)
        loss = pd.DataFrame(np.array([train_loss, valid_loss]).T, 
                            columns=['train_loss', 'valid_loss'],
                            dtype=np.float32)
        loss.to_csv(os.path.join(res_root_folder, 'loss.csv'))
        fig = myplot.plot_loss(train_loss, valid_loss)
        fig.savefig(os.path.join(res_root_folder, 'loss.png'), dpi=DPI)
        
        fig = myplot.pred_line_plot()
        fig.savefig(os.path.join(res_root_folder, 'pred_line.png'), dpi=DPI)
        
        fig = myplot.box_plot()
        fig.savefig(os.path.join(res_root_folder, 'box_plot.png'), dpi=DPI)
        
        fig = myplot.r2_plot(kde=True)
        fig.savefig(os.path.join(res_root_folder, 'r2_plot.png'), dpi=DPI)
        
        # calculate metrics
        mse = MSE(real, pred)
        rmse = MSE(real, pred, squared=False)
        mae = MAE(real, pred)
        mape = MAPE(real, pred)
        r2 = R2(real, pred)
        
        # record the results and params to txt
        model_name = args.model_type
        res_root_folder = os.path.join(res_root_folder, model_name)
        os.makedirs(res_root_folder)
        file_name = os.path.join(res_root_folder, 'record.txt')
        with open(file_name, 'a', encoding="utf-8") as f:
            f.writelines(['Target: ', args.target, '\n'])
            f.writelines(['Rmse: ', str(rmse), '\n'])
            f.writelines(['mse: ', str(mse), '\n'])
            f.writelines(['mae: ', str(mae), '\n'])
            f.writelines(['mape: ', str(mape), '\n'])   
            f.writelines(['R2: ', str(r2), '\n'])
            
            f.write('params: \n')
            f.writelines(str(args))
            f.write('\n')
            f.close()
