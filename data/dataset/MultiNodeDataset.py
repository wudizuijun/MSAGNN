import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data.dataset.data_process import ScalarStandardizer, MinMaxScaler
import torch
import numpy as np
import os

'''
record:
用于图神经网络多节点预测的数据集
后续可进一步考虑添加batch_normalization
'''
    
class TDDataset(Dataset):
    """ 带有时滞的数据集
        用于paperII 中的方法
    """
    def __init__(self, data, delay_data, targets) -> None:
        self.data = data
        self.delay_data = delay_data
        self.targets = targets
    
    def __getitem__(self, index):
        ''' raise error stop iteration
        '''
        # return {
        #     'data': self.data[index],
        #     'delay_data': self.delay_data[index],
        #     'target': self.targets[index]
        # }
        return self.data[index], self.delay_data[index], self.targets[index]
    
    def __len__(self):
        return len(self.data)-1
    
class MultiNodeDataset(Dataset):
    def __init__(self, adj, **args) -> None:
        ''' 带有时滞的数据集
        data format: timestamp | data...
        data: data_path, reuqiured data columns [timestamp, x, y]
        window_size: seq_len
        '''
        self._feature_num = None
        self.batch_size = args['batch_size']
        self.window_size = args['window_size']
        self.adj = adj
        
        
        # load the data
        file_type = args['data_path'].split('.')[-1]
        if file_type == 'csv':
            # data = pd.read_csv(data_path, encoding='gbk').iloc[:, :]
            # data = pd.read_csv(r'G:\研究生\毕业论文\chapter_III\FrameWork_version_1.0_multiNode_twoConvTunnel\data\all_data_caiyou.csv')
            data = pd.read_csv(args['data_path'])#.iloc[:, 1:]  # start from 1 index to remove timestamp
        elif file_type == 'xlsx':
            data = pd.read_excel(args['data_path'])#.iloc[:, 1:]
        else:
            raise ValueError('file type not support!')
        print('origin data dim: ', data.shape)

        # drop useless columns
        self.targets_name = [t for t in args['target'].split(',')]
        
        if args['rmcols'] != 'None':
            # rm_cols = [col for col in args['rmcols'].split(',') if col != args['target']]
            rm_cols = [col for col in args['rmcols'].split(',') if col not in args['target']]
            data = data.drop(columns=rm_cols)
        # self.target_idx = data.columns.get_loc(args['target']) # get the target index
        
        self.target = data.iloc[1:, :]
        data = data.iloc[:-1, :]
        # right shift 1 to get the target data
        # self.target = data.iloc[:, self.target_idx][1:] # target is y_t
        # data.iloc[:, self.target_idx] = data.iloc[:, self.target_idx].shift(1) # y in graph is y_{t-1}
        data = data.dropna()
        
        self.targets_idx = [self.target.columns.get_loc(t) for t in args['target'].split(',')]
        print(self.targets_idx, self.targets_name)
        
        assert args['train_ratio'] + args['valid_ratio'] < 1, 'train_ratio + valid_ratio must less than 1!'
        
        data = data.values # graph value
        train_idx, valid_idx = int(len(data)*args['train_ratio']), int(len(data)*(args['train_ratio']+args['valid_ratio']))
        # 由于考虑动态时滞，防止train过程在 __getitem__中计算导致计算时间增加， 这里手动计算
        self.train_data, self.valid_data, self.test_data = data[:train_idx], data[train_idx-self.window_size+1:valid_idx], data[valid_idx-self.window_size+1:]
        self.train_y, self.valid_y, self.test_y = self.target[:train_idx], self.target[train_idx-self.window_size+1:valid_idx], self.target[valid_idx-self.window_size+1:]
        
        # ori_train_target = self.train_data[:, self.target_idx]
        # ori_valid_target = self.valid_data[:, self.target_idx]
        # ori_test_target = self.test_data[:, self.target_idx]
        
        # scaler
        self.scaler = None
        self.target_scaler = None
        if args['scaler'] != 'None':
            self.scaler = MinMaxScaler() if args['scaler'] == 'MinMaxScaler' else ScalarStandardizer()
            self.target_scaler = MinMaxScaler() if args['scaler'] == 'MinMaxScaler' else ScalarStandardizer()
            self.train_data = self.scaler.fit_transform(self.train_data[:,:]) # target 和 data 分离了
            self.valid_data = self.scaler.transform(self.valid_data[:,:])
            self.test_data = self.scaler.transform(self.test_data[:,:])
            
            self.target_scaler.fit(self.train_y)
            # self.train_y = self.target_scaler.fit(self.train_y)
            # self.valid_y = self.target_scaler.transform(self.valid_y)
            # self.test_y = self.target_scaler.transform(self.test_y)
            
            # recover his target， 自回归上最修正
            # self.train_data[:, self.target_idx] = ori_train_target
            # self.valid_data[:, self.target_idx] = ori_valid_target
            # self.test_data[:, self.target_idx] = ori_test_target            
            
        
        self.train_targets = self.train_y[self.window_size:].values
        self.val_targets = self.valid_y[self.window_size:].values
        self.test_targets = self.test_y[self.window_size:].values
        
        self.train_data, self.X_D_train = self.get_model_data(self.train_data)
        self.valid_data, self.X_D_val = self.get_model_data(self.valid_data)
        self.test_data, self.X_D_test = self.get_model_data(self.test_data)

        print('input data dim: ', self.train_data.shape)
        # to tensor
        self.train_data = torch.tensor(self.train_data, dtype=torch.float32)
        self.valid_data = torch.tensor(self.valid_data, dtype=torch.float32)
        self.test_data = torch.tensor(self.test_data, dtype=torch.float32)
        self.X_D_train = torch.tensor(self.X_D_train, dtype=torch.float32)
        self.X_D_val = torch.tensor(self.X_D_val, dtype=torch.float32)
        self.X_D_test = torch.tensor(self.X_D_test, dtype=torch.float32)
        self.train_targets = torch.tensor(self.train_targets, dtype=torch.float32)
        self.val_targets = torch.tensor(self.val_targets, dtype=torch.float32)
        self.test_targets = torch.tensor(self.test_targets, dtype=torch.float32)

    def get_multi_idx(self):
        ''' 获取各个target 对应的idx'''
        return self.targets_name, self.targets_idx
    
    # @time_rec
    def get_model_data(self, data):
        ''' 计算__getitem__中一次返回的数据
        data: (whole time length, N)
        '''
        datasets = torch.zeros((data.shape[0]-self.window_size+1, data.shape[1], self.window_size)) # （B*）
        # datasets = np.zeros((data.shape[0]-self.window_size+1, data.shape[1], self.window_size)) # （B*）
        for i in range(data.shape[0]+1-self.window_size):
            d_per = data[i:i+self.window_size, :] # (window_size, N)
            datasets[i] = d_per.T
        return datasets, datasets
    
    def get_dataLoader(self):
        train_dataloader = DataLoader(TDDataset(self.train_data, self.X_D_train, self.train_targets), 
                                      batch_size=self.batch_size, 
                                      shuffle=False,
                                    #   num_workers=6
                                      )
        # 乐，try过后才懂什么是爆显存:(
        valid_dataloader = DataLoader(TDDataset(self.valid_data, self.X_D_val, self.val_targets), 
                                      batch_size=self.batch_size, 
                                      shuffle=False,
                                    #   num_workers=6
                                      )
        test_dataLoader = DataLoader(TDDataset(self.test_data, self.X_D_test, self.test_targets), 
                                     batch_size=self.batch_size, 
                                     shuffle=False,
                                    #  num_workers=6
                                     )
        return train_dataloader, valid_dataloader, test_dataLoader
        
    def get_scaler(self):
        return self.target_scaler
    
    @property
    def feature_num(self):
        return self._feature_num

    
if __name__ == '__main__':
    import sys
    sys.path.append('./../../')
    from utils.Utils import Utils as utils
    args = utils.get_args()
    d_args = vars(args)
    
    mydata = MultiNodeDataset(adj=torch.rand(10, 10), **d_args)
    train_dl, val_dl, test_dl = mydata.get_dataLoader()
    for i, (data, delay_data, target) in enumerate(train_dl):
        print(data.shape, delay_data.shape, target.shape)
        if i == 0:
            break