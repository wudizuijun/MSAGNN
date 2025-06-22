# calculate adjacency matrix
import pickle, os
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

class AdjMatCal:
    def __init__(self, **kwargs):
        ''' 用于获取不同数据集的邻接矩阵(global)
        '''
        self.dataset = kwargs['dataset']
        self.data_path = kwargs['data_path']
        self.args = kwargs
        
    
    def cal_by_mechanism(self):
        ''' 通过实际的变量空间结果获取邻接矩阵
        return type: torch.Tensor
        '''
        if self.dataset == 'LA':
            _, _, adj = self.load_graph_data('./data/METR-LA/adj_mx.pkl') # 存在自环, 而且非离散邻接矩阵
            # 离散化
            adj[adj > 0] = 1
        elif self.dataset == 'TE':
            adj = pd.read_excel('./data/te_adj_pri.xlsx', index_col=0, sheet_name='Sheet1').values
            adj = np.nan_to_num(adj)
    
        # 去自环
        adj = torch.from_numpy(adj).float()
        adj = adj - torch.diag(torch.diag(adj))
        return adj
    
    
    def cal_by_correlation(self):
        ''' 通过变量之间的相关性计算邻接矩阵(目前在数据格式上， 这几个交通数据集还不支持)
        '''
        # prior adjacency matrix when the graph structure is known
        # Description about element in adj matrix:
        # A_{i,j} means there is a connection point to node j from node i, 默认无自环
        print(os.getcwd())
        df_temp = pd.read_csv(self.data_path).iloc[:, :]
        target = self.args['target']
        if self.args['rmcols'] != 'None':
            drop_columns = [col for col in self.args['rmcols'].split(',')]
        # print(drop_columns)
            drop_columns.remove(target)
            df_temp = df_temp.drop(columns=drop_columns)
        
        train_idx = int(df_temp.shape[0] * self.args['train_ratio'])
        df_train = df_temp.iloc[:train_idx, :]
        adj = self._cal_prior_graph(df_train, thresh=0.3)
        del df_temp, df_train
        
        # 去自环，将对角线元素置为0
        adj = torch.from_numpy(adj).float()
        adj = adj - torch.diag(torch.diag(adj))
        return adj
    
    def _cal_prior_graph(self,data:pd.DataFrame, thresh=0.6):
        """ 按照变量相关性计算先验拓扑图
        """
        scalar = StandardScaler()
        data = scalar.fit_transform(data)
        
        corr = np.abs(np.corrcoef(data.T))
        graph = np.where(corr>thresh, 1, 0)
        # print(graph)
        return graph
    
    ''' 交通数据集 (below)'''
    def load_graph_data(self, pkl_filename):
        sensor_ids, sensor_id_to_ind, adj_mx = self.load_pickle(pkl_filename)
        return sensor_ids, sensor_id_to_ind, adj_mx

    def load_pickle(self, pickle_file):
        try:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f)
        except UnicodeDecodeError as e:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print('Unable to load data ', pickle_file, ':', e)
            raise
        return pickle_data
    
if __name__ == "__main__":
    # adj_cal = AdjMatCal(dataset='TE')
    # adj = adj_cal.cal_by_mechanism()
    
    ajd_cal = AdjMatCal(
        dataset='FCC',
        data_path='./../data/save_df.csv',
        target='汽油',
        train_ratio=0.6,
        rmcols='油浆,柴油,汽油,液化气,干气,焦炭'
    )
    adj = ajd_cal.cal_by_correlation()
    
    print(adj)