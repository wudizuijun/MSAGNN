import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import matplotlib.pyplot as plt 
import numpy as np
from models.Modules import gumbel_softmax
# from models.DAGG_backup import DAgg
from models.DAGG import DAgg, AttentionWithContext 
from models.graph_learning import Graph_constructor


# from Modules import gumbel_softmax
# from DAGG_backup import DAgg

class AttenLayer(nn.Module):
    def __init__(self, input_size, embed_size):
        """ 自注意力计算层
        """
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
                
        self.W_q = nn.Linear(input_size, embed_size)
        self.W_k = nn.Linear(input_size, embed_size)
        self.W_v = nn.Linear(input_size, embed_size)
        
    def forward(self, x, adj):
        """_summary_

        :param _type_ x: shape: (batch * seq_len *node)
        :param _type_ adj: shape: (node * node)
        
        return:
        V_prime: shape: (batch * seq_len * embed_size)
        atten: shape: (batch * node * node)
        """
        x = x.permute(0, 2, 1) # (batch, node)
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        V_prime, atten = self.masked_atten(Q, K, V, adj)
        return V_prime, atten
        
    def masked_atten(self, Q, K, V, adj):
        # adj: (node * node) -> 不加自环
        E = torch.matmul(Q, K.permute(0,2,1))
        zero_vec = -9e15*torch.ones_like(E)
        E = torch.where(adj > 0, E, zero_vec)
        atten = F.softmax(E, dim=1)
        V_prime = torch.matmul(atten, V) / sqrt(512)
        return V_prime, atten
    
class AttGRUcell(nn.Module):
    def __init__(self, in_dim, hidden_size) -> None:
        """ GRU cell with attention.
        in_dim: node num
        hidden_size: hidden state of gru
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        
        con_dim = 1
        self.W_ir = nn.Linear(in_dim, hidden_size)
        self.W_hr = nn.Linear(hidden_size+con_dim, hidden_size)
        self.W_iz = nn.Linear(in_dim, hidden_size)
        self.W_hz = nn.Linear(hidden_size+con_dim, hidden_size)
        self.W_in = nn.Linear(in_dim, hidden_size)
        self.W_hn = nn.Linear(hidden_size+con_dim, hidden_size)
        
        self.embed_layer1 = nn.Linear(hidden_size+con_dim, hidden_size) # g conv embedding layer for each node feature
        self.embed_layer2 = nn.Linear(hidden_size+con_dim, hidden_size) # ga conv embedding layer for each node feature
        
        self.delay_conv = nn.Linear(in_dim, 1) # 
        self.softmax = nn.Softmax(dim=0)
        self.weigh = nn.Parameter(torch.FloatTensor(torch.rand(3,2) + 1e-1))
        
    def _gc_aggregate(self, x, A_p, A_d, weight_idx):
        """Graph Convolution aggregate, 对特征进行不同种方式的加权然后再进行加权.
        x: graph data, shape:
        A_P: (B * N * N) prior weighted attention adjacency matrix
        adj: adjacent matrix or weighted matrix. (A_ij represents node j to node i's weight)

        """
        # insert a dimension to x
        # x = x.unsqueeze(-1) # (1, node_num, in_dim)
        v1 = self._gaconv(x, A_p)
        # v2 = self._gconv(x, A_d)
        # w1, w2 = self.softmax(self.weigh[weight_idx, :])
        # return v1
        return v1
        # return w1 * v1 + w2 * v2 
        
    
    def _gaconv(self, x, adj):
        ''' 图注意力卷积
        x: (B * fea * N)
        adj: B*N*N
        '''
        return self.embed_layer2(torch.matmul(x, adj).permute(0,2,1)) 
    
    def _gconv(self, x, adj):
        ''' 谱图卷积(laplacian)
        adj: a_ij represents node i to node j's weight
        '''
        laplacian = self.calculate_laplacian_with_self_loop(adj)
        x = torch.matmul(x, laplacian)
        return self.embed_layer1(x.permute(0,2,1))
    
    def calculate_laplacian_with_self_loop(self, matrix):
        '''
        revised: matrix(N*N) -> matrix(B*N*N)
        '''
        
        matrix = matrix + torch.eye(matrix.size(1)).to('cuda') # (B * N * N)
        row_sum = matrix.sum(2)
        # d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt = torch.pow(row_sum, -0.5).reshape(matrix.shape[0],-1)
        
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0 # (B * N)
        # d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        
        normalized_laplacian = (
            matrix.matmul(d_mat_inv_sqrt).transpose(1, 2).matmul(d_mat_inv_sqrt)
        )
        return normalized_laplacian
    
    
    def forward(self, x, h, adj, adj_d, x_d):
        '''
        h: shape(B, N*hidden_size)
        adj: prior weighted attention adjacency matrix (B*N*N)
        adj_d: dynamic local adjacency matrix
        x_d: delay matrix
        '''
        # temp = self._gc_aggregate(x, adj)
        x = x.unsqueeze(-1) # (B, N, 1)
        x_d = self.delay_conv(x_d) # (B, N, 1) # 对时滞数据的聚合
        # x = torch.cat([x, x_d], dim=-1) # (B, N, 2) （x||delayed x）
        
        h = h.reshape(-1, self.in_dim, self.hidden_size) # (B, N, hidden_size)
        conc_f = torch.concat([x, h],dim=-1) #(B * N *(hiddensize+con_dim))
        conc_f = conc_f.permute(0, 2, 1)
        # conc_f = conc_f.permute(0, 2, 1).reshape(-1, self.in_dim) # (B*(hiddensize+con_dim), N)
        
        
        r = nn.Sigmoid()(self._gc_aggregate(conc_f, adj, adj_d, 0) + self.W_hr(conc_f.permute(0,2,1))) # all hidden_size r和z可以放到一块计算，计算完成后再进行拆分
        z = nn.Sigmoid()(self._gc_aggregate(conc_f, adj, adj_d, 1) + self.W_hz(conc_f.permute(0,2,1)))
        n = nn.Tanh()(self._gc_aggregate(conc_f, adj, adj_d, 2) + r*self.W_hn(conc_f.permute(0,2,1)))
        h_prim = (1-z)*n + z*h
        return h_prim

# model Framework
class AttGRU(nn.Module):
    # def __init__(self,node_num, in_dim, hidden_size=64, num_layer=1, out_dim=1, seq_len=12, temp=0.5) -> None:
    def __init__(self, **kwargs) -> None:
        
        """_summary_

        :param _type_ in_dim: feature num, here is seq_len
        :param _type_ hidden_size: _description_
        :param int num_layer: _description_, defaults to 1
        :param int out_dim: _description_, defaults to 1
        :param float temp: gumbel softmax parameter, temperature, defaults to 0.5
        """
        super().__init__()
        out_dim = 1
        node_num = kwargs['node_num']
        self.hidden_size = kwargs['attgru_hidsize']
        self.num_layer = kwargs['attgru_hidslayer']
        self.node_num = kwargs['node_num']
        self.seq_len = kwargs['window_size']
        self.gl_type = kwargs['graph_learning_method']
        
    
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.gru = nn.GRU(in_dim, hidden_size, num_layer)
        # self.attgrucell = AttGRUcell(self.node_num, hidden_size)
        self.attgrucell = AttGRUcell(self.node_num, self.hidden_size)
        self.attgru_cell_corr = AttGRUcell(self.node_num, self.hidden_size)
        embedd_size = 256
        # self.atten_layer = AttenLayer(kwargs['window_size'], embedd_size) 
        self.atten_layer = DAgg(kwargs['node_num'])
        self.atten_layer_new = AttentionWithContext(kwargs['window_size'])
        # self.linear = nn.Linear(self.node_num * self.hidden_size * 2, out_dim) # output layer
        #  layer N*hidden_size*2 -> 1  to fc(B*N*)
        self.linear = nn.Linear(self.hidden_size * 2, out_dim) # output
        self.ab_linear = nn.Linear(self.hidden_size, out_dim) # 用于消融实验的输出层
        
        self.abla = False
        self.p_conv = False 
        
        if self.gl_type == 'new':
            self.graph_constructor = Graph_constructor(**kwargs) 
        
        # for dynamic discrete graph learning
        self.kernel_size = [3, 5]
        self.out_channels = [8, 16]
        embedding_dim = 100 
        self.conv1 = nn.Conv1d(1, self.out_channels[0], kernel_size=self.kernel_size[0], stride=1)
        self.conv2 = nn.Conv1d(self.out_channels[0], self.out_channels[1], kernel_size=self.kernel_size[1], stride=1)
        self.conv2d = nn.Conv2d
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        self.hidden_drop = nn.Dropout(0.2)
        self.fc = nn.Linear(self.out_channels[-1]*(self.seq_len-sum(self.kernel_size)+len(self.kernel_size)), embedding_dim)
        self.fc_out = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc_cat = nn.Linear(embedding_dim, 2)
        
        def encode_onehot(labels):
            classes = set(labels) # none zeros row / col
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot # (207 * 207, 207)
        # Generate off-diagonal interaction graph
        off_diag = np.ones([node_num, node_num])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32) # none zero row index (207*207)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32) # none zero col index
        self.rel_rec = torch.FloatTensor(rel_rec).to(self.device) # N * N
        self.rel_send = torch.FloatTensor(rel_send).to(self.device)
    
    def _reset_hidden(self, batch_size):
        # 多个batch并行计算，所以参数中有batch
        self.h0 = torch.zeros(batch_size, self.hidden_size*self.node_num).to(self.device)
        return self.h0
    def _reset_hidden1(self, batch_size):
        # 多个batch并行计算，所以参数中有batch
        self.h1 = torch.zeros(batch_size, self.hidden_size*self.node_num).to(self.device)
        return self.h1
    
    
    def dynamic_adj(self, x:torch.Tensor, adj, temp=0.5):
        '''计算动态邻接矩阵
        x: (batch_size, seq_len, node_num/in_dim)
        adj: (node_num, node_num) prior adjacency matrix
        方案1: 使用gumble softmax 进行构建(但是这样的化模型的超参数又会变多)
            map a pair of vector v_i, v_j to a scalar value theta_ij, the paper we refer to use two 
            fully connect layer to do this.
        '''
        assert sum(self.kernel_size) - len(self.kernel_size) < x.shape[1], "kernel size should be smaller than seq_len"
        batch_size, node_num = x.shape[0], x.shape[2] 
        x = x.permute(0, 2, 1) # (batch_size, node_num, seq_len)
        x = x.reshape(batch_size*node_num, -1) # (batch_size*node_num, seq_len) -> 这边改为batch 和 node做乘积
        x = x.unsqueeze(1) # (batch_size*node_num, 1， seq_len)
        
        x = self.conv1(x) # (B*N) * out_c(8) * (seq_len+1-kernel_size_1)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x) # (B*N) * out_c(16) * (seq_len+2-kernel_size_1-kernel_size_2)
        x = F.relu(x)
        x = self.bn2(x)
        x = x.reshape(batch_size, node_num, -1) # (batch*node_num*-1)
        x = self.fc(x) # (batch*node_num*embedding_dim)
        x = F.relu(x)
        x = x.reshape(batch_size*node_num, -1) # (batch*node_num*-1)
        x = self.bn3(x).reshape(batch_size, node_num, -1) # (batch*node_num*embedding_dim)

        # link predictor
        receivers = torch.matmul(self.rel_rec, x) # (N*N) * (B*N*E) -> (B*N*E)
        senders = torch.matmul(self.rel_send, x) # (N*N) * (B*N*E) -> (B*N*E)
        x = torch.cat([senders, receivers], dim=2) # (B*N*2E)
        x = torch.relu(self.fc_out(x))  # (B*N*E)
        x = self.fc_cat(x) # (B*N*2)

        # # gumbel softmax
        adj = gumbel_softmax(x, temperature=temp, hard=True)    
        adj = adj[:,:, 0].clone().reshape(batch_size, node_num, -1) # 一次性矩阵操作？ ， 不清楚他这边为啥fc的输出层为2
        adj_theta = adj.clone()
        # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
        mask = torch.eye(node_num, node_num).bool().to(self.device)
        adj.masked_fill_(mask, 0) # mask 中的 true fill value 0, 对角线fill 0         
        return adj, adj_theta
    
    def forward(self, x, adj, xd):
        '''
        x: (batch_size, seq_len, node_num)
        adj: (node_num, node_num) prior adjacency matrix
        xd: (batch_size, node_num, node_num, seq_len) delay matrix
        out: shape (batch)
        '''
        # calculate dynamic local adajacency matrix
        if self.gl_type == 'new':
            x_g = x.permute(0, 2, 1)
            x_g = torch.unsqueeze(x_g, 1)
            A_d, A_s, node_embed, graph_loss = self.graph_constructor(x_g) # A_d: (B*N*N)
            Ad_theta = A_d.clone()
        else:
            A_d, Ad_theta = self.dynamic_adj(x, adj) # B*N*N # A_d, discrete, 0 or 1
            graph_loss = torch.tensor([0]).to(self.device)
            # to float 
            graph_loss = graph_loss.float()
            
        atten = self.atten_layer_new(x, adj) 
        
        self.h = self._reset_hidden(x.shape[0]) # 
        self.h_corr = self._reset_hidden1(x.shape[0])

        for t in range(x.shape[1]):
            self.h = self.attgrucell(x[:, t, :], self.h, atten, A_d, xd[...,t])
            self.h_corr = self.attgru_cell_corr(x[:, t, :], self.h_corr, A_d, A_d, xd[...,t])
        
        # 这边直接对self.h, self.h_corr进行预测，可以确定对应的两个消融实验案例
        if not self.abla:
            self.concat_h = torch.cat([self.h, self.h_corr], dim=-1)
            y_hat = self.linear(self.concat_h).squeeze()
        else:
            if self.p_conv:
                y_hat = self.ab_linear(self.h).squeeze()
            else:
                y_hat = self.ab_linear(self.h_corr).squeeze()
        return y_hat, Ad_theta, A_d, atten, graph_loss # last layer，

    
if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard import notebook

    batch_size = 16
    node_num = 8
    in_dim = 12 # seq_len
    hidden_size = 128
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.randn(batch_size, node_num, in_dim).to(device)
    model1 = AttGRU(in_dim, hidden_size).to(device)
    adj = torch.ones(node_num, node_num).to(device)
    out = model1(data,adj)
    print(out.shape)
    print(out)
    
    # # tensorboard 
    # writer = SummaryWriter('./runs/GRU')
    # writer.add_graph(model, data)
    # # writer.add_scalar
    # notebook.list()
    # notebook.start("--logdir ./runs/GRU")
    # out = model(data)
    # print(out)