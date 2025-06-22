### LSTM
### paperII model
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import matplotlib.pyplot as plt 

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
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        
        self.W_ir = nn.Linear(in_dim, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size)
        self.W_iz = nn.Linear(in_dim, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size)
        self.W_in = nn.Linear(in_dim, hidden_size)
        self.W_hn = nn.Linear(hidden_size, hidden_size)
        
    def _gc_aggregate(self, x, adj):
        """Graph Convolution aggregate.
        x: graph data, shape:
        adj: adjacent matrix or weighted matrix. (A_ij represents node j to node i's weight)
        """
        # insert a dimension to x
        x = x.unsqueeze(-1) # (1, node_num, in_dim)
        return torch.matmul(adj, x).squeeze()
    
    def forward(self, x, h, adj):
        temp = self._gc_aggregate(x, adj)
        r = nn.Sigmoid()(self._gc_aggregate(x, adj) + self.W_hr(h)) # all hidden_size r和z可以放到一块计算，计算完成后再进行拆分
        z = nn.Sigmoid()(self._gc_aggregate(x, adj) + self.W_hz(h))
        n = nn.Tanh()(self._gc_aggregate(x, adj) + r*self.W_hn(h))
        h_prim = (1-z)*n + z*h
        return h_prim

class AttGRU(nn.Module):
    def __init__(self,node_num, in_dim, hidden_size, num_layer=1, out_dim=1) -> None:
        """_summary_

        :param _type_ in_dim: feature num / seq_len
        :param _type_ hidden_size: _description_
        :param int num_layer: _description_, defaults to 1
        :param int out_dim: _description_, defaults to 1
        """
        super().__init__()
    
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.node_num = node_num
    
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.gru = nn.GRU(in_dim, hidden_size, num_layer)
        # self.attgrucell = AttGRUcell(self.node_num, hidden_size)
        self.attgrucell = AttGRUcell(self.node_num, self.node_num)
        embedd_size = 256
        self.atten_layer = AttenLayer(in_dim, embedd_size) 
        self.linear = nn.Linear(node_num, out_dim)
    
    def _reset_hidden(self, batch_size):
        # 多个batch并行计算，所以参数中有batch
        self.h0 = torch.zeros(batch_size, self.node_num).to(self.device)
        return self.h0
    
    def forward(self, x, adj):
        '''
        x: (batch_size, seq_len, in_dim)
        '''
        _, atten = self.atten_layer(x, adj)
        self.h = self._reset_hidden(x.shape[0])
        # fig, ax = plt.subplots(figsize=(10,10))
        # ax.imshow(atten[0].cpu().detach().numpy())
        # plt.show()
        # 所有seq计算一次attention
        
        for t in range(x.shape[1]):
            self.h = self.attgrucell(x[:, t, :], self.h, atten)
        return self.linear(self.h).squeeze() # last layer
    
    # def forward(self, x, h, adj):
    #     r = nn.Sigmoid()(self.W_ir(x) + self.W_hr(h)) # all hidden_size
    #     z = nn.Sigmoid()(self.W_iz(x) + self.W_hz(h))
    #     n = nn.Tanh()(self.W_in(x) + r*self.W_hn(h))
    #     h_prim = (1-z)*n + z*h
    #     return h_prim
    
                
    
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