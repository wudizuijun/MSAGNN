import pandas as pd
import torch
import time

# 前面一个版本没有考虑batch

def generate_adj(node_num):
    adj = torch.rand((node_num, node_num))
    adj = torch.where(adj > 0.5, 1, 0)
    adj[torch.eye(node_num) == 1] = 0
    return adj

def time_rec(func):
    """ decorator, record the running time of a funtion.

    :param _type_ func: _description_
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__} takes {end-start} seconds.")
        return res
    return wrapper

class DAgg:
    ''' Paper II 数据聚合方法
    '''
    def __init__(self, adj:torch.Tensor) -> None:
        """_summary_

        :param torch.Tensor adj: 邻接矩阵, 无自环
        """
        self.adj = adj
        self.node_num = adj.shape[0]
        self._W = torch.rand((adj.shape[0], adj.shape[0])) # N * N
        self.ser_sm = torch.nn.Softmax(dim=0)
        self.softmax = torch.nn.Softmax(dim=1)
        self.parent_node_weigh = torch.nn.Parameter(torch.rand(adj.shape[0])) # 1 * N
    
    @time_rec
    def agg_ser(self, x:torch.Tensor):
        """序列化计算
        :param torch.Tensor x: shape: (N, seq_len), 第一维默认还有个batch,这里先不考虑
        """
        # 先不对Q,K,V进行线性变化， 默认Q=K=V=x
        Att = torch.zeros(x.shape[0], x.shape[0])
        for i in range(self.adj.shape[0]):
            idx = self.adj[:, i].unsqueeze(-1)
            masked_i = x * idx # N * seq
            temp_v = self._W @ masked_i # (N*N) * (N*seq) = N * seq
            # 再做一次mask, 也可以在后面做，但是稀疏矩阵的计算更快？
            # A： 放在后面做了， 如果要加softmax的话
            masked_i = temp_v * idx
            Att[:, i] = masked_i @ x[i, :]# (N*seq) * (1* seq) -> 可能有自动对其的机制
        # Att = Att + torch.eye(self.adj.shape[0])
        Att = Att + torch.diag(self.parent_node_weigh)
        Att = self.ser_sm(Att) * self.adj 
        agg_x = Att.T @ x
        return agg_x, Att
    
    @time_rec
    def agg_parallel(self, x:torch.Tensor):
        """并行计算
        :param torch.Tensor x: shape: (N, seq)
        """
        batch_size = x.shape[0]
        Att =torch.zeros(batch_size, self.node_num, self.node_num)
        adj_flat = self.adj.T.reshape(self.adj.shape[0]*self.adj.shape[0], 1) # 对应idx
        x_dup = x.repeat(1, self.adj.shape[0], 1) # 沿着行进行复制
        masked_x = x_dup * adj_flat # (N*N)*F
        masked_x_f = masked_x.view(batch_size,self.adj.shape[0],self.adj.shape[0],-1) # N*(N*F)

        temp_v = self._W @ masked_x_f # B*N*(N*F)
        masked_i = temp_v.view(batch_size, self.adj.shape[0]**2, -1) * adj_flat # B * (N*N) * seq
        # weighted matrix
        Att = torch.bmm(masked_i.view(batch_size*self.adj.shape[0],self.adj.shape[0],-1), 
                        x.reshape(batch_size*self.adj.shape[0],-1,1)
                        ).reshape(batch_size, self.adj.shape[0], self.adj.shape[0]).permute(0, 2, 1)
        # Att = Att + torch.eye(self.adj.shape[0])
        Att = Att + torch.diag(self.parent_node_weigh)
        Att = self.softmax(Att) * self.adj 
        agg_x = Att.permute(0, 2, 1) @ x
        return agg_x, Att
    
    @property 
    def W(self):
        return self._W
    
    
if __name__ == "__main__":
    torch.random.manual_seed(1)

    node_num, seq_len = 3, 5
    batch_size = 2
    adj = generate_adj(node_num)
    adj[0, -1] = 1
    # print('adj: ', adj) 
    data = torch.rand((batch_size, node_num, seq_len))

    dagg = DAgg(adj)
    res_1, att_1 = dagg.agg_ser(data[0,...])
    # print(res_1)
    print(att_1)
    res_2, att_2 = dagg.agg_parallel(data)
    # print(res_2)#.shape)
    print(att_2[0])
    print(res_1 - res_2[0,...])
    
    
    