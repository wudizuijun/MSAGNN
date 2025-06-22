import pandas as pd
import torch
import time


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
        self._W = torch.rand((adj.shape[0], adj.shape[0])) # N * N
        self.softmax = torch.nn.Softmax(dim=0)
    
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
        # Att = self.softmax(Att) * self.adj 
        Att = Att + torch.eye(self.adj.shape[0])
        agg_x = Att.T @ x
        return agg_x
    
    @time_rec
    def agg_parallel(self, x:torch.Tensor):
        """并行计算
        :param torch.Tensor x: shape: (N, seq)
        """
        Att =torch.zeros(x.shape[0], x.shape[0])
        adj_flat = self.adj.T.reshape(self.adj.shape[0]*self.adj.shape[0], 1) # 对应idx
        x_dup = x.repeat(self.adj.shape[0], 1) # 沿着行进行复制
        masked_x = x_dup * adj_flat # (N*N)*F
        masked_x_f = masked_x.view(self.adj.shape[0],self.adj.shape[0],-1) # N*(N*F)

        temp_v = self._W @ masked_x_f # N*N*F
        masked_i = temp_v.view(self.adj.shape[0]**2, -1) * adj_flat # (9*5)
        # weighted matrix
        Att = torch.bmm(masked_i.view(self.adj.shape[0],self.adj.shape[0],-1), 
                        x.reshape(self.adj.shape[0],-1,1)
                        ).squeeze().T
        Att = Att + torch.eye(self.adj.shape[0])
        agg_x = Att.T @ x
        return agg_x
    
    @property 
    def W(self):
        return self._W
    
    
if __name__ == "__main__":
    torch.random.manual_seed(1)

    node_num, seq_len = 270, 12
    adj = generate_adj(node_num)
    adj[0, -1] = 1
    # print('adj: ', adj) 
    data = torch.rand((node_num, seq_len))

    dagg = DAgg(adj)
    res_1 = dagg.agg_ser(data)
    # print(res_1)
    res_2 = dagg.agg_parallel(data)
    # print(res_2)
    
    
    