# Modules used in paper II

import numpy as np
import torch
import scipy
import time
from torch.nn import functional as F

class TLModule:
    ''' 先参考auotoformer那个atuo-correlation.
    '''
    def __init__(self) -> None:
        pass
    
    
    def forward(self, x, adj):
        # 计算子节点与父节点的相关性
        pass

def lag_cal(x_0, x_1, max_lag=10):
    """计算时滞

    :param _type_ x_0: child node
    :param _type_ x_1: parent node
    :param int max_lag: max lag num, 依据流程中物料进入到出去最大时间
    :return _type_: _description_
    """
    assert max_lag < len(x_0), "max_lag should be less than len(x_0)"
    time_lag = scipy.signal.correlate(x_0, x_1)
    # lag = len(time_lag)//2 -(np.argmax(time_lag[len(time_lag)//2-max_lag:len(time_lag)]) + len(time_lag)//2 - max_lag)
    # simply lag cal formula
    lag = max_lag -np.argmax(time_lag[len(time_lag)//2-max_lag:len(time_lag)])  # 先不考虑lag都是负的情况
    return lag

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

# @time_rec
def TimeDelayCal(X, adj:torch.Tensor):
    """_summary_
     Input
    :param _type_ X: shape:  (N * seq)
    :param _type_ adj: adjacency matrix, shape: (N * N), diagnal elements set to 0
    :return _type_: X_delay(N*N*seq), 经过平移的数据
                 D:
    """
    assert sum(adj.diagonal()) == 0, "diagonal elements of adj should be 0"
    
    D = np.zeros_like(adj)
    X_delay = np.zeros((X.shape[0],X.shape[0],X.shape[-1]))
    d_num = X.shape[-1] // 2
    for i in range(adj.shape[0]):
        node_i = X[i]
        parent_node = X[adj[i].T == 1]
        for j in range(len(parent_node)):
            node_j = parent_node[j]
            # 我们希望计算出的时滞都是负的，因果指向嘛
            D[i, j] = lag_cal(node_i, node_j)
            X_delay[i, j] = np.roll(node_j, D[i, j].astype(int).item())
    return X_delay, D

class DataAggregator:
    ''' 用于聚合数据
    '''
    def __init__(self) -> None:
        pass
    
    def forward(self, x, adj):
        '''
        input: data, adj
        ouput: Aggregated data, weighted adj
        '''
        pass

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to("cuda")
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
  if hard:
      shape = logits.size()
      _, k = y_soft.data.max(-1)
      y_hard = torch.zeros(*shape).to('cuda')
      y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
      y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
  else:
      y = y_soft
  return y

    
if __name__ == "__main__":
    ''' test gumbel_softmax '''
    Node_num = 3
    logits = torch.randn(Node_num, Node_num).to('cuda')
    print(logits)
    y = gumbel_softmax(logits.flatten(), temperature=1, hard=True)
    print(y.reshape(Node_num, Node_num))
    
    
    '''  test lag_cal  '''
    # N = 43
    # T = 100
    # # 随机生成邻接矩阵
    # adj = np.random.randint(0, 2, (N, N))
    # # set diagnal elements to 0
    # adj = adj - np.diag(np.diag(adj))
    # X = np.random.randn(N, 100)
    # X_d, D = TimeDelayCal(X, adj)
    
    # print(D)
    
    
    