import torch
import torch.nn as nn
import torch.nn.functional as F

class DAgg(torch.nn.Module):
    ''' 数据聚合方法
    '''
    def __init__(self, node_num) -> None:
        """_summary_

        :param torch.Tensor adj: 邻接矩阵, 无自环
        """
        # self.adj = adj
        super().__init__()
        self.node_num = node_num
        self._W = torch.rand((node_num, node_num)).to('cuda') # N * N
        # self._W = torch.eye(node_num).to('cuda') # test用
        self.ser_sm = torch.nn.Softmax(dim=0)
        self.softmax = torch.nn.Softmax(dim=1)
        self.parent_node_weigh = torch.nn.Parameter(torch.rand(node_num)) # 1 * N
        self.cycle_adj = None
        self.Wq = torch.nn.Linear(node_num, node_num).to('cuda')
        self.Wk = torch.nn.Linear(node_num, node_num).to('cuda')
    

    
    def forward(self, x:torch.Tensor, adj:torch.Tensor):
        """并行计算， 目前在框架中的使用只要得到atten就行
        :param torch.Tensor x: shape: (B * seq/feature_num * N)
        Input:
            x: (B * S * N), 
            adj: (N * N)
        return Att: Att_{ij}: 第i个节点对第j个节点的注意力, B*N*N
        """
        Q = self.Wq(x).permute(0, 2, 1)  # B * N * seq
        K = self.Wk(x).permute(0, 2, 1) 
        x = x.permute(0, 2, 1) # B * N * seq
        # Q = self.Wq(x) # B * N * seq
        # K = self.Wk(x)
        
        self.adj = adj
        if self.cycle_adj is None:
            self.cycle_adj = torch.eye(self.adj.shape[0]).to('cuda') + self.adj
        batch_size = x.shape[0]
        Att = torch.zeros(batch_size, self.node_num, self.node_num)
        adj_flat = self.adj.T.reshape(self.adj.shape[0]*self.adj.shape[0], 1) # 对应idx ###### 这块adj装置的意义何在
        x_dup = K.repeat(1, self.adj.shape[0], 1) # 沿着行进行复制
        masked_x = x_dup * adj_flat # (N*N)*F
        masked_x_f = masked_x.view(batch_size,self.adj.shape[0],self.adj.shape[0],-1) # N*(N*F) 父节点连接的子节点

        temp_v = masked_x_f
        temp_v = self._W @ masked_x_f # B*N*(N*F) # check done(对最后两个维度做矩阵乘法的)
        masked_i = temp_v.view(batch_size, self.adj.shape[0]**2, -1) * adj_flat # B * (N*N) * seq | （N*N*1),筛选出父节点连接的子节点
        # 某个batch: masked_i(N*N*seq): 前N行非0行表示连接第一个节点的父节点  
        # weighted matrix
        Att = torch.bmm(masked_i.view(batch_size*self.adj.shape[0],self.adj.shape[0],-1),  # K
                        Q.reshape(batch_size*self.adj.shape[0],-1,1) # Q
                        ).reshape(batch_size, self.adj.shape[0], self.adj.shape[0]).permute(0, 2, 1)
        # Att = Att + torch.eye(self.adj.shape[0]).to('cuda')
        Att = Att + torch.diag(self.parent_node_weigh).to('cuda')

        Att = self.softmax(Att) * self.cycle_adj # B * N * N
        agg_x = Att.permute(0, 2, 1) @ x
        return agg_x, Att
    
    @property 
    def W(self):
        return self._W
    
def generate_adj(node_num):
    adj = torch.rand((node_num, node_num))
    adj = torch.where(adj > 0.5, 1, 0)
    adj[torch.eye(node_num) == 1] = 0
    return adj

class AttentionWithContext(nn.Module):
    def __init__(self, input_dim):
        super(AttentionWithContext, self).__init__()
        self.W = nn.Linear(input_dim, input_dim, bias=False)  # learnable transformation
        self.a = nn.Linear(input_dim * 3, 1, bias=False)  # attention scoring
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, adj):
        """
        Args:
            x: Tensor, shape (batch_size, node_num, seq_len)
            adj: Tensor, shape (node_num, node_num) - adjacency matrix

        Returns:
            A: Tensor, shape (batch_size, node_num, node_num), attention matrix，
                A_{ij}, node i to node j impact
        """
        
        x = x.permute(0, 2, 1)  # shape: (batch_size, seq_len, node_num)
        batch_size, node_num, seq_len = x.shape

        # Transform input features using W
        h = self.W(x)  # shape: (batch_size, node_num, seq_len)

        # Compute context matrix
        h_expanded = h.unsqueeze(2).expand(-1, -1, node_num, -1)  # shape: (batch_size, node_num, node_num, seq_len)
        h_transposed = h.unsqueeze(1).expand(-1, node_num, -1, -1)  # shape: (batch_size, node_num, node_num, seq_len)
        similarity = torch.sum(h_expanded * h_transposed, dim=-1)  # shape: (batch_size, node_num, node_num)

        context = torch.zeros_like(h_expanded)  # shape: (batch_size, node_num, node_num, seq_len)
        for i in range(node_num):
            mask = adj[i].unsqueeze(0).expand(batch_size, -1).clone()   # shape: (batch_size, node_num)
            mask[:, i] = 0  # Exclude the current node
            # Expand mask to match the shape of h_transposed
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1)  # Shape: (batch_size, node_num, node_num, 1)

            # Compute weighted_h
            weighted_h = (similarity[:, i, :].unsqueeze(-1).unsqueeze(1) * h_transposed).masked_fill(mask_expanded == 0, 0)
            # weighted_h = (similarity[:, i, :].unsqueeze(-1).unsqueeze(1) * h_transposed).masked_fill(mask.unsqueeze(-1) == 0, 0)
            context[:, :, i, :] = weighted_h.sum(dim=2)  # Sum over neighbors

        # Compute attention scores
        concat_features = torch.cat([h_expanded, h_transposed, context], dim=-1)  # shape: (batch_size, node_num, node_num, 3 * seq_len)
        scores = self.leaky_relu(self.a(concat_features).squeeze(-1))  # shape: (batch_size, node_num, node_num)

        # Apply softmax across neighbors for each node i
        scores = scores.masked_fill(adj.unsqueeze(0) == 0, -9e15)  # Mask non-neighbors
        A = F.softmax(scores, dim=-1)  # Normalize along the last dimension (neighbors)

        return A
                


if __name__ == '__main__':
    torch.random.manual_seed(1)

    node_num, seq_len = 3, 2 
    batch_size = 1
    # adj = generate_adj(node_num).to('cuda')
    # adj[-1, 0] = 1
    adj = torch.Tensor([
        [0, 1, 1], [0, 0, 1], [1, 1, 0]
    ]).to('cuda')
    # print('adj: ', adj) 
    data = torch.rand((batch_size, seq_len, node_num)).to('cuda')
    data = torch.Tensor([
        [1,2],
        [3,4],
        [5,6]
    ]).T.to('cuda').unsqueeze(0)

    dagg = DAgg(node_num)
    res_2, att_2 = dagg(data, adj)
    print(att_2[0])
    
