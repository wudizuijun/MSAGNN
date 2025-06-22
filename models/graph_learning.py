import torch
from torch import nn
import torch.nn.functional as F

class fc_layer(nn.Module):
    def __init__(self, in_channels, out_channels, need_layer_norm):
        super(fc_layer, self).__init__()
        self.linear_w = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_uniform_(self.linear_w.data, gain=1.414)

        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=[1, 1], bias=True)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.need_layer_norm = need_layer_norm

    def forward(self, input):
        '''
        input = batch_size, in_channels, nodes, time_step
        output = batch_size, out_channels, nodes, time_step
        '''
        if self.need_layer_norm:
            result = F.leaky_relu(torch.einsum('bani,io->bano ', [input.transpose(1, -1), self.linear_w]))\
                     # + self.layer_norm(self.linear(input).transpose(1, -1))
        else:
            result = F.leaky_relu(torch.einsum('bani,io->bano ', [input.transpose(1, -1), self.linear_w])) \
                     # + self.linear(input).transpose(1, -1)
        return result.transpose(1, -1)


class gatedFusion_1(nn.Module):
    def __init__(self, dim, device):
        super(gatedFusion_1, self).__init__()
        self.device = device
        self.dim = dim
        self.w = nn.Linear(in_features=dim, out_features=dim)
        self.t = nn.Parameter(torch.zeros(size=(self.dim, self.dim)))
        nn.init.xavier_uniform_(self.t.data, gain=1.414)
        self.norm = nn.LayerNorm(dim)
        self.re_norm = nn.LayerNorm(dim)

        self.w_r = nn.Linear(in_features=dim, out_features=dim)
        self.u_r = nn.Linear(in_features=dim, out_features=dim)

        self.w_h = nn.Linear(in_features=dim, out_features=dim)
        self.w_u = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, batch_size, nodevec, time_node):

        if batch_size == 1 and len(time_node.shape) < 3:
            time_node = time_node.unsqueeze(0)

        nodevec = self.norm(nodevec)
        node_res = self.w(nodevec) + nodevec
        # node_res = batch_size, nodes, dim
        node_res = node_res.unsqueeze(0).repeat(batch_size, 1, 1)

        time_res = time_node + torch.einsum('bnd, dd->bnd', [time_node, self.t])

        # z = batch_size, nodes, dim
        z = torch.sigmoid(node_res + time_res)
        r = torch.sigmoid(self.w_r(time_node) + self.u_r(nodevec).unsqueeze(0).repeat(batch_size, 1, 1))
        h = torch.tanh(self.w_h(time_node) + r * (self.w_u(nodevec).unsqueeze(0).repeat(batch_size, 1, 1)))
        res = torch.add(z * nodevec, torch.mul(torch.ones(z.size()).to(self.device) - z, h))

        return res

class Graph_learn(nn.Module):
    def __init__(self, node_dim, heads, head_dim, nodes=207, eta=1,
                 gamma=0.001, dropout=0.5, n_clusters=5, l1=0.001, l2=0.001):
        super(Graph_learn, self).__init__()

        self.D = heads * head_dim  # node_dim #
        self.heads = heads
        self.dropout = dropout
        self.eta = eta
        self.gamma = gamma

        self.head_dim = head_dim
        self.node_dim = node_dim
        self.nodes = nodes

        self.l1 = l1
        self.l2 = l2

        self.query = fc_layer(in_channels=node_dim, out_channels=self.D, need_layer_norm=False)
        self.key = fc_layer(in_channels=node_dim, out_channels=self.D, need_layer_norm=False)
        self.value = fc_layer(in_channels=node_dim, out_channels=self.D, need_layer_norm=False)
        self.mlp = nn.Conv2d(in_channels=self.heads, out_channels=self.heads, kernel_size=(1, 1), bias=True)

        self.bn = nn.LayerNorm(node_dim)

        self.w = nn.Parameter(torch.zeros(size=(nodes, node_dim)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.attn_static = nn.LayerNorm(nodes)
        self.skip_norm = nn.LayerNorm(nodes)
        self.attn_norm = nn.LayerNorm(nodes)
        self.linear_norm = nn.LayerNorm(nodes)
        self.attn_linear = nn.Parameter(torch.zeros(size=(nodes, nodes)))
        nn.init.xavier_uniform_(self.attn_linear.data, gain=1.414)
        self.attn_linear_1 = nn.Parameter(torch.zeros(size=(nodes, nodes)))
        nn.init.xavier_uniform_(self.attn_linear_1.data, gain=1.414)
        self.static_inf_norm = nn.LayerNorm(nodes)
        self.attn_norm_1 = nn.LayerNorm(nodes)
        self.attn_norm_2 = nn.LayerNorm(nodes)

    def forward(self, nodevec_fusion, nodevec_s, node_input, nodevec_dy, batch_size=64):
        batch_size, nodes, node_dim = batch_size, self.nodes, self.node_dim
        node_orginal = nodevec_s
        # Static Graph Structure Learning
        adj_static = self.static_graph(node_orginal)

        nodevec_fusion = self.bn(nodevec_fusion)

        # Inductive bias
        static_graph_inf = self.static_inf_norm(torch.mm(nodevec_dy, nodevec_dy.transpose(1, 0)))

        # residual connection in Dynamic relationship construction
        nodevec1_1 = torch.einsum('bnd, nl -> bnl', nodevec_fusion, self.w) + nodevec_fusion
        skip_atten = torch.einsum('bnd,bdm->bnm', nodevec1_1, nodevec1_1.transpose(-1, -2))
        skip_atten = self.skip_norm(skip_atten)

        # Multi-Head Adjacent mechanism
        nodevec_fusion = nodevec_fusion.unsqueeze(1).transpose(1, -1)
        query = self.query(nodevec_fusion)
        key = self.key(nodevec_fusion)
        # value = self.value(nodevec_fusion)
        key = key.squeeze(-1).contiguous().view(batch_size, self.heads, self.head_dim, nodes)
        query = query.squeeze(-1).contiguous().view(batch_size, self.heads, self.head_dim, nodes).transpose(-1, -2)
        attention = torch.einsum('bhnd, bhdu-> bhnu', query, key)
        attention /= (self.head_dim ** 0.5)
        attention = F.dropout(attention, self.dropout, training=self.training)
        attention = self.mlp(attention) + attention
        adj_bf = self.attn_norm(torch.sum(attention, dim=1)) + skip_atten

        # feedforward neural network
        adj_af = F.relu(torch.einsum('bnm, ml->bnl', self.linear_norm(adj_bf), self.attn_linear))
        adj_af = torch.einsum('bnm, ml -> bnl', adj_af, self.attn_linear_1)

        # add & norm
        dy_adj_inf = self.attn_norm_1(adj_af + adj_bf)
        dy_adj_inf = F.dropout(dy_adj_inf, self.dropout, training=self.training)

        # add Inductive bias
        static_graph_inf = static_graph_inf.unsqueeze(0).repeat(batch_size, 1, 1)
        dy_adj = self.attn_norm_2(dy_adj_inf + static_graph_inf)

        # The final inferred dynamic graph structure
        adj_dynamic = F.softmax(F.relu(dy_adj), dim=2)
        adj_static = adj_static.unsqueeze(0).repeat(batch_size, 1, 1)

        # Graph Structure Learning Loss
        gl_loss =  torch.tensor([0]).to(nodevec_fusion.device).float()
        self.training = True
        if self.training:
            gl_loss = self.graph_loss_orginal(node_input, adj_static, self.eta, self.gamma, self.l1, self.l2)
        return adj_dynamic, adj_static, node_orginal, gl_loss,

    def static_graph(self, nodevec):
        resolution_static = torch.mm(nodevec, nodevec.transpose(1, 0))
        resolution_static = F.softmax(F.relu(self.attn_static(resolution_static)), dim=1)
        return resolution_static

    def graph_loss_orginal(self, input, adj, eta=1, gamma=0.001, l1=0.001, l2=0.001):
        B, N, D = input.shape
        x_i = input.unsqueeze(2).expand(B, N, N, D)
        x_j = input.unsqueeze(1).expand(B, N, N, D)
        dist_loss = torch.pow(torch.norm(x_i - x_j, dim=3), 2) * adj
        dist_loss = torch.sum(dist_loss, dim=(1, 2))
        f_norm = torch.pow(torch.norm(adj, dim=(1, 2)), 2)
        # gl_loss = dist_loss + gamma * f_norm
        gl_loss = dist_loss * l1 + f_norm * l2
        return gl_loss

class Graph_constructor(nn.Module):
    def __init__(self, **kwargs):
        super(Graph_constructor, self).__init__()
        
        nodes = kwargs['node_num']
        # nodes = 207
        dim = kwargs['ng_dim']
        device = kwargs['device']
        time_step = kwargs['window_size']
        cout = kwargs['ng_cout']
        heads = kwargs['ng_heads']
        head_dim = kwargs['ng_head_dim']
        eta = kwargs['ng_eta']
        gamma = kwargs['ng_gamma']
        dropout = kwargs['ng_dropout']
        m = kwargs['ng_m']
        batch_size = kwargs['batch_size']
        is_add1 = kwargs['ng_is_add1']
        in_dim = kwargs['in_dim']
        
        self.embed1 = nn.Embedding(nodes, dim)

        self.m = m
        self.embed2 = nn.Embedding(nodes, dim)
        for param in self.embed2.parameters():
            param.requires_grad = False
        for para_static, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_static.data = para_w.data

        self.device = device
        self.nodes = nodes
        self.time_step = time_step
        if is_add1:
            time_length = time_step + 1
        else:
            time_length = time_step

        self.trans_Merge_line = nn.Conv2d(in_dim, dim, kernel_size=(1, time_length), bias=True) # cout
        self.gate_Fusion_1 = gatedFusion_1(dim, device)

        self.graph_learn = Graph_learn(node_dim=dim, heads=heads, head_dim=head_dim, nodes=nodes,
                                       eta=eta, gamma=gamma, dropout=dropout, l1=kwargs['lambda1'], l2=kwargs['lambda2'])

        self.dim_to_channels = nn.Parameter(torch.zeros(size=(heads * head_dim, cout * time_step)))
        nn.init.xavier_uniform_(self.dim_to_channels.data, gain=1.414)
        self.skip_norm = nn.LayerNorm(time_step)
        self.time_norm = nn.LayerNorm(dim)

    def forward(self, input):
        # input = (batch_size, feature_dim, node_num, seq_len)
        batch_size, nodes, time_step = input.shape[0], self.nodes, self.time_step
        # Momentum update
        for para_dy, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_dy.data = para_dy.data * self.m + para_w.data * (1 - self.m)

        node_input = input

        node_input = self.time_norm(self.trans_Merge_line(node_input).squeeze(-1).transpose(1, 2))
        idx = torch.arange(self.nodes).to(self.device)
        nodevec_static = self.embed1(idx)

        nodevec_dy = self.embed2(idx)
        # Information fusion module
        nodevec_fusion = self.gate_Fusion_1(batch_size, nodevec_static, node_input) + nodevec_static
        # graph learning module (static and dynamic graph)
        adj = self.graph_learn(nodevec_fusion, nodevec_static, node_input, nodevec_dy, batch_size)
        return adj 


if __name__ == '__main__':
    import sys
    sys.path.append(r'./../')
    from utils.Utils import Utils as utils 
    args = utils.get_args()
    d_args = vars(args)
    
    
    graph_constructor = Graph_constructor(**d_args)
    graph_constructor = graph_constructor.to(args.device)
    
    test_data = torch.randn(args.batch_size, args.in_dim, 207, args.window_size)
    test_data = test_data.to(args.device)
    
    adj = graph_constructor(test_data)
    print(adj)