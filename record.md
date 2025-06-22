### AttenGRU
#### 问题:
- 1. hidden size 不得不保持与node num 同步， 为了进行卷积操作
- 2. 仔细想想，那个使用自注意力计算边的权重时候，还没仅考虑了pairwise这种关系，仅在softmax考虑了相邻节点。
- 3. 有必要对GRU中每个层的矩阵计算都做线性乘法嘛？
- 4. 神经网络数据预处理中的归一化以及逆归一化真的有用嘛，特别是在对y进行逆归一化时。
- 5. 在催化裂化装置上把这个图做出来，分析影响某值的重要变量，将是绝杀。
  - ![Alt text](image.png)
- 6. 图构建那边加个交叉熵损失函数，true对应先验的图（能通过拓扑图确定就通过，不行就那啥stastic method 确定）
- 7. !!!  虽然每个时间步图卷积中的节点的维数只有1维，但是可以和hidden state concate 一下捏。
- 阅读DCRNN 代码有感
  - Metr-LA
  - original format: all time * node num
    每个节点的数据是2维的额原因在于，第二个维度的信息为时间，那么我们能不能再考虑时滞的情况下也考虑时间信息，则我们的每个节点的维数为3。此外，时滞这块的数据，能不能再做数据集的时候就直接做好。（起飞）
  - need to know: 每个节点数据的维数为2维时，怎么考虑那个数据聚合（具体体现再变量之间的关系处理）
    - in data aggregation: multi-dimenstion predict single-dimension -> fully connect not slice(也是从use more information 的角度进行出发的)


### Date: 2024-01-03
- 代码框架修改，解耦

### Date: 2024-01-05
- 以实现代码解耦
- 目前待优化： 时滞数据集的制作以及模型的输入维数对不上
- 未做：那几个交通数据集


### 2025-02-17
- 替换一下模型注意力
  - T-GCN， DCRNN， GAT
- T-GCN:
  - A hat static ?
  - 问题： 使用的图结构好像是静态的，不适用于本文对应章节所提出的模块
- DCRNN: 
  - 这个网络也是使用GCN网络，并不会涉及到图注意力的计算
- GAT：
  - 注意力的代码替换：
    - GraphAttentionLayerV2
      - input: (batch size, node num, embedding dim)
      - output: (batch size, node num, node num)
    - AttentionWithContext
      - input:
        - x: (batch size, node num, seq len)
        - adj: (node num, node num)
      - output: (batch size, node num, node num)