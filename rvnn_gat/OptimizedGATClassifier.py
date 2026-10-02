from importlib import import_module

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import dgl
import numpy as np
import torch as th
from torch.utils.data import DataLoader

from .utils import NodesDataset, init_network


class OptimizedGATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, head_num, n_classes, embedding_matrix, node_vec_stg='mean', merge='cat',
                 device=None, layer_num=1, activation=F.relu, feat_drop=0., attn_drop=0.):
        super(OptimizedGATClassifier, self).__init__()
        self.merge = merge #cat
        self.node_vec_stg = node_vec_stg #mean
        if node_vec_stg != 'mean':
            x = import_module('gnnmodels.' + node_vec_stg)#动态调用这个模型（自己写的位置里）
            self.config = x.Config(embedding_matrix, hidden_dim, device)
            model = x.Model(self.config).to(device)
            if node_vec_stg != 'Transformer':
                init_network(model)
            self.node_model = model
        self.device = device
        
        self.gat_layers = nn.ModuleList()
        # input projection (no residual)#128       128       5 多层
        self.gat_layers.append(GATConv(in_dim, hidden_dim, head_num, feat_drop=feat_drop, attn_drop=attn_drop,
                                       residual=False, activation=activation)) # F.relu 激活函数
        if merge == 'cat':
            # hidden layers
            for i in range(1, layer_num):
                #in_feats,out_feats
                #ParameterConfig.HEAD_NUM = 5
                self.gat_layers.append(GATConv(hidden_dim * head_num, hidden_dim, head_num, feat_drop=feat_drop,
                                               attn_drop=attn_drop, residual=True, activation=activation))
            # add another graph convolution layer for output projection
            self.gat_layers.append(GATConv(hidden_dim * head_num, hidden_dim, num_heads=1, feat_drop=feat_drop,
                                           attn_drop=attn_drop, residual=True, activation=activation)) # 单层、残差
        else:
            # hidden layers
            for i in range(1, layer_num):
                self.gat_layers.append(GATConv(hidden_dim, hidden_dim, head_num, feat_drop=feat_drop,
                                               attn_drop=attn_drop, residual=True, activation=activation))
            # add another graph convolution layer for output projection
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim, num_heads=1, feat_drop=feat_drop,
                                           attn_drop=attn_drop, residual=True, activation=activation))

        self.classify = nn.Linear(hidden_dim, n_classes)
        self.embedding_matrix = embedding_matrix
        self.is_train_mode = True

    def generate_node_vecs(self, g):
        nodes_attrs = g.ndata['w']#点特征
        torches = []
        if self.node_vec_stg == 'mean':
            for ins_list in nodes_attrs:
                vec_list = []
                for ins in ins_list:
                    vec = self.embedding_matrix[ins.item()]
                    vec_list.append(vec)
                arr = np.array(vec_list)
                vec = arr.mean(axis=0)
                torches.append(th.tensor(vec).to(self.device))
            return th.stack(torches, 0)
        else:
            nodes_db = NodesDataset(nodes_attrs)
            data_loader = DataLoader(nodes_db, batch_size=self.config.batch_size, shuffle=False)
            node_vec_list = []
            if self.is_train_mode:
                self.node_model.train()
            else:
                self.node_model.eval()
            for iter, batch in enumerate(data_loader):
                # tensor_batch = batch.clone().detach().to(self.device)
                # batch_out = self.node_model(tensor_batch)
                # print(batch) 每个分支的点特征
                batch_out = self.node_model(batch)
                node_vec_list.append(batch_out)
                #print(batch_out.size())#torch.Size([128, 192])
            return th.cat(node_vec_list, dim=0)

    '''def forward(self, g):
        h = g.ndata['w']
        attention_weights_list = []  # 用于存储所有层的注意力权重

        for i, layer in enumerate(self.gat_layers):
            if i != 0:
                h = th.transpose(h, 0, 1)
                heads = [hd for hd in h]
                h = th.cat(heads, dim=1) if self.merge == 'cat' else th.mean(th.stack(heads), dim=0)

            h, attention_weights = layer(g, h, get_attention=True)  # 获取注意力权重
            attention_weights_list.append(attention_weights)  # 记录注意力

        h = h.squeeze(1)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        attention_weights_tensor = th.cat([attw.flatten() for attw in attention_weights_list])

        return hg, attention_weights_tensor  # 返回图表表示和注意力权重'''
    
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        # h = g.in_degrees().view(-1, 1).float()
        # 使用节点度作为初始节点特征（对于无向图，in-degree与out_degree相同）。
        #h = self.generate_node_vecs(g).float().to(self.device) # 对点特征进行处理
        h = g.ndata['w']   # 需要的维度是 shape=(128,)
        #print(h.size())
        # Perform graph convolution and activation function.
        for i, layer in enumerate(self.gat_layers):
            if i != 0:
                # concat on the output feature dimension (dim=1)
                #print("1",h.size())
                h = th.transpose(h, 0, 1)#转置
                #print("2",h.size())
                heads = [hd for hd in h]
                if self.merge == 'cat':
                    h = th.cat(heads, dim=1)
                    #print("3",h.size())
                else:
                    # merge using average
                    h = th.mean(th.stack(heads), dim=0)
            #print("4",h.size())
            h = layer(g, h)
            #print("5",h.size())
        h = h.squeeze(1)
        #print("6",h.size())
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        # 通过对所有节点表示求平均值来计算图形表示
        hg = dgl.mean_nodes(g, 'h')
        return hg
        # hg = dgl.max_nodes(g, 'h')
       # return self.classify(hg)