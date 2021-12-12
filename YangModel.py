#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   YangModel.py
@Time    :   2021/12/11 19:28:42
@Author  :   LiFeng
@Contact :   2807229316@qq.com
@Desc    :   使用GCN的二分类模型
'''
import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_size=8):
        super(GAT, self).__init__()
        conv = gnn.GCNConv
        linear = nn.Linear
        
        self.conv1 = conv(num_features, hidden_size)
        self.conv2 = conv(hidden_size, hidden_size)
        self.conv3 = conv(hidden_size, 1)

        self.linear1 = linear(2 * hidden_size + 1, 16)
        self.linear2 = linear(16, 2)

    def forward(self, data):
        fea_list = []
        x, edge_index = data.x, data.edge_index
                
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        fea_list.append(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        fea_list.append(x)
        x = self.conv3(x, edge_index)
        fea_list.append(torch.relu(x))

        x = torch.cat(fea_list, dim=1)

        offset, idx = 0, []
        for g_index in range(data.num_graphs):
            graph = data[g_index]
            idx.append(offset + graph.target_id[0])
            offset += graph.num_nodes

        x = torch.relu(self.linear1(x[idx, :]))
        x = self.linear2(x)
        return x
