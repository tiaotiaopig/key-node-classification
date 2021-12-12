#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   YangDataset.py
@Time    :   2021/12/11 10:59:29
@Author  :   LiFeng
@Contact :   2807229316@qq.com
@Desc    :   None
'''

from typing import Tuple
from SIR import SIR_network

import torch, os, random
import numpy as np
import networkx as nx
import multiprocessing as mp
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx, k_hop_subgraph


def pairwise2matrix_nx(filepath : str) -> Tuple[nx.Graph, Data]:
    '''使用.txt存储的点对，先转化为nx.Graph,再转化为PyG.Data'''
    pair_wise = np.loadtxt(filepath, dtype=np.int32)
    graph = nx.from_edgelist(pair_wise, create_using=nx.Graph)
    return graph, from_networkx(graph)

def compute_node_feature(graph: nx.Graph) -> torch.Tensor:
    '''分别计算了四种中心性指标作为节点的特征'''
    degree_cen = list(nx.degree_centrality(graph).values())
    sub_cen = list(nx.subgraph_centrality(graph).values())
    close_cen = list(nx.closeness_centrality(graph).values())
    betweenness_cen = list(nx.betweenness_centrality(graph).values())
    fea_list = [degree_cen, sub_cen, close_cen, betweenness_cen]
    return torch.tensor(fea_list, dtype=torch.float32).T

def compute_node_sir(graph: nx.Graph, id, num_test=100) -> np.float64:
    '''使用SIR模型计算节点标签,id是待计算sir值的节点id'''
    sir_list = []
    for _ in range(num_test):
        # SIR参数设置，可自行设置
        gamma, beta, step = 1, 0.01, 20  # 免疫率 感染率 SIR模型中的感染传播轮次
        # 节点的感染情况
        sir_source = [id]  # 方法输入为数组，将节点强制转换为数组，且SIR实现中使用的为节点索引号[0~n-1]，此处使用j索引号
        sir_values = SIR_network(graph, sir_source, beta, gamma, step)
        Fc = sir_values[step - 1]  # 最终的感染范围
        # 由于有概率出现节点直接免疫，传播停止的“异常”情况
        # 我们设置阈值，只统计传播覆盖范围大于1%（0.01）的情况
        if Fc > 0.01: sir_list.append(Fc)
    # 对100实验的输出结果求均值
    sir =  np.mean(sir_list) if len(sir_list) > 0 else 0
    return (id, sir)

def compute_node_label(sir_list: list, pos_ratio=0.10):
    '''根据sir的排序，取前10%的节点作为重要节点'''
    sir_list.sort(key=lambda item: item[1], reverse=True)
    ids = [id for id, sir in sir_list]
    split = int(len(sir_list) * pos_ratio)
    return ids[: split]

def parallel_worker(x):
    return compute_node_sir(*x)

def extract_enclosing_subgraphs(data: Data, num_hops=2) -> list:
    '''抽取以每个节点为中心的 n-hop 子图'''
    data_list = []
    for node_id in range(data.num_nodes):
        target_node = node_id
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
                target_node, num_hops, data.edge_index, relabel_nodes=True)
        target_node = mapping.tolist()
        # 这里可以基于子图，计算一些子图特征作为输入
        sub_graph = Data(
            x=data.x[sub_nodes], edge_index=sub_edge_index, 
            y=data.y[node_id], target_id=target_node)
        data_list.append(sub_graph)
    return data_list

def train_test_split(data_list: list, pos_ids: list) -> list:
    '''进行训练集和测试集的划分，保证二者正负例的比例为1:1'''
    num_graph = len(data_list)
    random.shuffle(pos_ids)
    # 生成正例,这是所有的正例了
    pos_list = [data_list[id] for id in pos_ids]
    # 随机抽取同样数量的负例
    neg_ids = set(range(num_graph))
    neg_ids = neg_ids - set(pos_ids)
    neg_ids = random.sample(neg_ids, len(pos_ids))
    neg_list = [data_list[id] for id in neg_ids]
    return pos_list, neg_list

class YangDataset(InMemoryDataset):

    def __init__(self, path, num_hops=2, ratio=0.7, split='train'):
        self.num_hops, self.ratio = num_hops, ratio
        self.graph, self.data = pairwise2matrix_nx(path)
        self.root = os.path.dirname(path)
        super().__init__(root=self.root)
        index = ['train', 'test'].index(split)
        self.data, self.slices = torch.load(self.processed_paths[index])

    @property
    def processed_file_names(self):
        return ['train_data.pt', 'test_data.pt']

    def speed_sir(self) -> list:
        res = None
        # 如果计算过，就不重新计算啦
        res_path = os.path.join(self.root, 'sir_list.txt')
        if not os.path.exists(res_path):
            num_nodes = self.data.num_nodes
            with mp.Pool(mp.cpu_count()) as pool:
                res = pool.map_async(parallel_worker, [(self.graph, id) for id in range(num_nodes)])
                res = res.get()
            np.savetxt(res_path, np.array(res), fmt='%d %1.32f')
        else:
            # 从.txt文件中恢复出res
            res = np.loadtxt(res_path, dtype=np.float64)
            id = res[:, 0].astype(np.int32).tolist()
            sir = res[:, 1].tolist()
            res = list(zip(id, sir))
        return res

    def process(self):
        '''
            数据集的主要处理流程
        '''
        num_nodes = self.data.num_nodes
        # 1. 计算节点特征
        self.data.x = compute_node_feature(self.graph)
        # 2. 计算节点标签
        # 为了调试方便，我们把 sir 值存到文件
        # 这里使用多进程进行加速
        res = self.speed_sir()
        y = torch.zeros((num_nodes, 1), dtype=torch.long)
        pos_ids = compute_node_label(res)
        y[pos_ids, :] = 1
        self.data.y = y
        # 3. 数据集划分并保存
        # 二分类问题中，正例和负例的比例最好相差不多
        # 本问题中，正例（关键节点）数量过少,使用 正例 : 负例 = 1 : 1
        # 由于正例很少所以
        data_list = extract_enclosing_subgraphs(self.data)
        split = int(len(pos_ids) * self.ratio)
        pos_list, neg_list = train_test_split(data_list, pos_ids)
        torch.save(self.collate(pos_list[:split] + neg_list[:split]), self.processed_paths[0])
        torch.save(self.collate(pos_list[split:] + neg_list[split:]), self.processed_paths[1])
