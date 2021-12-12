# yangyang
# time:2021/11/1 21:01
import random
import networkx as nx
import numpy as np
import pandas as pd

'''
程序主要功能
输入:网络图邻接矩阵，需要被设置为感染源的节点序列，感染率，免疫率，迭代次数step
输出:被设置为感染源的节点序列的SIR感染情况---每次的迭代结果（I+R）/n
'''


def update_node_status(G: nx.Graph, node, beta, gamma):
    """
    更新节点状态
    :param G: 输入图
    :param node: 节点序数
    :param beta: 感染率
    :param gamma: 免疫率
    """
    # 如果当前节点状态为 感染者(I) 有概率gamma变为 免疫者(R)
    if G.nodes[node]['status'] == 'I':
        p = random.random()
        if p < gamma:
            G.nodes[node]['status'] = 'R'
    # 如果当前节点状态为 易感染者(S) 有概率beta变为 感染者(I)
    if G.nodes[node]['status'] == 'S':
        # 对该节点邻居节点进行遍历
        for adj_node in G[node]:
            # 邻居节点中存在 感染者(I)，该节点有概率转变为 感染者(I)
            if G.nodes[adj_node]['status'] == 'I':
                p = random.random()
                if p < beta:
                    G.nodes[node]['status'] = 'I'
                    break


def count_node(G):
    """
    计算当前图内各个状态节点的数目
    :param G: 输入图
    :return: 各个状态（S、I、R）的节点数目
    """
    s_num, i_num, r_num = 0, 0, 0
    for node in G:
        if G.nodes[node]['status'] == 'S':
            s_num += 1
        elif G.nodes[node]['status'] == 'I':
            i_num += 1
        else:
            r_num += 1
    return s_num, i_num, r_num


def SIR_network(graph, source, beta, gamma, step):
    """
    获得感染源的节点序列的SIR感染情况
    :param graph: networkx创建的网络
    :param source: 需要被设置为感染源的节点序列
    :param beta: 感染率
    :param gamma: 免疫率
    :param step: 迭代次数
    """
    n = graph.number_of_nodes()  # 网络节点个数
    sir_values = []  # 存储每一次的感染节点数
    # 初始化节点状态
    for i in range(n):
        graph.nodes[i]['status'] = 'S'  # 将所有节点的状态设置为 易感者（S）
    # 若生成图G中的node编号（从0开始）与节点Id编号（从1开始）不一致，需要减1
    for j in source:
        graph.nodes[j]['status'] = 'I'  # 将感染源序列中的节点设置为感染源，状态设置为 感染者（I）
    # 记录初始状态
    sir_values.append(len(source) / n)
    # 开始迭代感染
    for s in range(step):
        # 针对对每个节点进行状态更新以完成本次迭代
        for node in range(n):
            update_node_status(graph, node, beta, gamma)  # 针对node号节点进行SIR过程
        s, i, r = count_node(graph)  # 得到本次迭代结束后各个状态（S、I、R）的节点数目
        sir = (i + r) / n  # 本次sir值为迭代结束后 (感染节点数i+免疫节点数r)/总节点数n
        sir_values.append(sir)  # 将本次迭代的sir值加入数组
    return sir_values
