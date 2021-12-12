#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   YangMain.py
@Time    :   2021/12/11 19:42:20
@Author  :   LiFeng
@Contact :   2807229316@qq.com
@Desc    :   None
'''

import os
import argparse
import torch

from YangModel import GAT
from YangDataset import YangDataset
from torch_geometric.data import DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

parser = argparse.ArgumentParser(description="测试参数集")

parser.add_argument('-data_name', type=str, default='BUP', help='数据集名称')
parser.add_argument('-num_hop', type=int, default='2', help='子图抽取的跳数')

args = parser.parse_args()

# 获取命令行参数
data_name = args.data_name
num_hop = args.num_hop
batch_size = 5
num_epoch = 15


file_dir = os.path.dirname(os.path.realpath(__file__))
root = os.path.join(file_dir, f'yang_data/{data_name}')
file_path = os.path.join(root, f'{data_name}.txt')

# 加载数据集
train_dataset = YangDataset(file_path, split='train')
test_dataset = YangDataset(file_path, split='test')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(train_dataset.num_features, hidden_size=8).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005)
criterion = CrossEntropyLoss()

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(train_dataset)

@torch.no_grad()
def test():
    model.eval()

    total_loss, y_pred, y_true = 0, [], []
    label_pred, label_true = [], []
    for data in test_loader:
        y_true.append(data.y.to(torch.float))
        label_true.append(data.y)

        data = data.to(device)
        out = model(data)

        y_pred.append(torch.softmax(out, dim=1)[:, 1].cpu())
        label_pred.append(out.argmax(dim=1))

        loss = criterion(out, data.y)
        total_loss += float(loss) * data.num_graphs

    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    label_true, label_pred = torch.cat(label_true).tolist(), torch.cat(label_pred).tolist()
    return (
        total_loss / len(test_dataset), 
        roc_auc_score(y_true, y_pred),
        average_precision_score(y_true, y_pred),
        accuracy_score(label_true, label_pred))
        

if __name__ == '__main__':
    for epoch in range(num_epoch):
        train_loss = train()
        test_loss, test_auc, test_ap, test_acc = test()
        print(f'epoch: {epoch}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}, test_auc: {test_auc:.4f}, test_ap: {test_ap:.4f}, test_acc: {test_acc:.4f}')
