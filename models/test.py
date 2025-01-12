#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

def test_stock(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    # 用于存储所有的真实值和预测值
    all_y_true = []
    all_y_pred = []
    
    data_loader = DataLoader(datatest, batch_size=args.bs) # 将测试数据集包装到DataLoader中进行批量处理
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        predicted_values = net_g(data)
        # sum up batch loss
        test_loss += F.mse_loss(predicted_values, target, reduction='mean').item() # 预测log_probs和真实标签target之间的均方误差损失。
        # 收集真实值和预测值
        all_y_true.append(target.cpu().detach().numpy())  # 将真实值添加到列表中
        all_y_pred.append(predicted_values.cpu().detach().numpy())  # 将预测值添加到列表中

    # 合并所有批次的真实值和预测值
    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    test_loss /= len(data_loader.dataset)
    r2 = r2_score(all_y_true, all_y_pred)
    return r2, test_loss




def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs) # 将测试数据集包装到DataLoader中进行批量处理
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.mse_loss(log_probs, target, reduction='sum').item() # 预测log_probs和真实标签target之间的交叉熵损失。
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

