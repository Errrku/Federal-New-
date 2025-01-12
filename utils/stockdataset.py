import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockDataset(Dataset):
    def __init__(self, x_train, y_train, x_test, y_test, train=True): # 初始化StockDataset类，决定使用训练数据还是测试数据
        """
        初始化 StockDataset 类
        Args:
            x_train(np.ndarray):训练集特征.
            y_train(np.ndarray):训练集目标值。
            x_test(np.ndarray):测试集特征。
            y_test (np.ndarray):测试集目标值。
            train(bool):如果为 True,使用训练数据;否则使用测试数据。
        """
        super().__init__()
        if train:
            self.x_data = x_train
            self.y_data = y_train
        else:
            self.x_data = x_test
            self.y_data = y_test

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        返回指定索引的数据样本
        Args:
            idx(int):样本索引。
        Returns:
            tuple:（输入特征，目标值）
        """
        x = torch.tensor(self.x_data[idx],dtype=torch.float32)
        y = torch.tensor(self.y_data[idx],dtype=torch.float32)
        return x, y




