import math
import random
from turtle import ycor

import numpy as np
import torch
import scipy.io as scio
import torch.nn as nn
import pandas as pd
def train_test_spilt(X, y, p=0.7):
    len = X.shape[0]
    train_len = int(len * p)
    train_data = X[:train_len, :]
    test_data = X[train_len:, :]
    train_label = y[:train_len, :]
    test_label = y[train_len:, :]
    print("Spilt finish!")
    return [train_data, test_data, train_label, test_label]


def readData(dataPath):
    dataFile = "./data/"+dataPath
    data = scio.loadmat(dataFile)
    # 读取特征和标记分布矩阵
    X = data['features']
    y = data['labels']
    X = torch.tensor(X).float()
    y = torch.tensor(y).float()
    print("Read finish!")
    return [X, y]


def getYGrid(y):
    # print("train",y.shape)
    y_matrix = torch.zeros(y.shape[0], y.shape[1], y.shape[1])
    for i in range(1, y.shape[1] - 1):
        for j in range(1, y.shape[1] - 1):
            y_matrix[:, i - 1, j - 1] = y[:, j - 1] - y[:, i - 1]

    y_grid = torch.zeros((y_matrix.shape[0], y_matrix.shape[1], y_matrix.shape[1], y_matrix.shape[1]))
    for i in range(y_matrix.shape[0]):
        for j in range(y_matrix.shape[1]):
            for k in range(y_matrix.shape[1]):
                y_grid[i, :, :, k] = torch.normal(y_matrix[i, j, k], 0.5, size=(1, y_matrix.shape[1]))
    # 归一化
    y_grid = nn.Tanh()(y_grid)
    # print('end getGRid')
    return y_grid


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_csv(lst,path):
    df = pd.DataFrame(lst)
    col = ['cheby', 'clark', 'can', 'kl', 'cosine', 'inter',"EPOCH"]
    df.to_csv(path,
              header=col
              )
    csv2excel(path)

def csv2excel(path):
    t = pd.read_csv(path)
    t.to_excel(path[:-3]+'xlsx')
if __name__ == '__main__':
    X,y = readData("Human_Gene.mat")
    print(X.shape,y.shape)