import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms

from time_count import count_time
from utils import readData, train_test_spilt, setup_seed
from train import train
from test import test

if __name__ == '__main__':
    setup_seed(0)
    # print(os.getcwd())
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # print(device)
    dataSetName = 'Human_Gene'
    X, y = readData(dataSetName + '.mat')
    dataModelPath = os.path.join("./model", dataSetName)
    if not os.path.exists(dataModelPath):
        os.makedirs(dataModelPath)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=5)
    model_path = "model/Human_Gene/0_0.0001_v1.pth"
    gpu_total_time = count_time(test_X, test_y, model_path, device)
    device = "cpu"
    cpu_total_time = count_time(test_X, test_y, model_path, device)
    print("gpu", gpu_total_time)
    print("cpu", cpu_total_time)
    # print(train_X.shape,test_X.shape)
    # temp = np.ones((test_y.shape[0],68))/68
    # print(temp)
    # print(temp.shape)
    # op = test_y.cpu().detach().numpy()
    # print(op.shape)
    # print(score(temp,op))
    # print(train_X.shape,train_y.shape,test_X.shape,test_y.shape)

    # path = train(train_X,train_y,test_X,test_y,device,dataModelPath)

    # path = "./model/2_0.0002_v4.pth"
    # test(test_X,test_y,path,device)
