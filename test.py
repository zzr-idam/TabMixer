import os

import numpy as np
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TabMixer, LDLDataset
import torch.nn as nn
import torch
# from metric import score
from measures import score
from utils import readData, train_test_spilt, getYGrid, setup_seed
import numpy


def test(test_X, test_y, model_path, device):
    names = ["cheby", "clark", "can", "kl", "cosine", "inter"]
    model = TabMixer(
        input_dims=test_X.shape[1],
        dims=512,
        depths=12,
        num_classess=test_y.shape[1]
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    testDataset = LDLDataset(test_X, test_y, train=False)
    testLoader = DataLoader(dataset=testDataset, batch_size=32, shuffle=True, num_workers=1)
    total_batch = len(testLoader)
    all = [0, 0, 0, 0, 0, 0]
    with torch.no_grad():
        test_tqdm = tqdm(enumerate(testLoader), total=total_batch, leave=True)
        for i, (X,y) in test_tqdm:
            # X, y = data
            X, y = X.to(device), y.to(device)
            o1 = model(X)
            temp = score(y.cpu().detach().numpy(),o1.cpu().detach().numpy())
            for i in range(6):
                all[i] += temp[i]
            # print(temp)
            functional.reset_net(model)
    all = numpy.array(all) / len(testLoader)
    print("Metric: ")
    for i in range(len(all)):
        print(names[i] + ': ' + str(all[i]))
    print('Test end!')
    return all


if __name__ == '__main__':
    setup_seed(0)
    print(os.getcwd())
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    X, y = readData('Human_Gene.mat')
    X = X
    y = y
    train_X, test_X, train_y, test_y = train_test_spilt(X, y, 0.7)
    path = "./model/2_0.0002_v2.pth"
    test(test_X, test_y, path, device)

