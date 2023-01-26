import os.path

from spikingjelly.activation_based import functional
from tqdm import tqdm

from model import TabMixer, LDLDataset
import torch.nn as nn
import torch
import numpy as np
from test import test
from utils import readData, train_test_spilt, getYGrid,save_csv
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from measures import KL_div


def train(train_X, train_y, test_X, test_y, device, model_Path):
    print("Train begin!")
    # param
    epochs = 150
    lr = 0.0002
    batch_size = 64
    # model param
    model = TabMixer(
        input_dims=train_X.shape[1],
        dims=512,
        depths=12,
        num_classess=train_y.shape[1]
    )

    model.to(device)
    model.train()
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # y_grid = getYGrid(train_y)

    trainDataset = LDLDataset(train_X, train_y, train=True)
    trainLoader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True, num_workers=1)
    total_batch = len(trainLoader)
    # train
    loss_all = []
    all_metric = []
    for epoch in range(epochs):
        loss_item = 0
        train_tqdm = tqdm(enumerate(trainLoader), total=total_batch, leave=True)
        model.train()
        for i, (X, y) in train_tqdm:
            # X, y = data

            X, y= X.to(device), y.to(device),
            o1 = model(X)
            # loss = torch.nn.functional.kl_div(y.log(), o1,reduction='mean')
            loss = criterion(o1, y)+torch.nn.functional.kl_div(torch.log(o1), y,reduction='batchmean')
            # print(torch.nn.functional.kl_div(torch.log(o1), y,reduction='batchmean').item())
            loss_item += loss.item()
            optimizer.zero_grad()
            loss.backward()
            # print("`")
            optimizer.step()
            functional.reset_net(model)
        print("epochs: " + str(epoch), str(loss_item / len(trainLoader)))
        loss_all.append(loss_item / len(trainLoader))
        save_path = os.path.join(model_Path, str(epoch) + '_' + str(lr) + "_v1.pth")
        torch.save(model.state_dict(), save_path)
        t= test(test_X, test_y, save_path, device).tolist()
        t.append(epoch)
        all_metric.append(t)
    # save model
    # p = "./model"
    print(loss_all)
    plt.plot(range(1, len(loss_all) + 1), loss_all)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    # Yeast_alpha
    # all_metric.append([
    #     0.0134,
    #     1,
    #     0.6812,
    #     0.0056,
    #     0.9946,
    #     0.9624,
    #     -1
    # ])
    # Human_Gene
    all_metric.append([
        0.0533,
        1,
        14.4423,
        0.2262,
        0.8345,
        0.7852,
        -1
    ])
    save_csv(all_metric,os.path.join(model_Path,'student.csv'))
    # save_path = './model/' + str(epochs - 1) + '_' + str(lr) + "_v1.pth"
    print('Model train finish!')
    return ""
