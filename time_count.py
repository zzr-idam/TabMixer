import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
from spikingjelly.activation_based import functional
from torchvision import transforms

from model import TabMixer
from utils import readData, train_test_spilt, setup_seed
from train import train
from test import test
import time


def count_time(test_X, test_y, model_path, device):
    names = ["cheby", "clark", "can", "kl", "cosine", "inter"]
    model = TabMixer(
        input_dims=test_X.shape[1],
        dims=512,
        depths=12,
        num_classess=test_y.shape[1]
    )
    model.load_state_dict(torch.load(model_path))
    if device != 'cpu':
        model.to(device)
    model.eval()
    Input = test_X[0:2]
    allNum = 1000
    with torch.no_grad():
        start_time = time.time()
        for i in range(allNum):
            if device != 'cpu':
                Input = Input.to(device)
            o1 = model(Input,device)
            functional.reset_net(model)
        end_time = time.time()
    d = (end_time-start_time)/allNum
    return d
