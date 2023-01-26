from torch import nn
from einops.layers.torch import Reduce
from spikingjelly.activation_based import neuron , layer
import torch


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = layer.nn.LayerNorm(dim)
        self.mask = nn.Sequential(
            layer.nn.Linear(dim, dim),
            neuron.IFNode(),
            layer.nn.GELU(),
            layer.nn.Linear(dim, dim),
            neuron.IFNode(),
            layer.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mask(x) * self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=layer.nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        neuron.IFNode(),
        layer.nn.GELU(),
        layer.nn.Dropout(dropout),
        dense(inner_dim, dim),
        neuron.IFNode(),
        layer.nn.Dropout(dropout)
    )


def MLPMixer(*, input_dim, dim, depth, num_classes, expansion_factor=4, expansion_factor_token=0.5, dropout=0.):
    return nn.Sequential(
        nn.Linear(input_dim, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout))
        ) for _ in range(depth)],
        layer.nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )


class TabMixer(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, input_dims, dims, depths, num_classess):
        super().__init__()
        self.executor = MLPMixer(
            input_dim=input_dims,
            dim=dims,
            depth=depths,
            num_classes=num_classess)
        self.learn_D = nn.Sequential(
            layer.nn.Linear(input_dims, dims),
            neuron.IFNode(),
            layer.nn.ReLU(),
            layer.nn.Linear(dims, dims),
            neuron.IFNode(),
            layer.nn.ReLU(),
            nn.Linear(dims, 2 * input_dims),
            layer.nn.Sigmoid()
        )

        self.learn_attention_coeff = nn.Sequential(
            layer.nn.Linear(input_dims, dims),
            neuron.IFNode(),
            layer.nn.ReLU(),
            layer.nn.Linear(dims, dims),
            neuron.IFNode(),
            layer.nn.ReLU(),
            nn.Linear(dims, 2),
            Reduce('n c -> c', 'mean'),
            layer.nn.Sigmoid()
        )

    def forward(self, x):
        distribtuions = self.learn_D(x)

        attention_coeff = self.learn_attention_coeff(x)

        coeff = torch.normal(size=(x.shape[0], 1, 1), mean=attention_coeff[0].clone().detach(),
                             std=attention_coeff[1].clone().detach())

        # build feature grid
        grid = torch.zeros(x.shape[0], x.shape[1], x.shape[1])
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                grid[i, j, :] = torch.normal(size=(1, x.shape[1]), mean=distribtuions[i, 2 * j].clone().detach(),
                                             std=distribtuions[i, 2 * j + 1].clone().detach())

        grid = grid * coeff

        output = self.executor(grid)

        return output


model = TabMixer(
    input_dims=36,
    dims=512,
    depths=12,
    num_classess=68
)

import scipy.io as scio

# 读取文件内容
dataFile = './data/Human_Gene.mat'
data = scio.loadmat(dataFile)

# 读取特征和标记分布矩阵
X = data['features'][0:2, :]
y = data['labels']

X = torch.tensor(X).float()

grid = model(X)

print(grid.shape)