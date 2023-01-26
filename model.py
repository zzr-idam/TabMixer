from torch import nn
from einops.layers.torch import Reduce
from spikingjelly.activation_based import neuron, layer
import torch
from torch.utils.data import Dataset


class LDLDataset(Dataset):
    def __init__(self, X, y, train=True, ):
        self.X = X
        self.y = y
        self.train = train
        self.len = X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.train:
            return self.X[index], self.y[index]
        else:
            return self.X[index], self.y[index]


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
        # print("res",x.shape)
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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, device):
        # print('x',x.shape)
        # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        distribtuions = self.learn_D(x)
        # print("distribtuions", distribtuions.shape)
        attention_coeff = self.learn_attention_coeff(x)
        # print("attention_coeff",attention_coeff.shape)
        coeff = torch.normal(size=(x.shape[0], 1, 1), mean=attention_coeff[0].clone().detach(),
                             std=attention_coeff[1].clone().detach())
        # print('coeff',coeff.shape)
        # build feature grid
        grid = torch.zeros(x.shape[0], x.shape[1], x.shape[1])
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                grid[i, j, :] = torch.normal(size=(1, x.shape[1]), mean=distribtuions[i, 2 * j].clone().detach(),
                                             std=distribtuions[i, 2 * j + 1].clone().detach())

        grid = grid * coeff
        # print("grid",grid.shape)
        grid = grid.to(device)
        output = self.executor(grid)
        # print("output",output.shape)
        output = self.softmax(output)
        return output
