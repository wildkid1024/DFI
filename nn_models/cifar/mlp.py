# coding=utf-8
import torch
from torch import nn
from CIFAR.model_init_params import *


class MLP(nn.Module):
    def __init__(self, in_dim=in_dim_layer, n_hidden_1=layer_1_init, n_hidden_2=layer_2_init, n_hidden_3=layer_3_init,
                 num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.fc4 = nn.Linear(n_hidden_3, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        x = self.fc4(x)
        return x


def mlp(**kwargs):
    return MLP(**kwargs)