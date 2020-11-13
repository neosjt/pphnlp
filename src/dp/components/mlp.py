# -*- coding: utf-8 -*-
"""
    pph
    2020.11.13
"""


from .dropout import SharedDropout

import torch.nn as nn


"""
        MLP在这个模型的作用是给bilstm的输出降维
"""
class MLP(nn.Module):

    def __init__(self, n_in, n_hidden, dropout):
        super(MLP, self).__init__()

        self.linear = nn.Linear(n_in, n_hidden)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
