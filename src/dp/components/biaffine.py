# -*- coding: utf-8 -*-
"""
    pph
    2020.11.12
"""

import torch
import torch.nn as nn

"""
  双仿射函数，这里用于：
        1.抽取序列中每个词是当前词的中心词的概率，此时的输出类别为1
        2.在1的基础上继续抽取每条依赖弧的类别，此时的输出类别为关系个数。
        双仿射，即两次仿射变换，H_d*C*H_h,因为是使用bilstm的所有隐向量进行计算，
        所以可以看作某种注意力机制。
"""
class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        info = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            info += f", bias_x={self.bias_x}"
        if self.bias_y:
            info += f", bias_y={self.bias_y}"

        return info

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat([x, x.new_ones(x.shape[:-1]).unsqueeze(-1)], -1)
        if self.bias_y:
            y = torch.cat([y, y.new_ones(y.shape[:-1]).unsqueeze(-1)], -1)
        # [batch_size, 1, seq_len, d]
        x = x.unsqueeze(1)
        # [batch_size, 1, seq_len, d]
        y = y.unsqueeze(1)
        # [batch_size, n_out, seq_len, seq_len]
        #坑爹，高维矩阵相乘在python中其实就是其它维广播到相同长度，在最后两维做矩阵乘法
        s = x @ self.weight @ torch.transpose(y, -1, -2)
        # remove dim 1 if n_out == 1
        #pytorch中squeeze只有维度为1才会去掉
        s = s.squeeze(1)

        return s
