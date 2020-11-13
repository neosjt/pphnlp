# -*- coding: utf-8 -*-
"""
    pph
    2020.11.13
    写代码是门艺术
"""


from .dropout import SharedDropout

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence



"""
    逐LstmCell处理，基于pack_padded的输入，不支持普通的输入
    所有流程默认以Batch_First的方式处理
"""
class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1,
                 dropout=0, bidirectional=False):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for layer in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            if bidirectional:
                self.b_cells.append(nn.LSTMCell(input_size=input_size,
                                                hidden_size=hidden_size))
            input_size = hidden_size * self.num_directions

        self.reset_parameters()

    def reset_parameters(self):
        for i in self.parameters():
            # apply orthogonal_ to weight
            if len(i.shape) > 1:
                nn.init.orthogonal_(i)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(i)

    """
        输入的x是一个列表[第一个位置的输入向量，第二个位置的输入向量，...]
        batch_sie也是一个列表[第一个位置的有效个数，第二位置的有效个数,...]
        reverse代表句子反过来处理，用于bilstm的反向
    """
    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        h, c = hx
        init_h, init_c = h, c
        output, seq_len = [], len(x)
        steps = reversed(range(seq_len)) if reverse else range(seq_len)
        #在训练时不仅对输入进行dropout,我们还对lstm的隐向量进行dropout
        if self.training:
            hid_mask = SharedDropout.get_mask(h, self.dropout)

        for t in steps:
            batch_size = batch_sizes[t]
            if len(h) < batch_size:
                h = torch.cat((h, init_h[last_batch_size:batch_size]))
                c = torch.cat((c, init_c[last_batch_size:batch_size]))
            else:
                h = h[:batch_size]
                c = c[:batch_size]
            h, c = cell(input=x[t], hx=(h, c))
            output.append(h)
            if self.training:
                h = h * hid_mask[:batch_size]
            last_batch_size = batch_size
        if reverse:
            output.reverse()
        output = torch.cat(output)

        return output

    """
        传入的x为packedsequence:
        data的维度为[S*B的所有有效个数，H嵌入向量维度]
        batch_size的维度为[S]，其实就是输入数据的逐列有效长度
    """
    def forward(self, x, hx=None):
        x, batch_sizes = x.data,x.batch_sizes
        batch_size = batch_sizes[0]

        if hx is None:
            init = x.new_zeros(batch_size, self.hidden_size)
            hx = (init, init)

        #sharedropout的作用是逐句子各层使用相同的mask,依据你的长度，从最大长度的mask中抽
        #当然这是针对同一个batch里的所有句子
        #mask的使用惯例为1代表保留，0代码遮挡
        for layer in range(self.num_layers):
            if self.training:
                mask = SharedDropout.get_mask(x[:batch_size], self.dropout)
                mask = torch.cat([mask[:batch_size]
                                  for batch_size in batch_sizes])
                x *= mask
            #对batch中各个句子位置分组
            x = torch.split(x, batch_sizes.tolist())
            f_output = self.layer_forward(x=x,
                                          hx=hx,
                                          cell=self.f_cells[layer],
                                          batch_sizes=batch_sizes,
                                          reverse=False)

            if self.bidirectional:
                b_output = self.layer_forward(x=x,
                                              hx=hx,
                                              cell=self.b_cells[layer],
                                              batch_sizes=batch_sizes,
                                              reverse=True)
            if self.bidirectional:
                x = torch.cat([f_output, b_output], -1)
            else:
                x = f_output
        x = PackedSequence(x, batch_sizes)

        return x
