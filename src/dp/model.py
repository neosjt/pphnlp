"""
    sjt
    2020.11.16
    主模型文件，分别以bilstm_mlp_biaffine和
                    bert_bilstm_mlp_biaffine为例
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .components.dropout import IndependentDropout, SharedDropout
from .components import LSTM, MLP, Biaffine


#模型1：原生的bilstm_mlp_biaffine
class BiaffineParser(nn.Module):
    def __init__(self, args):
        super(BiaffineParser, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden
        self.batch_size = args.batch_size
        self.bidirectional = True
        self.lstm_layters = args.lstm_layers
        self.dropout = args.dropout
        self.save_path = args.save_path

        vocabulary_size = args.word_num
        word_dim = args.word_dim
        pos_num = args.pos_num
        pos_dim = args.pos_dim

        # embedding层
        self.word_embedding = nn.Embedding(vocabulary_size, word_dim)
        self.pos_embedding = nn.Embedding(pos_num, pos_dim)
        self.embed_dropout = IndependentDropout(p=args.embed_dropout)




        # bilstm layer
        self.lstm = LSTM(word_dim + pos_dim, self.hidden_dim, bidirectional=self.bidirectional,
                            num_layers=self.lstm_layters, dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

        # MLP层
        self.mlp_arc_h = MLP(n_in=args.lstm_hidden*2, n_hidden=args.mlp_arc, dropout=args.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=args.lstm_hidden*2, n_hidden=args.mlp_arc, dropout=args.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=args.lstm_hidden*2, n_hidden=args.mlp_rel, dropout=args.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=args.lstm_hidden*2, n_hidden=args.mlp_rel, dropout=args.mlp_dropout)

        # Biaffine层
        self.arc_attn = Biaffine(n_in=args.mlp_arc, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=args.mlp_rel, n_out=args.rel_num, bias_x=True, bias_y=True)

        #self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.zeros_(self.word_embedding.weight)
    #
    # def init_hidden(self, batch_size=None):
    #     if batch_size is None:
    #         batch_size = self.batch_size
    #
    #     h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)
    #     c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)
    #
    #     return h0, c0

    def forward(self, words, tags):
        # 得到每batch数据的mask和lens
        mask = words.ne(self.args.pad_index)
        lens = mask.sum(dim=1)

        #1.embedding处理流程
        word_emb=self.word_embedding(words)
        pos_emb = self.pos_embedding(tags)
        word_emb, pos_emb = self.embed_dropout(word_emb, pos_emb)
        x = torch.cat((word_emb, pos_emb), dim=-1)

        #2.bilstm处理流程
        #1).按句长降序排列
        sorted_lens, indices = torch.sort(lens, descending=True)
        #2).还原句子排列
        inverse_indices = indices.argsort()
        x = pack_padded_sequence(x[indices], sorted_lens, True)
        x = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.lstm_dropout(x)[inverse_indices]


        # 3.MLP处理
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # 4.Biaffine处理流程
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_((mask.ne(1)).unsqueeze(1), float('-inf'))


        return s_arc, s_rel
