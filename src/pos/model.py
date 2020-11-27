"""
    pph
    2020.11.24
    bert_bilstm_crt
    bert使用预训练的bert-base-chinese,并且不参与训练
    bilstm使用packed_sequence
    所有流程默认batch_first
"""




import torch
import torch.nn as nn
import torch.functional as F
from torchcrf import CRF

from transformers import  BertModel
from .config import BERT_PRETRAINED_PATH,bert_config,DEFAULT_CONFIG
from .config import PAD,PAD_INDEX
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BERT_BiLSTM_CRF(nn.Module):

    def __init__(self,  rnn_dim=DEFAULT_CONFIG['bilstm_dim'],num_labels=DEFAULT_CONFIG['num_tags'],num_layers=DEFAULT_CONFIG['num_layers']):
        super(BERT_BiLSTM_CRF, self).__init__()

        #配置bert不参与训练
        self.bert = BertModel.from_pretrained(BERT_PRETRAINED_PATH,config=bert_config)
        for name,param in self.bert.named_parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)


        self.bilstm = nn.LSTM(bert_config.hidden_size, rnn_dim, num_layers=num_layers, bidirectional=True,batch_first=True)
        out_dim = rnn_dim * 2

        self.hidden2tag = nn.Linear(out_dim, num_labels)
        self.crf = CRF(num_labels)


    def loss(self,input_ids,bert_mask,token_type_ids,sorted_lens,crf_mask,gt_tags):
        """
        :param input_ids:传入的是pad过的等长的batch_sentence
        :return:
        """
        #1.bert处理
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=bert_mask)
        sequence_output = outputs[0]
        sequence_output=sequence_output[:,1:]
        #print(sequence_output.shape)

        #2.bilstm
        sequence_output = pack_padded_sequence(sequence_output, sorted_lens,batch_first=True)
        sequence_output, _ = self.bilstm(sequence_output)
        sequence_output, _ = pad_packed_sequence(sequence_output,batch_first=True)
        #print(sequence_output.shape)

        #3.hidden-->tags
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        #print(sequence_output.shape)

        #4.crf_loss
        loss = -1 * self.crf(emissions, gt_tags, mask=crf_mask.byte())
        #print(loss)

        return loss

    def predict(self,input_ids,bert_mask,token_type_ids,sorted_lens,crf_mask):
        # 1.bert处理
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=bert_mask)
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, 1:]
        # print(sequence_output.shape)

        # 2.bilstm
        sequence_output = pack_padded_sequence(sequence_output, sorted_lens, batch_first=True)
        sequence_output, _ = self.bilstm(sequence_output)
        sequence_output, _ = pad_packed_sequence(sequence_output, batch_first=True)
        # print(sequence_output.shape)

        # 3.hidden-->tags
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        # print(sequence_output.shape)

        # 4.crf_decode
        result=self.crf.viterbi_decode(emissions, crf_mask.byte())
        return result

