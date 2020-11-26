"""
    pph
    2020.11.18
    module为模块封装的最高层
"""


import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.pos.model  import  BERT_BiLSTM_CRF
from torch.utils.data import DataLoader
from src.pos.config import bert_tokenizer
from transformers import BertModel
from src.pos.config import BERT_PRETRAINED_PATH


seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


if __name__=='__main__':
    teststr="爸爸爱你！"
    res=bert_tokenizer.encode(teststr)
    print(res)
    res2=bert_tokenizer.convert_ids_to_tokens(res)
    print(res2)
    model=BERT_BiLSTM_CRF()
    ids=[res]

    input_ids=torch.tensor(ids,dtype=torch.long)
    bert_mask=torch.tensor([[1,1,1,1,1,1,1]],dtype=torch.long)
    token_type_ids=torch.zeros((1,7),dtype=torch.long)
    crf_mask=torch.tensor([[1,1,1,1,1]],dtype=torch.long)
    sent_lens=torch.sum(crf_mask,dim=-1)

    #排序
    sorted_lens, indices = torch.sort(sent_lens, descending=True)
    gt_labels=torch.tensor([[1,1,1,2,2]],dtype=torch.long)





    loss=model.loss(input_ids,bert_mask,token_type_ids,sorted_lens,crf_mask,gt_labels)
    print(loss)


