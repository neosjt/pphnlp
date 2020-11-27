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
from src.pos.module import POSModule



if __name__=='__main__':
    module=POSModule()
    # module.train()
    res=module.predict(['爸爸爱你'])
    print(res)
