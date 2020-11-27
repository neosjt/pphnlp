"""
    pph
    2020.11.18
    module为模块封装的最高层
"""


import torch
import torch.nn as nn
from tqdm import tqdm
from .datahelper import pos_dict ,pos2id,id2pos,train_loader
from .config import DEVICE,DEFAULT_CONFIG,MODEL_PATH,bert_tokenizer,CLS_INDEX,PAD_INDEX
from .model import  BERT_BiLSTM_CRF
from torch.utils.data import DataLoader
from ..utils.log import logger


seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class POSModule(object):

    def __init__(self):
        self._model=None

    def train(self):
        logger.info("训练开始")

        self._model=BERT_BiLSTM_CRF()
        self._model.to(DEVICE)
        self._model.train()
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()),
                                 lr=DEFAULT_CONFIG['lr'],
                                 betas=(DEFAULT_CONFIG['beta_1'], DEFAULT_CONFIG['beta_2']),
                                 eps=DEFAULT_CONFIG['epsilon'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim,
                                                      lr_lambda=lambda x: DEFAULT_CONFIG['decay'] ** (
                                                                    x / DEFAULT_CONFIG['decay_steps']))

        for epoch in tqdm(range(DEFAULT_CONFIG['epoch'])):
            acc_loss = 0
            for ii, (input_ids,bert_mask,token_type_ids,sorted_lens,crf_mask,gt_tags) in enumerate(tqdm(train_loader)):
                optim.zero_grad()


                train_loss = self._model.loss(input_ids,bert_mask,token_type_ids,sorted_lens,crf_mask,gt_tags)
                train_loss=torch.sum(train_loss,dim=-1).to(DEVICE)
                train_loss.backward()
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self._model.parameters()), 5.0)
                optim.step()
                scheduler.step()
                acc_loss += train_loss.cpu().item()

                print('epoch{}-batch{}, acc_loss: {}'.format(epoch,ii, train_loss.cpu().item()))
            acc_loss /= len(train_loader)

            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
        self._save()
        logger.info("训练完成")

    def _save(self):
        logger.info('保存模型')
        torch.save(self._model,MODEL_PATH)

    def load(self):
        if self._model is None:
            logger.info('加载模型')
            self._model= BERT_BiLSTM_CRF()
            self._model=torch.load(MODEL_PATH)
            self._model.to(DEVICE)

    def _prepare_data(self,sentences):
        input_ids=[]
        for sent in sentences:
            word_segment = bert_tokenizer.tokenize(sent)
            word_ids = bert_tokenizer.convert_tokens_to_ids(word_segment)
            input_ids.append(word_ids)

        sent_lens = list(map(len, input_ids))
        max_len = max(sent_lens)
        input_ids = [[CLS_INDEX] + words + [PAD_INDEX] * (max_len - len(words)) for words in input_ids]

        bert_mask = []
        crf_mask = []
        for each_len in sent_lens:
            line_bert_mask = [1] * (each_len + 1) + [0] * (max_len - each_len)
            bert_mask.append(line_bert_mask)
            line_crf_mask = [1] * (each_len) + [0] * (max_len - each_len)
            crf_mask.append(line_crf_mask)

        input_ids = torch.tensor(input_ids, dtype=torch.long).to(DEVICE)
        bert_mask = torch.tensor(bert_mask, dtype=torch.long).to(DEVICE)
        crf_mask = torch.tensor(crf_mask, dtype=torch.long).to(DEVICE)
        token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long).to(DEVICE)
        sent_lens = torch.tensor(sent_lens, dtype=torch.long)
        sorted_lens, indices = torch.sort(sent_lens, descending=True)
        sorted_lens = sorted_lens.to(DEVICE)
        return  input_ids,bert_mask,token_type_ids,sorted_lens,crf_mask

    def predict(self,sentences):
        self.load()
        self._model.eval()
        input_ids, bert_mask, token_type_ids, sorted_lens, crf_mask=self._prepare_data(sentences)

        result=self._model.predict(input_ids, bert_mask, token_type_ids, sorted_lens, crf_mask)

        return result






