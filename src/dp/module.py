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
from .datatool  import  DataHelper,my_collate
from .config import DEVICE ,DATA_PATH,Config,MODEL_PATH,CONFIG_PATH,ROOT,UKN,PAD
from .model import  BiaffineParser
from torch.utils.data import DataLoader
from ..utils.log import logger


seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class DPModule(object):

    def __init__(self):

        self._config = None

        self._model=None

    def train(self):
        logger.info("训练开始")
        datahelper = DataHelper(DATA_PATH)
        self._config=Config(datahelper.wordvocab,
                             datahelper.posvocab, datahelper.relvocab)
        train_dataset = datahelper.mydataset
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self._config.batch_size,
                                  shuffle=True,
                                  collate_fn=my_collate)
        self._model=BiaffineParser(self._config)
        self._model.to(DEVICE)
        self._model.train()
        optim = torch.optim.Adam(self._model.parameters(), lr=self._config.lr,
                                 betas=(self._config.beta_1, self._config.beta_2),
                                 eps=self._config.epsilon)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim,
                                                      lr_lambda=lambda x: self._config.decay ** (
                                                                    x / self._config.decay_steps))

        for epoch in tqdm(range(self._config.epoch)):
            acc_loss = 0
            for ii, (words,pos,heads,rels) in enumerate(train_loader):
                optim.zero_grad()
                mask = words.ne(self._config.pad_index)
                mask[:, 0] = 0
                s_arc, s_rel = self._model(words, pos)
                s_arc, s_rel = s_arc[mask], s_rel[mask]
                gold_arcs, gold_rels = heads[mask], rels[mask]


                item_loss = self._get_loss(s_arc, s_rel, gold_arcs, gold_rels)
                acc_loss += item_loss.cpu().item()
                item_loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                optim.step()
                scheduler.step()
#                print('epoch{}-batch{}, acc_loss: {}'.format(epoch,ii, item_loss.cpu().item()))
            acc_loss /= len(train_loader)
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))
        self._save()
        logger.info("训练完成")

    def _save(self):
        logger.info('保存模型')
        with open(CONFIG_PATH, 'wb') as fw:
            pickle.dump(self._config, fw)
        torch.save(self._model,MODEL_PATH)

    def load(self ):
        if self._config is None or self._model is None:
            logger.info('加载模型')
            with open(CONFIG_PATH, 'rb') as fr:
                self._config = pickle.load(fr)

            self._model= BiaffineParser(self._config)
            self._model=torch.load(MODEL_PATH)
            self._model.to(DEVICE)

            self._word2id = {word: index for index, word in enumerate(self._config.word_vocab)}
            self._id2word = {index: word for index, word in enumerate(self._config.word_vocab)}
            self._pos2id = {pos: index for index, pos in enumerate(self._config.pos_vocab)}
            self._id2pos = {index: pos for index, pos in enumerate(self._config.pos_vocab)}
            self._rel2id = {rel: index for index, rel in enumerate(self._config.rel_vocab)}
            self._id2rel = {index: rel for index, rel in enumerate(self._config.rel_vocab)}


    def predict(self, word_list, pos_list):
        self._model.eval()
        assert len(word_list) == len(pos_list)
        word_list.insert(0, ROOT)
        pos_list.insert(0, ROOT)
        wordids = [self._word2id.get(word, self._word2id[UKN]) for word in word_list]
        posids = [self._pos2id[pos] for pos in pos_list]
        wordids=torch.tensor([wordids],dtype=torch.long).to(DEVICE)
        posids=torch.tensor([posids],dtype=torch.long).to(DEVICE)
        mask = wordids.ne(self._word2id[PAD])
        s_arc, s_rel = self._model(wordids, posids)
        s_arc, s_rel = s_arc[mask], s_rel[mask]

        pred_arcs, pred_rels = self._decode(s_arc, s_rel)
        pred_arcs = pred_arcs.cpu().tolist()
        pred_rels = pred_rels.cpu().tolist()
        pred_arcs[0] = 0
        pred_rels = [self._id2rel[rel] for rel in pred_rels]
        pred_rels[0] = ROOT
        return pred_arcs, pred_rels

    def _get_loss(self, s_arc, s_rel, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]
        arc_loss = F.cross_entropy(s_arc, gold_arcs)
        rel_loss = F.cross_entropy(s_rel, gold_rels)
        loss = arc_loss + rel_loss
        return loss

    def _decode(self, s_arc, s_rel):
        pred_arcs = s_arc.argmax(dim=-1)
        pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)
        return pred_arcs, pred_rels




