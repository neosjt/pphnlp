""""
    sjt
    2020.11.17
    训练集默认为conll格式
    需要从训练集中提取word，pos,rel词典
    并制作训练集，测试集
"""
from src.dp.config import ROOT,UKN,PAD,DEFAULT_CONFIG,DEVICE,WORD_PAD_INDEX,POS_PAD_INDEX,HEAD_PAD_INDEX,REL_PAD_INDEX,DATA_PATH
from torch.utils.data import Dataset
from src.utils.log import logger
import torch
from src.pos.datahelper import pos_dict

class DataHelper(object):
    def __init__(self,data_path):
        self.wordvocab,self.word2id,self.id2word,\
        self.posvocab,self.pos2id,self.id2pos,\
        self.relvocab,self.rel2id,self.id2rel=self._getVocab(data_path)
        self.mydataset=self._getDataset(data_path)

    #根据data_path生成vocab
    def _getVocab(self,data_path):
        wordvocab, posvocab, relvocab = [], [], []
        with open(data_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                linestr = line.strip()
                if len(linestr) == 0:
                    continue
                sp = line.strip().split("\t")
                word = sp[1]
                pos = sp[4]
                rel = sp[7]
                wordvocab.append(word)
                posvocab.append(pos)
                relvocab.append(rel)
        wordvocab = set(wordvocab)
        wordvocab = [PAD, ROOT, UKN] + list(wordvocab)
        word2id = {word: index for index, word in enumerate(wordvocab)}
        id2word = {index: word for index, word in enumerate(wordvocab)}
        posvocab = set(posvocab)
        posvocab = [PAD,ROOT] + list(posvocab)
        pos2id = {pos: index for index, pos in enumerate(posvocab)}
        id2pos = {index: pos for index, pos in enumerate(posvocab)}
        relvocab = set(relvocab)
        relvocab = [PAD,ROOT] + list(relvocab)
        rel2id = {rel: index for index, rel in enumerate(relvocab)}
        id2rel = {index: rel for index, rel in enumerate(relvocab)}
        logger.info("字典抽取完成")
        logger.info("word字典:{}".format(wordvocab))
        logger.info("pos字典：{}".format(posvocab))
        logger.info("rel字典：{}".format(relvocab))
        return wordvocab,word2id,id2word,\
                posvocab,pos2id,id2pos,\
                relvocab,rel2id,id2rel

    #制作训练集
    def _getDataset(self,data_path):

        wordmatrix,posmatrix,headmatrix,relmatrix=[],[],[],[]
        wordlist, poslist, headlist, rellist = [], [], [], []
        with open(data_path,'r',encoding='utf-8') as fr:
            for line in fr:
                linestr=line.strip()
                if linestr=="":
                    wordlist=[ROOT]+wordlist
                    poslist=[ROOT]+poslist
                    headlist=[0]+headlist
                    rellist=[ROOT]+rellist
                    wordids=[self.word2id.get(word,self.word2id[UKN])  for word in wordlist]
                    posids=[ self.pos2id[pos]  for pos in poslist]
                    headids=[headid     for headid in headlist]
                    relids=[self.rel2id[rel]  for rel in rellist ]
                    wordmatrix.append(wordids)
                    posmatrix.append(posids)
                    headmatrix.append(headids)
                    relmatrix.append(relids)
                    wordlist, poslist, headlist, rellist = [], [], [], []
                else:
                    sp=linestr.split("\t")
                    word=sp[1]
                    pos=sp[4]
                    head=int(sp[6])
                    rel=sp[7]
                    wordlist.append(word)
                    poslist.append(pos)
                    headlist.append(head)
                    rellist.append(rel)
            if wordlist is not None:
                wordlist = [ROOT] + wordlist
                poslist = [ROOT] + poslist
                headlist = [0] + headlist
                rellist = [ROOT] + rellist
                wordids = [self.word2id.get(word, self.word2id[UKN]) for word in wordlist]
                posids = [self.pos2id[pos] for pos in poslist]
                headids = [headid for headid in headlist]
                relids = [self.rel2id[rel] for rel in rellist]
                wordmatrix.append(wordids)
                posmatrix.append(posids)
                headmatrix.append(headids)
                relmatrix.append(relids)
        mydataset=MyDataset(wordmatrix,posmatrix,headmatrix,relmatrix)
        logger.info("训练集大小为：{}".format(len(wordmatrix)))
        return mydataset

class MyDataset(Dataset):
    def __init__(self,wordmatrix,posmatrix,
                 headmatrix,relmatrix):
        self.wordmatrix=wordmatrix
        self.posmatrix=posmatrix
        self.headmatrix=headmatrix
        self.relmatrix=relmatrix
        self.len = len(wordmatrix)

    def __getitem__(self, index):
        return self.wordmatrix[index],\
               self.posmatrix[index],\
               self.headmatrix[index],\
               self.relmatrix[index]

    def __len__(self):
        return self.len

def my_collate(batch_data):
    batch_words,batch_pos,batch_heads,batch_rels=zip(*batch_data)
    max_len = max(list(map(len,batch_words)))
    batch_words=[  words+[WORD_PAD_INDEX]*(max_len-len(words))  for words in batch_words]
    batch_pos=[  pos+[POS_PAD_INDEX]*(max_len-len(pos))  for pos in batch_pos]
    batch_heads = [ heads + [HEAD_PAD_INDEX] * (max_len - len(heads)) for heads in batch_heads]
    batch_rels=  [ rels + [REL_PAD_INDEX] * (max_len - len(rels)) for rels in batch_rels]
    return torch.tensor(batch_words,dtype=torch.long).to(DEVICE),\
           torch.tensor(batch_pos,dtype=torch.long).to(DEVICE), \
           torch.tensor(batch_heads, dtype=torch.long).to(DEVICE),\
           torch.tensor(batch_rels,dtype=torch.long).to(DEVICE),\



if __name__=='__main__':
    dh=DataHelper(DATA_PATH)
    posvocab1=set(dh.posvocab)
    posvocab2=set(pos_dict.keys())
    intersect=posvocab1.intersection(posvocab2)
    print(intersect)
    print(len(intersect))
    diff= posvocab1.difference(posvocab2)
    print(diff)
    print(len(diff))








