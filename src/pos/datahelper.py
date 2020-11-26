"""
    pph
    2020.11.26
    所有的数据预处理工作都在这里完成
"""

import codecs
from src.pos.config import TRAINSET_PATH,LABELSET_PATH,POSDICT_PATH,bert_tokenizer
from collections  import  defaultdict
import json
from torch.utils.data import Dataset


#将PFR人民日报标注语料库.txt------>pos_dict.json，
#这一步并不是必须的，只不过是我想将字典存成Json格式而已
def generatePosDict(input_path,output_path):
    pos_dict=defaultdict(str)
    with open(input_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            linestr=line.strip()
            if linestr!="":
                sp=linestr.split()
                pos_symbol=sp[0]
                pos_desc=sp[1]
                pos_dict[pos_symbol]=pos_desc
    with codecs.open(output_path, "w", encoding="utf-8") as f:
        json.dump(pos_dict, f, ensure_ascii=False, indent=4)

#加载pos_dict.json
def loadPosDict(path):
    with codecs.open(path, "r", encoding="utf-8-sig") as f:
        pos_dict=json.load(f)
    return pos_dict


#pos_dict--->pos2id,id2pos
def genPOSIDDict(pos_dict:dict):
    pos_vocab=pos_dict.keys()
    new_pos_vocab=[]
    for pos in pos_vocab:
        new_pos_vocab.append('B-'+pos)
        new_pos_vocab.append('I-'+pos)
        new_pos_vocab.append('S-' + pos)
    pos2id={pos:index for index,pos in enumerate(new_pos_vocab)}
    id2pos={index:pos for index,pos in enumerate(new_pos_vocab)}
    return pos2id,id2pos

#得到训练集的posvocab，发现汉语规范的记性标注集比它多个'x'
def getPOSVocab(path):
    pos_vocab=set()
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr:
            linestr=line.strip()
            if linestr!="":
                sp=linestr.split()
                for word_pos in sp:
                    pos=word_pos.split('/')[1]
                    if pos.count(']'):
                        pos=pos.split(']')[0]
                    pos_vocab.add(pos)
    return pos_vocab


#制作data_loader
class MyDataset(Dataset):
    def __init__(self,trainset_path):
        with open(trainset_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line_tags=[]
                line_words=[]
                linestr = line.strip()
                if linestr != "":
                    sp = linestr.split()
                    for word_pos in sp:
                        ssp= word_pos.split('/')
                        assert (len(ssp)==2)
                        word,pos =ssp
                        if pos.count(']'):
                            pos = pos.split(']')[0]


                        word_segment=bert_tokenizer.tokenize(word)
                        word_ids=bert_tokenizer.convert_tokens_to_ids(word_segment)
                        word_len=len(word_segment)
                        if word_len>1:
                            line_tags+=["B-"+pos]+["I-"+pos]*(word_len-1)
                        else:
                            line_tags+=["S-"+pos]
                        line_words+
                    assert(len(line_words_str)==len(line_tags))
                    print(line_words_str)
                    print(line_tags)


    def __getitem__(self, index):
        return self.wordmatrix[index],\
               self.posmatrix[index],\
               self.headmatrix[index],\
               self.relmatrix[index]

    def __len__(self):
        return self.len

    # def my_collate(batch_data):
    #     batch_words,batch_pos,batch_heads,batch_rels=zip(*batch_data)
    #     max_len = max(list(map(len,batch_words)))
    #     batch_words=[  words+[WORD_PAD_INDEX]*(max_len-len(words))  for words in batch_words]
    #     batch_pos=[  pos+[POS_PAD_INDEX]*(max_len-len(pos))  for pos in batch_pos]
    #     batch_heads = [ heads + [HEAD_PAD_INDEX] * (max_len - len(heads)) for heads in batch_heads]
    #     batch_rels=  [ rels + [REL_PAD_INDEX] * (max_len - len(rels)) for rels in batch_rels]
    #     return torch.tensor(batch_words,dtype=torch.long).to(DEVICE),\
    #            torch.tensor(batch_pos,dtype=torch.long).to(DEVICE), \
    #            torch.tensor(batch_heads, dtype=torch.long).to(DEVICE),\
    #            torch.tensor(batch_rels,dtype=torch.long).to(DEVICE),\








#预处理数据
pos_dict=loadPosDict(POSDICT_PATH)
pos2id,id2pos=genPOSIDDict(pos_dict)




if __name__=='__main__':
    # pos_vocab2=getPOSVocab2(LABELSET_PATH)
    # pos_vocab=getPOSVocab(TRAINSET_PATH)
    # pos_vocab2=set(pos_dict.keys())
    # dif=pos_vocab2.difference(pos_vocab)
    # print(pos_vocab2)
    # print(len(pos_vocab2))
    # print(pos_vocab)
    # print(len(pos_vocab))
    # print(dif)
    # pos_dict=loadPosDict(POSDICT_PATH)
    # print(pos_dict)
    #generatePosDict(LABELSET_PATH,POSDICT_PATH)
    # print(pos2id)
    dataset=MyDataset(TRAINSET_PATH)
