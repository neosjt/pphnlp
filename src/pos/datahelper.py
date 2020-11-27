"""
    pph
    2020.11.26
    所有的数据预处理工作都在这里完成
"""

import codecs
from src.pos.config import TRAINSET_PATH,LABELSET_PATH,POSDICT_PATH,bert_tokenizer,CLS_INDEX,SEP_INDEX,PAD_INDEX,DEVICE,DEFAULT_CONFIG
from collections  import  defaultdict
import json
from torch.utils.data import Dataset,DataLoader
import torch


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
        self.sentence_matrix=[]
        self.tag_matrix=[]
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
                        line_words.extend(word_ids)
                    assert(len(line_words)==len(line_tags))
                    line_tags=[pos2id[pos] for pos in line_tags]
                    self.sentence_matrix.append(line_words)
                    self.tag_matrix.append(line_tags)
        self.len=len(self.sentence_matrix)

    def __getitem__(self, index):
        return self.sentence_matrix[index],\
               self.tag_matrix[index],\


    def __len__(self):
        return self.len

def my_collate(batch_data):
    batch_sentences,batch_tags=zip(*batch_data)
    sent_lens=list(map(len,batch_sentences))
    max_len = max(sent_lens)
    input_ids=[ [CLS_INDEX]+ words +[PAD_INDEX]*(max_len-len(words)) for words in batch_sentences]
    gt_tags=[  pos+[-1]*(max_len-len(pos))  for pos in batch_tags]
    bert_mask=[]
    crf_mask=[]
    for each_len in sent_lens:
        line_bert_mask=[1]*(each_len+1)+[0]*(max_len-each_len)
        bert_mask.append(line_bert_mask)
        line_crf_mask=[1]*(each_len)+[0]*(max_len-each_len)
        crf_mask.append(line_crf_mask)

    input_ids=torch.tensor(input_ids, dtype=torch.long).to(DEVICE)
    gt_tags=torch.tensor(gt_tags, dtype=torch.long).to(DEVICE)
    bert_mask=torch.tensor(bert_mask, dtype=torch.long).to(DEVICE)
    crf_mask=torch.tensor(crf_mask, dtype=torch.long).to(DEVICE)
    token_type_ids=torch.zeros(input_ids.shape,dtype=torch.long).to(DEVICE)
    sent_lens=torch.tensor(sent_lens,dtype=torch.long)
    sorted_lens, indices = torch.sort(sent_lens, descending=True)
    sorted_lens=sorted_lens.to(DEVICE)
    return input_ids,bert_mask,token_type_ids,sorted_lens,crf_mask,gt_tags


def getDataLoader(path):
    mydataset=MyDataset(path)
    mydataloader = DataLoader(dataset=mydataset,
                              batch_size=DEFAULT_CONFIG['batch_size'],
                              shuffle=True,
                              collate_fn=my_collate)
    return mydataloader





#预处理数据
pos_dict=loadPosDict(POSDICT_PATH)
pos2id,id2pos=genPOSIDDict(pos_dict)
train_loader=getDataLoader(TRAINSET_PATH)




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
    #dataset=MyDataset(TRAINSET_PATH)

    # for ii, (input_ids,bert_mask,token_type_ids,sorted_lens,crf_mask,gt_tags) in enumerate(train_loader):
    #     print("#####################one_batch################")
    #     print(input_ids.shape)
    #     print(input_ids)
    #     print(bert_mask.shape)
    #     print(bert_mask)
    #     print(token_type_ids.shape)
    #     print(token_type_ids)
    #     print(sorted_lens.shape)
    #     print(sorted_lens)
    #     print(crf_mask.shape)
    #     print(crf_mask)
    #     print(gt_tags.shape)
    #     print(gt_tags)
    print(len(pos2id))
    print(pos2id)
