"""
    pph
    2020.11.23
    有两个配置，一个是专门针对bert_base_chinese,另外一个是其它部分的
"""
import torch
import  os
from transformers import  BertConfig
from transformers import BertTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




DEFAULT_CONFIG = {
    'lr': 0.02,
    'beta_1': 0.9,
    'beta_2': 0.9,
    'epsilon': 1e-12,
    'decay': .75,
    'decay_steps': 5000,
    'epoch': 1,
    'lr_decay': 0.05,
    'batch_size': 128,
    'dropout': 0.5,
    'embedding_dim': 300,
    'num_layers': 2,
    'bilstm_dim':128,
    'num_tags':129,
    'save_path': 'output/pos/model.pth',
    'trainset_path':'data/pos/词性标注人民日报199801.txt',
    'labelset_path':'data/pos/PFR人民日报标注语料库.txt',
    'posdict_path':'data/pos/pos_dict.json',
    'model_path':'output/pos/model.pth',
    #bert相关配置
    'bert_config_path':'data/bert-base-chinese/config.json',
    'bert_vocab_path':'data/bert-base-chinese/vocab.txt',
    'bert_pretrained_path':'data/bert-base-chinese'
}




ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAINSET_PATH=ROOT_PATH+"/"+DEFAULT_CONFIG['trainset_path']
LABELSET_PATH=ROOT_PATH+"/"+DEFAULT_CONFIG['labelset_path']
POSDICT_PATH=ROOT_PATH+"/"+DEFAULT_CONFIG['posdict_path']

MODEL_PATH=ROOT_PATH+"/"+DEFAULT_CONFIG['model_path']

#**********************************BERT*********************************#
BERT_CONFIG_PATH=ROOT_PATH+"/"+DEFAULT_CONFIG['bert_config_path']
bert_config=BertConfig.from_json_file(json_file=BERT_CONFIG_PATH)


BERT_VOCAB_PATH=ROOT_PATH+"/"+DEFAULT_CONFIG['bert_vocab_path']
bert_tokenizer=BertTokenizer(vocab_file=BERT_VOCAB_PATH)

BERT_PRETRAINED_PATH=ROOT_PATH+"/"+DEFAULT_CONFIG['bert_pretrained_path']

PAD='[PAD]'
UNK='[UNK]'
CLS='[CLS]'
SEP='[SEP]'
MASK='[MASK]'

PAD_INDEX=0
UNK_INDEX=100
CLS_INDEX=101
SEP_INDEX=102
MASK_INDEX=103

# if __name__=='__main__':
#     res=bert_tokenizer.encode("我草")
#     ids=bert_tokenizer.convert_tokens_to_ids(res)
#     print(res)
#     print(type(res))
#     print(ids)
#     print(type(ids))






