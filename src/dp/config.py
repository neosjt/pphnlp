"""
    pph
    2020.11.13
"""
import torch
import  os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_CONFIG = {
    'lr': 2e-3,
    'beta_1': 0.9,
    'beta_2': 0.9,
    'epsilon': 1e-12,
    'decay': .75,
    'decay_steps': 5000,
    'epoch': 30,
    'lr_decay': 0.05,
    'batch_size': 256,
    'dropout': 0.5,
    'word_dim': 300,
    'embed_dropout': 0.33,
    'save_path': 'output/dp/model.pth',
    'pos_dim': 100,
    'lstm_hidden': 400,
    'lstm_layers': 3,
    'lstm_dropout': 0.33,
    'mlp_arc': 500,
    'mlp_rel': 100,
    'mlp_dropout': 0.33,
    'data_path':'data/dp/train.sample.conll',
    'config_path':'output/dp/conf.pkl'
}

ROOT = '<ROOT>'
UKN='<UKN>'
PAD='<PAD>'
WORD_PAD_INDEX=0
POS_PAD_INDEX=0
HEAD_PAD_INDEX=-1
REL_PAD_INDEX =0
ROOT_PATH= os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH=ROOT_PATH+"/"+DEFAULT_CONFIG['data_path']
MODEL_PATH=ROOT_PATH+"/"+DEFAULT_CONFIG['save_path']
CONFIG_PATH=ROOT_PATH+"/"+DEFAULT_CONFIG['config_path']


class Config(object):
    def __init__(self, word_vocab, pos_vocab, rel_vocab, **kwargs):
        super(Config, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.rel_vocab = rel_vocab
        self.pos_num = len(self.pos_vocab)
        self.rel_num = len(self.rel_vocab)
        self.word_num = len(self.word_vocab)
        self.pad_index = self.word_vocab.index(PAD)
        for name, value in kwargs.items():
            setattr(self, name, value)

