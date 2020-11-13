"""
    pph
    2020.11.13
"""

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_CONFIG = {
    'lr': 2e-3,
    'beta_1': 0.9,
    'beta_2': 0.9,
    'epsilon': 1e-12,
    'decay': .75,
    'decay_steps': 5000,
    'epoch': 50,
    'patience': 100,
    'pad_index': 1,
    'lr_decay': 0.05,
    'batch_size': 2,
    'dropout': 0.5,
    'static': True,
    'non_static': False,
    'word_dim': 300,
    'embed_dropout': 0.33,
    'vector_path': '',
    'class_num': 0,
    'vocabulary_size': 0,
    'word_vocab': None,
    'pos_vocab': None,
    'ref_vocab': None,
    'save_path': './output',
    'pos_dim': 100,
    'lstm_hidden': 400,
    'lstm_layers': 3,
    'lstm_dropout': 0.33,
    'mlp_arc': 500,
    'mlp_rel': 100,
    'mlp_dropout': 0.33
}

ROOT = '<ROOT>'


