import os
import torch
SEED = 123
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DIR_BIN = os.path.join(ROOT, 'bin')
DIR_DATA = os.path.join(ROOT, 'data')
DIR_MODEL = os.path.join(ROOT, 'model')
DIR_LOG = os.path.join(ROOT, 'log')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# !echo $(mecab-config --dicdir)/mecab-ipadic-neologd の出力結果
DIR_MECAB_DIC = '/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd'