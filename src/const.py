import os
import torch
ROOT = '../../'
DIR_DATA = os.path.join(ROOT, 'data')
DIR_MODEL = os.path.join(ROOT, 'model')
DIR_LOG = os.path.join(ROOT, 'log')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKENIZER = 'mecab' # 'mecab' or 'sudachi'
# !echo $(mecab-config --dicdir)/mecab-ipadic-neologd の出力結果
DIR_MECAB_DIC = '/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd'