import sys
sys.path.append('/tmp/work/livedoor/src')
import os
import pickle5 as pickle
from const import *

file = os.path.join(DIR_BIN, 'corpus.train.tokenized.pkl')
with open(file, 'rb') as f:
    rows = pickle.load(f)

print(rows.head())
print(type(rows['text'][0]))
print(rows.shape)