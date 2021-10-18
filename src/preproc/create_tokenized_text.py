import os
import pandas as pd
import pickle
from const import *
from sudachi_tokenizer import SudachiTokenizer
from datetime import datetime
from tqdm import tqdm

if __name__ == '__main__':
    tokenizer = SudachiTokenizer()
    splits = ['train', 'test']

    for split in splits:
        print(datetime.now(), f'{split} start')
        # output
        file_o = os.path.join(DIR_DATA, f'corpus.{split}.tokenized.pkl')
        if os.path.isfile(file_o):
            print(datetime.now(), f'{split} file exists: {file_o}')
        else:

            # load
            file_i = os.path.join(DIR_DATA, f'{split}.csv')
            corpus = pd.read_csv(file_i)

            # tokenize
            corpus['text'] = [[t for t in text] for text in tqdm(tokenizer.tokenized_corpus(corpus['text']))]
            print(corpus.head())
            print(datetime.now(), f'{split} tokenizing done')

            with open(file_o, 'wb') as f:
                pickle.dump(corpus, f)
            print(datetime.now(), f'{split} file created')
            del corpus
        print(datetime.now(), f'{split} done')