import sys
sys.path.append('/tmp/work/livedoor/src')
import os
import pandas as pd
import pickle5 as pickle
from const import *
# from sudachi_tokenizer import SudachiTokenizer
from mecab_tokenizer import MecabTokenizer
from datetime import datetime
from tqdm import tqdm
# from pyarrow.csv import read_csv, ReadOptions, ParseOptions, ConvertOptions
# from pyarrow import int16, string

if __name__ == '__main__':
    # tokenizer = SudachiTokenizer()
    tokenizer = MecabTokenizer()
    splits = ['train', 'test']

    for split in splits:
        print(datetime.now(), f'{split} start')
        # output
        file_o = os.path.join(DIR_BIN, f'corpus.{split}.tokenized.pkl')
        if os.path.isfile(file_o):
            print(datetime.now(), f'{split} file exists: {file_o}')
        else:

            # load
            file_i = os.path.join(DIR_DATA, f'{split}.csv')
            # convert_options = ConvertOptions(
            #     check_utf8=True,
            #     column_types={'media': int16(), 'text': string()}
            # )
            # corpus = read_csv(file_i, convert_options=convert_options)
            corpus = pd.read_csv(file_i)#.head()

            # tokenize
            corpus['text'] = [[t for t in text] for text in tqdm(tokenizer.tokenized_corpus(corpus['text']))]
            print('stop')  
            print(corpus.head())
            print(datetime.now(), f'{split} tokenizing done')

            with open(file_o, 'wb') as f:
                print(pickle.HIGHEST_PROTOCOL)
                pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
            # corpus.to_parquet(file_o)
            print(datetime.now(), f'{split} file created')
            del corpus
        print(datetime.now(), f'{split} done')