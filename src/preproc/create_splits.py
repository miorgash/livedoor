import os
from typing import Tuple
import pandas as pd
from const import *

def create_splits(input_file: str) -> Tuple:
    '''学習用・テスト用にデータセットを分割する
    Args:
        input_file (str): 入力 csv ファイル
    Returns:
        (学習用データ, テスト用データ) のタプル
    '''
    ld_df = pd.read_csv(input_file)
    test = ld_df.sample(frac=0.2, random_state=123)
    train = ld_df.drop(test.index)

    print(ld_df.shape)
    print(test.shape)
    print(train.shape)

    return train, test

if __name__ == '__main__':
    input_file = os.path.join(DIR_DATA, 'livedoor&text=text.csv')

    train, test = create_splits(input_file)
    
    for split, dataframe in zip(('train', 'test'), (train, test)):
        output_file = os.path.join(DIR_DATA, f'{split}.csv')
        if os.path.isfile(output_file):
            print(f'file exists: {output_file}')
        else:
            dataframe.to_csv(output_file, index=False)
            print(f'file created: {output_file}')