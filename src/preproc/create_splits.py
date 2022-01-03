import os
from typing import Tuple
import pandas as pd
from const import *

def create_splits(ldcc_df: pd.DataFrame) -> Tuple:
    '''学習用・テスト用にデータセットを分割する
    Args:
        input_file (pd.DataFrame): 入力データ（media, {text_column}）
    Returns:
        (学習用データ, テスト用データ) のタプル
    '''
    test = ldcc_df.sample(frac=0.2, random_state=123)
    train = ldcc_df.drop(test.index)

    print(ldcc_df.shape)
    print(test.shape)
    print(train.shape)

    return train, test

if __name__ == '__main__':
    TEXT_COLUMN = 'title'
    input_file = os.path.join(DIR_DATA, f'livedoor&text={TEXT_COLUMN}.csv')
    ldcc_df = pd.read_csv(input_file)

    train, test = create_splits(ldcc_df)
    
    for split, dataframe in zip(('train', 'test'), (train, test)):
        output_file = os.path.join(DIR_DATA, f'{TEXT_COLUMN}.{split}.csv')
        if os.path.isfile(output_file):
            print(f'file exists: {output_file}')
        else:
            dataframe.to_csv(output_file, index=False)
            print(f'file created: {output_file}')