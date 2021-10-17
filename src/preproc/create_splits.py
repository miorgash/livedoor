import os
import pandas as pd
from const import *

if __name__ == '__main__':
    ld_df = pd.read_csv(os.path.join(DIR_DATA, 'livedoor&text=text.csv'))
    test = ld_df.sample(frac=0.2, random_state=123)
    train = ld_df.drop(test.index)

    print(ld_df.shape)
    print(test.shape)
    print(train.shape)

    splits = [('train', train), ('test', test)]
    for split, dataframe in splits:
        file = os.path.join(DIR_DATA, f'{split}.csv')
        if os.path.isfile(file):
            print(f'file exists: {file}')
        else:
            dataframe.to_csv(file, index=False)
            print(f'file created: {file}')