import pandas as pd
from preproc.create_table import create_table
from preproc.create_splits import create_splits
from const import *

if __name__ == '__main__':

    # **
    # create_table
    # *
    LDCC_DATA_DIR = '/data/ldcc/text/'
    TEXT_COLUMN = 'title'
    output_file = os.path.join(ROOT, 'data', f'livedoor&text={TEXT_COLUMN}.csv')

    table = create_table(LDCC_DATA_DIR, TEXT_COLUMN)
    print(table.shape)
    print(table.head())

    # output
    if os.path.isfile(output_file):
        print(f'file exists: {output_file}')
    else:
        table.to_csv(output_file, index=False)
        print(f'file created: {output_file}')
    
    # **
    # create_splits
    # *
    input_file = os.path.join(DIR_DATA, f'livedoor&text={TEXT_COLUMN}.csv')
    ldcc_df = pd.read_csv(input_file)
    train, test = create_splits(ldcc_df)
    
    for split, dataframe in zip(('train', 'test'), (train, test)):
        output_file = os.path.join(DIR_DATA, f'title.{split}.csv')
        if os.path.isfile(output_file):
            print(f'file exists: {output_file}')
        else:
            dataframe.to_csv(output_file, index=False)
            print(f'file created: {output_file}')