from preproc.create_table import create_table
from preproc.create_splits import create_splits
from const import *

if __name__ == '__main__':

    # **
    # create_table
    # *
    input_dir = '/data/livedoor/text/'
    text_column = 'text'
    output_file = os.path.join(ROOT, 'data', f'livedoor&text={text_column}.csv')

    table = create_table(input_dir, text_column)
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
    input_file = os.path.join(DIR_DATA, 'livedoor&text=text.csv')

    train, test = create_splits(input_file)
    
    for split, dataframe in zip(('train', 'test'), (train, test)):
        output_file = os.path.join(DIR_DATA, f'{split}.csv')
        if os.path.isfile(output_file):
            print(f'file exists: {output_file}')
        else:
            dataframe.to_csv(output_file, index=False)
            print(f'file created: {output_file}')