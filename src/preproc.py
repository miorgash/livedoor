from preproc.create_table import create_table
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
        print('file exists')
    else:
        table.to_csv(output_file, index=False)
        print('file created')