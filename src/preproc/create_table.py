# primitive
import os
import pandas as pd

# handmade libs
from const import *

# For livedoor news corpus dataset
def get_files(dir_path):
    ngs = ['LICENSE.txt']
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f not in ngs]
    return files

def get_lines(file_path):
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    return [l for l in lines if len(l) > 0]

def create_table(input_dir: str, text_column: str) -> pd.DataFrame:
    '''指定されたディレクトリ下のファイルからテーブルデータを作成する
    Args:
        input_dir (str): livedoornewscorpus 格納ディレクトリパス
        text_column (str): テキストデータとして用いる列名
    Returns:
        9 メディアのいずれかを示すラベルとテキストの 2 列を持つテーブル
    '''
    # メディア一覧をディレクトリ名から取得
    medium = [d for d in os.listdir(input_dir)
                if os.path.isdir(os.path.join(input_dir, d))]

    table = []

    for media in medium:
        dir_path = os.path.join(input_dir, media)
        files = get_files(dir_path)
        
        for file in files:
            
            file_path = os.path.join(dir_path, file)
            lines = get_lines(file_path)
            
            url, timestamp, title, text = lines[0], lines[1], lines[2], ''.join(lines[3:])
            table.append((media, url, timestamp, title, text))

    table = pd.DataFrame(table, columns=['media', 'url', 'timestamp', 'title', 'text'])
    table = table[table.index!=6031].reset_index(drop=True)

    # 必要な列のみ残す
    table = table[['media', text_column]]

    # label を数値化
    # TODO: 毎回数字が変わる。同じメディアに同じ数字が割り当てられるよう修正する。
    itos = {i: s for i, s in enumerate(set(table['media']))}
    stoi = {s: i for i, s in itos.items()}
    table.loc[:, 'media'] = table['media'].map(stoi)
    print(table.head())
    print(table.shape)

    return table

if __name__ == '__main__':
    '''main として実行時はファイルを出力する
    '''
    # Args
    input_dir = '/data/ldcc/text/'
    text_column = 'text'
    output_file = os.path.join(ROOT, 'data', f'livedoor&text={text_column}.csv')

    table = create_table(input_dir, text_column)

    # 概観 todo: logging に変更
    print(table.shape)
    print(table.head())
    print(pd.DataFrame(table.media.value_counts()).sort_index().T)

    # output
    if os.path.isfile(output_file):
        print('file exists')
    else:
        table.to_csv(output_file, index=False)
        print('file created')