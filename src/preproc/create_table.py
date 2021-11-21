# primitive
import sys
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

if __name__ == '__main__':
    # dataset
    p = '/data/livedoor/text/'
    medium = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]

    dataset = []

    for media in medium:
        dir_path = os.path.join(p, media)
        files = get_files(dir_path)
        
        for file in files:
            
            file_path = os.path.join(dir_path, file)
            lines = get_lines(file_path)
            
            url, timestamp, title, text = lines[0], lines[1], lines[2], ''.join(lines[3:])
            dataset.append((media, url, timestamp, title, text))

    dataset = pd.DataFrame(dataset, columns=['media', 'url', 'timestamp', 'title', 'text'])
    dataset = dataset[dataset.index!=6031].reset_index(drop=True)

    # for prediction
    TEXT_COL = 'text'
    dataset = dataset[['media', TEXT_COL]]

    # label を数値化
    itos = {i: s for i, s in enumerate(set(dataset['media']))}
    stoi = {s: i for i, s in itos.items()}
    dataset.loc[:, 'media'] = dataset['media'].map(stoi)

    # 概観
    print(dataset.shape)
    print(dataset.head())
    print(pd.DataFrame(dataset.media.value_counts()).sort_index().T)

    # output
    f = os.path.join(ROOT, 'data', f'livedoor&text={TEXT_COL}.csv')
    if not os.path.isfile(f):
        dataset.to_csv(f, index=False)
        print('file created')
    else:
        print('file exists')