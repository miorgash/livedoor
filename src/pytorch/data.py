import pandas as pd
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer
from sudachipy import dictionary
from collections import Counter
import gensim
from tqdm import tqdm
import torchtext
import numpy as np
import os
import torch

class LivedoorDataset(Dataset):
    def __init__(self):
        '''
        input:
            dataframe: pandas.DataFrame of (label, text) shape.
        output:
            None
        '''
        INPUT_DIR    = "/data/ldcc/text/"
        LABEL_COLUMN = "media"
        TEXT_COLUMN  = "title"
        # TODO: text_column の内容が利用者から隠蔽されていてよくない；全項目返すよう変更？
        table = self.__create_table(input_dir=INPUT_DIR, text_column=TEXT_COLUMN)
        self.label = torch.tensor(table[LABEL_COLUMN].tolist())
        self.text = table[TEXT_COLUMN].tolist()
    def __len__(self):
        return len(self.label)
    def __getitem__(self):
        raise NotImplementedError
    def __get_files(self, dir_path):
        ngs = ['LICENSE.txt']
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f not in ngs]
        return files
    def __get_lines(self, file_path):
        with open(file_path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
        return [l for l in lines if len(l) > 0]
    def __create_table(self, input_dir: str, text_column: str) -> pd.DataFrame:
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
            files = self.__get_files(dir_path)
            
            for file in files:
                file_path = os.path.join(dir_path, file)
                lines = self.__get_lines(file_path)
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

class LstmLivedoorDataset(LivedoorDataset):
    def __init__(self):
        super().__init__()
        INPUT_PRETRAINED_VECTORS = "/data/chive_v1.2mc90/chive-1.2-mc90_gensim/chive-1.2-mc90.kv"
        pretrained_vectors = gensim.models.KeyedVectors.load(INPUT_PRETRAINED_VECTORS)
        # vectors を用いたフィルタリング
        words_found_in_pretrained_vectors = set([t for t in pretrained_vectors.vocab.keys()])
        # 全テキストのうち、vector に含まれる語のみ残し、vocab とする
        self.tokenizer = dictionary.Dictionary().create()
        # tokenize
        token_sequences = []
        for instance in tqdm(self.text):
            token_sequence = [token.surface() for token in self.tokenizer.tokenize(instance)]
            token_sequences.append(token_sequence)
        # 学習用コーパスと学習済み分散表現に含まれる語の Vocab を作成する。
        counter = Counter()
        for token_sequence in tqdm(token_sequences):
            counter.update(words_found_in_pretrained_vectors & set(token_sequence))
        self.vocab = torchtext.vocab.vocab(counter)
        self.vocab.insert_token("<pad>", 0)
        self.vocab.insert_token("<unk>", 1)
        self.vocab.set_default_index(1)
        # token 列を id 列に変換
        self.id_sequences = []
        for token_sequence in token_sequences:
            self.id_sequences.append(torch.tensor([self.vocab[token] for token in token_sequence]))
        # 学習用 vectors を作成する。軽量化のため学習用データ語彙との積集合のみ使用。
        vectors_for_unk_and_pad = np.zeros((2, 300))
        itos = self.vocab.get_itos()
        words = [itos[i] for i in range(len(self.vocab))]
        self.vectors = np.concatenate((vectors_for_unk_and_pad,
                            np.array([pretrained_vectors[w] for w in words[2:]])), axis=0)
    def __len__(self):
        return super().__len__()
    def __getitem__(self, i):
        return (self.label[i], self.id_sequences[i])

class BertLivedoorDataset(LivedoorDataset):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v2")
        inputs = self.tokenizer(self.text, return_tensors="pt", 
                           padding=True, truncation=True, max_length=50)
        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]
    def __len__(self):
        return super().__len__()
    def __getitem__(self, i):
        return (self.label[i], self.input_ids[i], self.attention_mask[i])

if __name__=='__main__':
    from utils.const import *
    f = os.path.join(DIR_DATA, 'livedoor&text=title.csv')
    dataset = pd.read_csv(f)
    dataset = LstmLivedoorDataset()
    print('dataset length', len(dataset))
    print('dataset sample', dataset[1000])