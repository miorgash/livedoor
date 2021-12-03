import csv
import os
from collections import Counter

import gensim
import pickle5 as pickle
from const import *
from tokenizer.sudachi_tokenizer import SudachiTokenizer
from torchtext.vocab import Vocab
from tqdm import tqdm


def create_vocab(input_corpus, input_pretrained_vectors) -> Vocab:
    '''学習用コーパスと学習済み分散表現に含まれる語の Vocab を作成する。
    Args:
        input_corpus (str): 学習用コーパスのファイルパス
        input_pretrained_vectors (str): 学習済み分散表現のファイルパス
    Returns:
        モデル構築用の語彙 (torchtext.vocab.Vocab)
    '''
    # Vocab
    tokenizer = SudachiTokenizer()
    counter = Counter()
    vectors = gensim.models.KeyedVectors.load(input_pretrained_vectors)

    # vectors を用いたフィルタリング
    words_found_in_pretrained_embedding = set([t for t in vectors.vocab.keys()])

    with open(input_corpus, 'r') as f:
        reader = csv.reader(f)
        # header
        _ = next(reader)
        # all text
        texts = map(lambda row: tokenizer.tokenized_text(row[1]), reader)
        for text in tqdm(texts):
            counter.update(words_found_in_pretrained_embedding & set(text))
    
    return Vocab(counter)

if __name__ == '__main__':
    input_corpus = os.path.join(DIR_DATA, 'train.csv')
    input_pretrained_vectors = '/data/chive_v1.2mc90/chive-1.2-mc90_gensim/chive-1.2-mc90.kv'
    output_file = os.path.join(DIR_BIN, 'vocab.pkl')
    vocab = create_vocab(input_corpus, input_pretrained_vectors)

    if os.path.isfile(output_file):
        print(f'file exists: {output_file}')
    else:
        with open(output_file, 'wb') as f:
            pickle.dump(vocab, f, protocol=5)
        print(f'file created: {output_file}')