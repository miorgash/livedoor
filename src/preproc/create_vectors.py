import os
import numpy as np
import pickle5 as pickle
import gensim

from const import *
def create_vectors(input_vocab: str, input_pretrained_vectors: str) -> np.array:
    '''学習用 vectors を作成する。軽量化のため学習用データ語彙との積集合のみ使用。
    Args:
        input_vocab (str): 学習用語彙
        input_pretrained_vectors (str): 学習済み分散表現のファイルパス
    Returns:
        モデル構築用の語彙に対応する分散表現
    '''
    # load
    with open(input_vocab, 'rb') as f:
        vocab = pickle.load(f)
    vectors = gensim.models.KeyedVectors.load(input_pretrained_vectors)
    
    vectors_for_unk_and_pad = np.zeros((2, 300))
    words = [vocab.itos[i] for i in range(len(vocab))]
    return np.concatenate((vectors_for_unk_and_pad,
                           np.array([vectors[w] for w in words[2:]])), axis=0)

if __name__ == '__main__':
    input_vocab = os.path.join(DIR_BIN, 'vocab.pkl')
    input_pretrained_vectors = '/data/chive_v1.2mc90/chive-1.2-mc90_gensim/chive-1.2-mc90.kv'
    output_file = os.path.join(DIR_BIN, 'vectors.pkl')
    vectors = create_vectors(input_vocab, input_pretrained_vectors)

    if os.path.isfile(output_file):
        print(f'file exists: {output_file}')
    else:
        with open(output_file, 'wb') as f:
            pickle.dump(vectors, f)