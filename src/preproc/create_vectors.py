import os
import numpy as np
import pickle5 as pickle
import gensim
from util import create_pickle_if_not_exists

from const import *
def create_vectors(vocab, pretrained_vectors) -> np.array:
    '''学習用 vectors を作成する。軽量化のため学習用データ語彙との積集合のみ使用。
    Args:
        input_vocab (torchtext.vocab.Vocab): 学習用語彙
        input_pretrained_vectors (gensim.models.KeyedVectors): 学習済み分散表現
    Returns:
        モデル構築用の語彙に対応する分散表現
    '''
    # load
    vectors_for_unk_and_pad = np.zeros((2, 300))
    itos = vocab.get_itos()
    words = [itos[i] for i in range(len(vocab))]
    vectors = np.concatenate((vectors_for_unk_and_pad,
                           np.array([pretrained_vectors[w] for w in words[2:]])), axis=0)
    return vectors

if __name__ == '__main__':
    # vocab
    with open(os.path.join(DIR_BIN, 'title.vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    # pretrained vectors
    file = '/data/chive_v1.2mc90/chive-1.2-mc90_gensim/chive-1.2-mc90.kv'
    pretrained_vectors = gensim.models.KeyedVectors.load(file)

    # 学習用 pretrained vectors （train に含まれる語のみ）
    vectors = create_vectors(vocab, pretrained_vectors)

    create_pickle_if_not_exists(
        pickle_obj=vectors,
        output_file=os.path.join(DIR_BIN, 'title.vectors.pkl')
    )