import os
from tokenizer.sudachi_tokenizer import SudachiTokenizer
from collections import Counter
from torchtext.vocab import Vocab
from const import *
import pickle5 as pickle
import gensim

def create_vocab(input_corpus, input_pretrained_vectors) -> Vocab:
    '''学習用コーパスと学習済み分散表現に含まれる語の Vocab を作成する。
    Args:
        input_corpus (str): 学習用コーパスのファイルパス
        input_pretrained_vectors (str): 学習済み分散表現のファイルパス
    Returns:
        モデル構築用の語彙 (torchtext.vocab.Vocab)
    '''
    # load
    vectors = gensim.models.KeyedVectors.load(input_pretrained_vectors)
    # create
    vocab = Vocab()
    return vocab

if __name__ == '__main__':
    input_corpus = os.path.join(DIR_DATA, 'train.csv')
    input_pretrained_vectors = '/data/chive_v1.2mc90/chive-1.2-mc90_gensim/chive-1.2-mc90.kv'
    output_file = os.path.join(DIR_DATA, 'vocab.pkl')
    vocab = create_vocab(input_corpus, input_pretrained_vectors)

    if os.path.isfile(output_file):
        print(f'file exists: {output_file}')
    else:
        with open(output_file, 'wb') as f:
            pickle.dump(vocab, f, protocol=5)
        print(f'file created: {output_file}')