import pandas as pd
from preproc.create_table import create_table
from preproc.create_splits import create_splits
from preproc.create_vocab import create_vocab
from preproc.create_vectors import create_vectors
from const import *
from util import create_dataframe_if_not_exists, create_pickle_if_not_exists
import gensim
import pickle5 as pickle


if __name__ == '__main__':

    # shared constants
    TEXT_COLUMN = 'title'

    print("# create_table")
    LDCC_DATA_DIR = '/data/ldcc/text/'
    create_dataframe_if_not_exists(
        dataframe=create_table(LDCC_DATA_DIR, TEXT_COLUMN),
        output_file=os.path.join(DIR_DATA, f'livedoor&text={TEXT_COLUMN}.csv')
    )
    
    print("# create_splits")
    input_file = os.path.join(DIR_DATA, f'livedoor&text={TEXT_COLUMN}.csv')
    ldcc_df = pd.read_csv(input_file)
    train, test = create_splits(ldcc_df)
    
    for split, dataframe in zip(('train', 'test'), (train, test)):
        create_dataframe_if_not_exists(
            dataframe=dataframe,
            output_file=os.path.join(DIR_DATA, f'{TEXT_COLUMN}.{split}.csv')
        )
    
    print("# create_vocab")
    input_corpus = os.path.join(DIR_DATA, f'{TEXT_COLUMN}.train.csv')
    input_pretrained_vectors = '/data/chive_v1.2mc90/chive-1.2-mc90_gensim/chive-1.2-mc90.kv'
    output_file = os.path.join(DIR_BIN, f'{TEXT_COLUMN}.vocab.pkl')
    vocab = create_vocab(input_corpus, input_pretrained_vectors)
    create_pickle_if_not_exists(vocab, output_file)

    print("# create_vectors")
    with open(os.path.join(DIR_BIN, 'title.vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    pretrained_vectors = gensim.models.KeyedVectors.load(
        '/data/chive_v1.2mc90/chive-1.2-mc90_gensim/chive-1.2-mc90.kv'
    )
    vectors = create_vectors(vocab, pretrained_vectors)
    create_pickle_if_not_exists(
        pickle_obj=vectors,
        output_file=os.path.join(DIR_BIN, 'title.vectors.pkl')
    )