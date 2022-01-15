import os
import pandas as pd
import pickle
from const import *
from modeling import lstm_modeling_pipeline
from utils.data import LstmLivedoorDataset

if __name__ == "__main__":
    # File i/o
    with open(os.path.join(DIR_BIN, "title.vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    with open(os.path.join(DIR_BIN, "title.vectors.pkl"), "rb") as f:
        vectors = pickle.load(f)
    dataset = LstmLivedoorDataset()

    lstm_modeling_pipeline.run(vocab, vectors, dataset,
        train_batch_size=128, test_batch_size=1024,
        h_dim=100, lr=1e-1, epoch=100)