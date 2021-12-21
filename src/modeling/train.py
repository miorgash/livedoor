import os
import pandas as pd
from pandas.core.frame import DataFrame
from torch import nn
from torch.utils.data import DataLoader
from const import *
import pickle5 as pickle
from modeling.lstm_classifier import LSTMClassifier
from livedoor_dataset import LivedoorDataset

def train(model: nn.Module, labels, text) -> nn.Module:
    """1 バッチの学習
    Args:
        batch (): 学習用データのバッチ
    Returns:
        学習済み model
    """
    # Tokenize & indexing


    # Pad

    # Feed forward

    # Get loss

    # Get gradient

    # Back propagate

    # Evaluate score with train data

    # Logging

    return model

if __name__ == '__main__':

    # File i/o
    with open(os.path.join(DIR_BIN, "vectors.pkl"), "rb") as f:
        vectors = pickle.load(f)
    with open(os.path.join(DIR_BIN, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    dataframe = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'))

    # Declare
    H_DIM = 100
    CLASS_DIM = 9
    dataloader = DataLoader(
        LivedoorDataset(dataframe), 
        batch_size=64, 
        shuffle=True)
    model = LSTMClassifier(
        embedding = vectors,
        h_dim = H_DIM,
        class_dim = CLASS_DIM)
    epoch = 100

    # 指定のエポック数だけ繰り返し
    for i in range(epoch):
        # バッチごとにループ
        for batch, (labels, text) in enumerate(dataloader):
            model = train(model, labels, text)
        # エポックごとに保存