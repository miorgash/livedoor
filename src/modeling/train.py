import os
from typing import Tuple, Callable
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from const import *
import pickle5 as pickle
from modeling.lstm_classifier import LSTMClassifier
from livedoor_dataset import LivedoorDataset
from tokenizer.sudachi_tokenizer import SudachiTokenizer
from torch.nn.utils.rnn import pad_sequence

def train(model: nn.Module, texts: Tuple, labels: torch.tensor, text_pipeline: Callable) -> nn.Module:
    """1 バッチの学習
    Args:
        batch (): 学習用データのバッチ
    Returns:
        学習済み model
    """
    # Tokenize, indexing and padding
    texts = [torch.tensor(text_pipeline(text)) for text in texts]
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])

    # Feed forward
    labels, texts = labels.to(DEVICE), texts.to(DEVICE)
    print(1)

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
    tokenizer = SudachiTokenizer()
    text_pipeline = lambda text: [vocab[token] for token in tokenizer.tokenized_text(text)]

    # 指定のエポック数だけ繰り返し
    for i in range(epoch):
        # バッチごとにループ
        for batch, (labels, texts) in enumerate(dataloader):
            model = train(model, texts, labels, text_pipeline)
        # エポックごとに保存