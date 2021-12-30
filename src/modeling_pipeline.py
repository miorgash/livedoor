from modeling.train import train
from modeling.test import test

import os
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from const import *
import pickle5 as pickle
from modeling.lstm_classifier import LSTMClassifier
from livedoor_dataset import LivedoorDataset
from tokenizer.sudachi_tokenizer import SudachiTokenizer
from torchtext.vocab import Vocab

if __name__ == "__main__":
    # File i/o
    with open(os.path.join(DIR_BIN, "vectors.pkl"), "rb") as f:
        vectors = pickle.load(f)
    with open(os.path.join(DIR_BIN, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    df_train = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'))
    df_test = pd.read_csv(os.path.join(DIR_DATA, 'test.csv'))

    # Declare
    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 128
    H_DIM = 100
    CLASS_DIM = 9
    LR = 1e-1
    train_dataloader = DataLoader(
        LivedoorDataset(df_train), 
        batch_size=TRAIN_BATCH_SIZE, 
        shuffle=True)
    test_dataloader = DataLoader(
        LivedoorDataset(df_test), 
        batch_size=TEST_BATCH_SIZE, 
        shuffle=True)
    epoch = 100
    tokenizer = SudachiTokenizer()
    text_pipeline = lambda text: [vocab[token] for token in tokenizer.tokenized_text(text)]
    model = LSTMClassifier(
        embedding = torch.Tensor(vectors).to(DEVICE),
        h_dim = H_DIM,
        class_dim = CLASS_DIM)
    model = model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    # 指定のエポック数だけ繰り返し
    for i in range(epoch):
        if i == 0:
            print(f"|           | train                 | test                  |")
            print(f"|           | accuracy  | mean_loss | accuracy  | mean_loss |")
        print(f"| epoch {i:3d} ", end='')
        accuracy_train, loss_train = train(vocab, model, train_dataloader, text_pipeline, loss_fn, optimizer)
        print(f"|  {accuracy_train:0.6f} |  {loss_train:0.6f} ", end='')
        accuracy_test, loss_test = test(vocab, model, test_dataloader, text_pipeline, loss_fn)
        print(f"|  {accuracy_test:0.6f} |  {loss_test:0.6f} |")