import os
import pandas as pd
import pickle
from torch import nn
from torch.utils.data import DataLoader, random_split
from const import *
from livedoor_dataset import LstmLivedoorDataset
from tokenizer.sudachi_tokenizer import SudachiTokenizer
from modeling.train import train
from modeling.test import test
from modeling.lstm_classifier import LSTMClassifier

def run(vocab, vectors, dataset,
        train_batch_size: int,  test_batch_size: int,
        h_dim: int, lr: float, epoch: int) -> None:
    CLASS_DIM = 9
    MOMENTUM = 0.9
    
    # dataloader
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=train_batch_size,
                                shuffle=True)
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=test_batch_size, 
                                shuffle=True)
    # tokenizer
    tokenizer = SudachiTokenizer()
    text_pipeline = lambda text: [vocab[token] for token in tokenizer.tokenized_text(text)]
    model = LSTMClassifier(
        embedding = torch.Tensor(vectors).to(DEVICE),
        h_dim = h_dim,
        class_dim = CLASS_DIM)
    model = model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)

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
    return 1

if __name__ == "__main__":
    # File i/o
    with open(os.path.join(DIR_BIN, "title.vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    with open(os.path.join(DIR_BIN, "title.vectors.pkl"), "rb") as f:
        vectors = pickle.load(f)
    dataset = LstmLivedoorDataset()

    run(vocab, vectors, dataset,
        train_batch_size=64, test_batch_size=1024,
        h_dim=100, lr=1e-1, epoch=100)