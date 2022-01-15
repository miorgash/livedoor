import os
from typing import Tuple, Callable
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from const import *
# import pickle5 as pickle
import pickle
from modeling.lstm_classifier import LSTMClassifier
from utils.data import LivedoorDataset
from tokenizer.sudachi_tokenizer import SudachiTokenizer
from torch.nn.utils.rnn import pad_sequence

def test(vocab: Vocab,
        model: nn.Module, 
        dataloader: DataLoader,
        text_pipeline: Callable, 
        loss_fn: Callable) -> Tuple:
    current_size = 0
    current_correct, mean_correct = 0, 0
    current_loss, mean_loss = 0, 0

    with torch.no_grad():

        for labels, texts in dataloader:
            # Tokenize, indexing and padding
            texts = [torch.tensor(text_pipeline(text)) for text in texts]
            texts = pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])

            # Feed forward
            labels, texts = labels.to(DEVICE), texts.to(DEVICE)
            pred = model(texts)[0]

            # Get loss
            loss = loss_fn(pred, labels)

            # Evaluate score with test data
            labels_pred = pred.argmax(axis=1).squeeze()
            current_correct += (labels_pred==labels).type(torch.int).sum().item()
            current_loss += loss.item()
            current_size += len(labels)
            mean_correct = current_correct / current_size   # accuracy
            mean_loss = current_loss / current_size

    return mean_correct, mean_loss

if __name__ == '__main__':
    # File i/o
    with open(os.path.join(DIR_BIN, "vectors.pkl"), "rb") as f:
        vectors = pickle.load(f)
    with open(os.path.join(DIR_BIN, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    dataframe = pd.read_csv(os.path.join(DIR_DATA, 'test.csv'))

    # Declare
    BATCH_SIZE = 4
    H_DIM = 100
    CLASS_DIM = 9
    LR = 1e-1
    dataloader = DataLoader(
        LivedoorDataset(dataframe), 
        batch_size=BATCH_SIZE, 
        shuffle=True)
    epoch = 100
    tokenizer = SudachiTokenizer()
    text_pipeline = lambda text: [vocab[token] for token in tokenizer.tokenized_text(text)]
    model = LSTMClassifier(
        embedding = vectors,
        h_dim = H_DIM,
        class_dim = CLASS_DIM)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    # 指定のエポック数だけ繰り返し
    for i in range(epoch):
        correct, loss = test(vocab, model, dataloader, text_pipeline, loss_fn)