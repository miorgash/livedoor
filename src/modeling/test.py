import torch
from torch import nn
from torch.utils.data import DataLoader
from modeling.lstm_classifier import LSTMClassifier

def test(model: nn.Module, data: DataLoader, params: dict) -> nn.Module:
    # Tokenize & indexing

    # Pad

    # Feed forward

    # Get loss

    # Get gradient
    pass

    # Back propagate
    pass

    # Evaluate score with test data

    # Logging

if __name__ == '__main__':
    dataloader = DataLoader()
    for batch, (labels, texts) in enumerate(dataloader):
        test()