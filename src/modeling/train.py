from typing import Tuple, Callable
from torch import nn
from torch.utils.data import DataLoader
from const import *

def train(dataloader: DataLoader,
          model: nn.Module, 
          loss_fn: Callable,
          optimizer: Callable) -> Tuple:
    """1 バッチの学習
    Args:
        batch (): 学習用データのバッチ
    Returns:
        学習済み model
    """
    current_size = 0
    current_correct, mean_correct = 0, 0
    current_loss, mean_loss = 0, 0

    # バッチごとにループ
    for labels, id_sequence in dataloader:

        # Feed forward
        labels, id_sequence = labels.to(DEVICE), id_sequence.to(DEVICE)
        pred = model(id_sequence)[0]

        # Get loss
        loss = loss_fn(pred, labels)

        # Get gradient
        optimizer.zero_grad()
        loss.backward()

        # Back propagate
        optimizer.step()

        # Evaluate score with train data
        labels_pred = pred.argmax(axis=1).squeeze()
        current_correct += (labels_pred==labels).type(torch.int).sum().item()
        current_loss += loss.item()
        current_size += len(labels)
        mean_correct = current_correct / current_size   # accuracy
        mean_loss = current_loss / current_size

    return mean_correct, mean_loss

if __name__ == '__main__':
    pass