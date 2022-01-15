from typing import Tuple, Callable
from torch import nn
from torch.utils.data import DataLoader
from const import *

def test(dataloader: DataLoader,
        model: nn.Module, 
        loss_fn: Callable) -> Tuple:
    current_size = 0
    current_correct, mean_correct = 0, 0
    current_loss, mean_loss = 0, 0

    with torch.no_grad():

        for labels, texts in dataloader:

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
    pass