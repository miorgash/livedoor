from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from const import *
from utils.data import LstmLivedoorDataset
from modeling.train import train
from modeling.test import test
from modeling.lstm_classifier import LSTMClassifier
def collate_fn(batch):
    labels, id_sequences = list(zip(*batch))
    labels = torch.stack(labels)
    id_sequences = pad_sequence(id_sequences, batch_first=True, padding_value=0)
    return labels, id_sequences

def run(dataset,
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
                                collate_fn=collate_fn,
                                shuffle=True)
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=test_batch_size, 
                                collate_fn=collate_fn,
                                shuffle=True)
    model = LSTMClassifier(
        embedding = torch.Tensor(dataset.vectors).to(DEVICE),
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
        accuracy_train, loss_train = train(train_dataloader, model, loss_fn, optimizer)
        print(f"|  {accuracy_train:0.6f} |  {loss_train:0.6f} ", end='')
        accuracy_test, loss_test = test(test_dataloader, model, loss_fn)
        print(f"|  {accuracy_test:0.6f} |  {loss_test:0.6f} |")
    return 1

if __name__ == "__main__":
    dataset = LstmLivedoorDataset()
    run(dataset,
        train_batch_size=64, test_batch_size=1024,
        h_dim=100, lr=1e-1, epoch=100)