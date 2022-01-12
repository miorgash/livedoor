import os
import pandas as pd
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW
from const import *
from torch.utils.data import Dataset, DataLoader, random_split
from bert_livedoor_dataset import BertLivedoorDataset

def train(model, dataloader, optimizer):
    current_size = 0
    current_correct, mean_correct = 0, 0
    current_loss, mean_loss = 0, 0

    model.train()

    for labels, input_ids, attention_mask in dataloader:
        labels = labels.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        optimizer.zero_grad()
        output = model(input_ids, token_type_ids=None, attention_mask=attention_mask, labels= labels)
        loss = output['loss']
        logits = output['logits']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluate score with train data
        labels_pred = logits.argmax(axis=1)
        current_correct += (labels_pred==labels).type(torch.int).sum().item()
        current_loss += loss.item()
        current_size += len(labels)
        mean_correct = current_correct / current_size   # accuracy
        mean_loss = current_loss / current_size

    return mean_correct, mean_loss

def test(model, dataloader):
    current_size = 0
    current_correct, mean_correct = 0, 0
    current_loss, mean_loss = 0, 0

    model.eval()

    with torch.no_grad():
        for labels, input_ids, attention_mask in dataloader:
            labels = labels.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            
            output = model(input_ids, token_type_ids=None, attention_mask=attention_mask, labels= labels)
            loss = output['loss']
            logits = output['logits']
            optimizer.step()

            # Evaluate score with train data
            labels_pred = logits.argmax(axis=1)
            current_correct += (labels_pred==labels).type(torch.int).sum().item()
            current_loss += loss.item()
            current_size += len(labels)
            mean_correct = current_correct / current_size   # accuracy
            mean_loss = current_loss / current_size

    return mean_correct, mean_loss
        
if __name__ == "__main__":
    CHECKPOINT = "cl-tohoku/bert-base-japanese-v2"
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    dataset = BertLivedoorDataset()

    # split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    TRAIN_BATCH_SIZE, TEST_BATCH_SIZE = 32, 64
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    model = BertForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=9)
    model = model.to(DEVICE)

    for param in model.parameters():
        param.requires_grad = False

    # BERTの最後の層だけ更新ON
    for param in model.bert.encoder.layer[-1].parameters():
        param.requires_grad = True

    # クラス分類のところもON
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # optimizer = Adam(model.parameters(), lr=LR)
    optimizer = torch.optim.Adam([
        {'params': model.bert.encoder.layer[-1].parameters(), 'lr': 5e-7},
        {'params': model.classifier.parameters(), 'lr': 1e-5}
    ])

    # loop epochs
    EPOCHS = 50
    for i in range(EPOCHS):
        if i == 0:
            print(f"|           | train                 | test                  |")
            print(f"|           | accuracy  | mean_loss | accuracy  | mean_loss |")
        print(f"| epoch {i:3d} ", end='')
        accuracy_train, loss_train = train(model, train_dataloader, optimizer)
        print(f"|  {accuracy_train:0.6f} |  {loss_train:0.6f} ", end='')
        accuracy_test, loss_test = test(model, train_dataloader)
        print(f"|  {accuracy_test:0.6f} |  {loss_test:0.6f} |")