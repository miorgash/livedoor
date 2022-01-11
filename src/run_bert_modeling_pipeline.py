import os
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from const import *
from torch.utils.data import Dataset, DataLoader, random_split
from bert_livedoor_dataset import BertLivedoorDataset
        
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v2")
    dataset = BertLivedoorDataset()

    # split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    TRAIN_BATCH_SIZE, TEST_BATCH_SIZE = 128, 1024
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

# DataSet/DataLoader

# loop epochs