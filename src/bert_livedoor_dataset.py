import pandas as pd
from torch.utils.data.dataset import Dataset
from transformers import AutoModel, AutoTokenizer
from preproc import create_table

class BertLivedoorDataset(Dataset):
    def __init__(self):
        '''
        input:
            dataframe: pandas.DataFrame of (label, text) shape.
        output:
            None
        '''
        tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v2")
        TEXT_COLUMN = 'title'
        table = create_table.create_table(input_dir='/data/ldcc/text/', text_column=TEXT_COLUMN)
        self.label = list(table.iloc[:, 0])
        self.text = list(table.iloc[:, -1])
        inputs = tokenizer(self.text, return_tensors="pt", 
                           padding=True, truncation=True, max_length=50)
        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]
      
    def __len__(self):
        return len(self.label)
  
    def __getitem__(self, i):
        return (self.label[i], self.input_ids[i], self.attention_mask[i])

if __name__=='__main__':
    dataset = BertLivedoorDataset()
    print('dataset length', len(dataset))
    print('dataset sample', dataset[1000])