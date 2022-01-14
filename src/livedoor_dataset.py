import pandas as pd
from torch.utils.data.dataset import Dataset
from preproc.create_table import create_table
from transformers import AutoTokenizer

class LivedoorDataset(Dataset):
    def __init__(self):
        '''
        input:
            dataframe: pandas.DataFrame of (label, text) shape.
        output:
            None
        '''
        INPUT_DIR    = "/data/ldcc/text/"
        LABEL_COLUMN = "media"
        TEXT_COLUMN  = "title"
        # TODO: text_column の内容が利用者から隠蔽されていてよくない；全項目返すよう変更？
        table = create_table(input_dir=INPUT_DIR, text_column=TEXT_COLUMN)
        self.label = table[LABEL_COLUMN].tolist()
        self.text = table[TEXT_COLUMN].tolist()
    def __len__(self):
        return len(self.label)
    def __getitem__(self):
        raise NotImplementedError

class LstmLivedoorDataset(LivedoorDataset):
    def __init__(self):
        super().__init__()
    def __len__(self):
        return super().__len__()
    def __getitem__(self, i):
        return (self.label[i], self.text[i])

class BertLivedoorDataset(LivedoorDataset):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v2")
        inputs = self.tokenizer(self.text, return_tensors="pt", 
                           padding=True, truncation=True, max_length=50)
        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]
    def __len__(self):
        return super().__len__()
    def __getitem__(self, i):
        return (self.label[i], self.input_ids[i], self.attention_mask[i])

if __name__=='__main__':
    from const import *
    f = os.path.join(DIR_DATA, 'livedoor&text=title.csv')
    dataset = pd.read_csv(f)
    dataset = LstmLivedoorDataset()
    print('dataset length', len(dataset))
    print('dataset sample', dataset[1000])