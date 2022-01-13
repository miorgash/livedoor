import pandas as pd
from torch.utils.data.dataset import Dataset
from preproc.create_table import create_table

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
        self.label = table[LABEL_COLUMN]
        self.text = table[TEXT_COLUMN]
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        raise NotImplementedError

class LstmLivedoorDataset(LivedoorDataset):
    def __init__(self):
        super().__init__()
    def __len__(self):
        return super().__len__()
    def __getitem__(self, i):
        return (self.label[i], self.text[i])

class BertLivedoorDataset(LivedoorDataset):
    def __init__(self, dataframe):
        super().__init__(dataframe)
    def __len__(self):
        return super().__len__()
    def __getitem__(self, idx):
        return super().__getitem__(idx)

if __name__=='__main__':
    from const import *
    f = os.path.join(DIR_DATA, 'livedoor&text=title.csv')
    dataset = pd.read_csv(f)
    dataset = LstmLivedoorDataset()
    print('dataset length', len(dataset))
    print('dataset sample', dataset[1000])