import pandas as pd
from torch.utils.data.dataset import Dataset

class LivedoorDataset(Dataset):
    def __init__(self, dataframe):
        '''
        input:
            dataframe: pandas.DataFrame of (label, text) shape.
        output:
            None
        '''
        self.label = dataframe.iloc[:, 0]
        self.text = dataframe.iloc[:, -1]
      
    def __len__(self):
        raise NotImplementedError
  
    def __getitem__(self, idx):
        raise NotImplementedError

class LstmLivedoorDataset(LivedoorDataset):
    def __init__(self, dataframe):
        super().__init__(dataframe)
    def __len__(self):
        return len(self.label)
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
    f = os.path.join(DIR_DATA, 'livedoor&text=text.csv')
    dataset = pd.read_csv(f)
    dataset = LivedoorDataset(dataset)
    print('dataset length', len(dataset))
    print('dataset sample', dataset[1000])