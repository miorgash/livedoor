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
        return len(self.label)
  
    def __getitem__(self, idx):
        return (self.label[idx], self.text[idx])

if __name__=='__main__':
    from const import *
    f = os.path.join(DIR_DATA, 'livedoor&text=text.csv')
    dataset = pd.read_csv(f)
    dataset = LivedoorDataset(dataset)
    print('dataset length', len(dataset))
    print('dataset sample', dataset[1000])