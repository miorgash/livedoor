import pandas as pd
from sudachi_tokenizer import SudachiTokenizer
from const import *
tokenizer = SudachiTokenizer()
df = pd.read_csv(os.path.join(DIR_DATA, 'train.csv')).head()
df['text'] = [[token for token in text] for text in tokenizer.tokenized_corpus(df['text'])]
print(df.head())
file = os.path.join(DIR_BIN, 'sample.parquet')
# df.to_parquet(file)

df2 = pd.read_parquet(file)
print(df2.head())