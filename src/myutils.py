from collections import Counter
import pandas as pd
def get_n_label(dataset):
    '''データセットのラベル内訳を取得する
    
    Args:
      dataset: (label, text) 2 列の torch.utils.data.Subset
    Returns:
      らべる内訳横持ちの pandas.DataFrame
    '''
    c = Counter([l for l, _ in dataset])
    return pd.DataFrame(c.most_common()).set_index(0).sort_index().T