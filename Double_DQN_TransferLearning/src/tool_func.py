import pandas as pd
import numpy as np
def load_datasets(data_dir, tst_frac = 0.1):
    dataframe = pd.read_csv(data_dir,index_col= 0 ,dtype=np.float32)
    dataframe[['label','gmm_id']] = dataframe[['label','gmm_id']].astype(np.int64)
    dataframe = dataframe.reset_index(drop = True)
    data_num, _ = dataframe.shape
    dataframe['weights'] = 1.0
    dataframe['weights'] = dataframe['weights'].astype(np.float32)
    dataframe = dataframe.sample(frac = 1.0)
    train_num = int(data_num*(1-tst_frac))
    train_data = dataframe[0:train_num]
    test_data = dataframe[train_num:data_num]
    return {'train':train_data, 'tst':test_data}

def load_localdata(data_dir, part_frac = 0.5):
    dataframe = pd.read_csv(data_dir,index_col= 0 ,dtype=np.float32)
    dataframe[['label','gmm_id']] = dataframe[['label','gmm_id']].astype(np.int64)
    dataframe = dataframe.reset_index(drop = True)
    data_num, _ = dataframe.shape
    dataframe['weights'] = 1.0
    dataframe['weights'] = dataframe['weights'].astype(np.float32)
    train_num = int(data_num*(1-part_frac))
    train_data = dataframe[0:train_num]
    test_data = dataframe[train_num:data_num]
    train_data = train_data.sample(frac = 1.0)
    test_data = test_data.sample(frac = 1.0)
    return {'idata':train_data, 'jdata':test_data, 'alldata':dataframe}