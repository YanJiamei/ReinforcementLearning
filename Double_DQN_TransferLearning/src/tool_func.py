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

def load_weighted_datasets(data_dir):
    dataframe = pd.read_csv(data_dir,index_col= 0 ,dtype=np.float32)
    dataframe[['label','gmm_id']] = dataframe[['label','gmm_id']].astype(np.int64)
    return dataframe

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

def load_dataframe(data_dir):
    dataframe = pd.read_csv(data_dir,index_col= 0 ,dtype=np.float32)
    dataframe[['label','gmm_id']] = dataframe[['label','gmm_id']].astype(np.int64)
    dataframe = dataframe.reset_index(drop = True)
    data_num, _ = dataframe.shape
    dataframe['weights'] = 1.0
    dataframe['weights'] = dataframe['weights'].astype(np.float32)
    return dataframe

def reconcat_dataset(data_0, data_1, data_2):
    dataframe_0 = load_dataframe(data_0)
    dataframe_1 = load_dataframe(data_1)
    dataframe_2 = load_dataframe(data_2)
    set_0 = dataframe_0.sample(10000)
    set_1 = dataframe_1.sample(10000)
    set_2 = dataframe_2.sample(10000)
    set_all = pd.DataFrame()
    set_all = pd.concat((set_all,set_0,set_1,set_2),axis=0, ignore_index=True )
    set_all = set_all.sample(frac=1.0)
    datanum,_ = set_all.shape
    tst_all = set_all[0:int(datanum*0.2)]
    train_all = set_all[int(datanum*0.2):datanum]
    return {'data_1':set_0, 'data_2':set_1, 'data_3':set_2, 'set_all':set_all}

if __name__ == "__main__":
    datadir1 = './datas/50div10gmm10-1.csv'
    datadir2 = './datas/50div10gmm10-2.csv'
    datadir3 = './datas/50div10gmm10-all.csv'
    datas = reconcat_dataset(datadir1, datadir2, datadir3)
    datas['data_1'].to_csv('./datas/50div10gmm10/d1.csv')
    datas['data_2'].to_csv('./datas/50div10gmm10/d2.csv')
    datas['data_3'].to_csv('./datas/50div10gmm10/d3.csv')
    datas['set_all'].to_csv('./datas/50div10gmm10/dall.csv')