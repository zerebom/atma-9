# %%

import gc
import os
import random
import numpy as np
from numpy.core.fromnumeric import prod
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

import seaborn as sns
import IPython.display as ipD
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
from sklearn.metrics import roc_auc_score

from pathlib import Path
sys.path.append('../')

from src.dataset import RetailDataset, create_target_from_log, only_payment_session_record, create_payment, annot_category
from src.features.features import CountEncodingBlock, DateBlock, PublicLogBlock, MetaInformationBlock, UserHistoryBlock,TSNEEmbBlock,UserMasterBlock

from src.model_utils import fit_and_predict, create_predict
from src.utils import timer, savefig

# %%
path = Path('../input')
dataset = RetailDataset(file_path=path, thres_sec=10*60)
dataset.prepare_data()


save_dir = Path('../input/tutorial2/')

train_target = pd.read_pickle(save_dir/'train_target.pkl')

train_meta = pd.read_pickle(save_dir/'train_meta.pkl')
test_meta = pd.read_pickle(save_dir/'test_meta.pkl')

train_pub_log = pd.read_pickle(save_dir/'train_pub_log.pkl')
train_pri_log = pd.read_pickle(save_dir/'train_pri_log.pkl')

# train_pub + test_whole_log
public_log = pd.read_pickle(save_dir/'public_log.pkl')

#%%
cartlog = dataset.cartlog
purchase_log = cartlog[cartlog['kind_1']=='商品']
purchase_log['price'] = purchase_log['n_items'] * purchase_log['unit_price']
#%%

#セッションごとの買い物の値段に関する特徴量
session_price = purchase_log.groupby('session_id')['price'].sum().reset_index()

session_price_df = meta[['user_id','session_id']].\
                    merge(session_price,how='left',on='session_id').fillna(0)

session_price_agg_u_df = session_price_df.groupby('user_id')['price'].agg(['mean','std','max','min','count']).fillna(0).add_prefix('ses_price_').reset_index()


session_df = meta[['session_id','user_id']]

cumsum = session_price_df.groupby('user_id')['price'].cumsum().fillna(0)
session_df['session_price_rolling_mean'] = cumsum / (session_price_df.groupby('user_id')['price'].cumcount()+1)

session_price_last10 = cumsum - cumsum.groupby(session_price_df['user_id']).shift(10)
ses_price_mean_10x = session_price_agg_u_df[['ses_price_mean','user_id']].\
                    merge(session_df[['user_id']],how='right',on='user_id')['ses_price_mean'] * 10

session_df['session_price_last10'] = session_price_last10.fillna(ses_price_mean_10x)


#%%
purchase_log.groupby('user_id')['price'].agg(['mean','std','max','min','count'])



# %%


product_master = dataset.product_master
target_category = '麺類__カップ麺'
log = dataset.whole_log
meta = dataset.meta

_df = pd.DataFrame()
from tqdm import tqdm

for bumon in tqdm(product_master['部門名'].unique()):
    jans = product_master[product_master['部門名']==bumon]['JAN'].unique()
    # 商品追加&カップ麺のjanのlog
    idx = (log['kind_1']=='商品') & (log['value_1'].isin(jans))
    # session_idごとにsum
    _log = log[idx].groupby('session_id')['n_items'].first()

    #'session_id', 'user_id', 'date'のユニーク値を抽出
    df = meta.groupby(['session_id', 'user_id', 'date'])\
        .first().reset_index().sort_values(['user_id', 'session_id'])

    #購入数をマージ
    df = pd.merge(df, _log, on='session_id', how='left')[['session_id', 'user_id', 'n_items']]
    df = df.set_index('session_id')
    df = df.fillna(0)


    cumsum = df.groupby('user_id')['n_items'].cumsum().fillna(0)
    _df[bumon] = (cumsum / (df.groupby('user_id')['n_items'].cumcount() + 1)).clip(0,1)
    break



#%%

#df:10_351_843 rows
purchase_df = dataset.get_payment_log(dataset.whole_log).rename(columns={ 'value_1': 'JAN' })

#series:10_351_843 rows(category名)
category = annot_category(purchase_df, dataset.product_master)
#series:10_351_843 rows(Bool)
idx_null = category.isnull()  # JAN が紐付かないやつ

# target の情報はリークになる可能性があるので削除する
#series:10_351_843 rows(~Bool)
idx_none_target = ~category.isin(dataset.target_category_ids)

# 商品マスタの部門名を取り出して集計
#series:10_351_843 rows(部門名)
bumon_name = pd.merge(purchase_df['JAN'],
                        dataset.product_master[['JAN', '部門名']], on='JAN', how='left')['部門名']

_df = pd.pivot_table(data=purchase_df[idx_none_target],
        index='user_id',
        #列にするlabel
        columns=bumon_name[idx_none_target],
        values='n_items',
        aggfunc='sum')\
        .fillna(0)



