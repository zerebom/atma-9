
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
import japanize_matplotlib

# %%
from pathlib import Path
sys.path.append('../')

from src.dataset import RetailDataset, create_target_from_log, only_payment_session_record, create_payment, annot_category
from src.features.features import CountEncodingBlock, DateBlock, PublicLogBlock, MetaInformationBlock, UserHistoryBlock
from src.model_utils import fit_and_predict, create_predict
from src.utils import timer, savefig

# %%
path = Path('../input')
dataset = RetailDataset(file_path=path, thres_sec=10*60)
dataset.prepare_data()

# %%

save_dir = Path('../input/tutorial2/')

train_target = pd.read_pickle(save_dir/'train_target.pkl')

train_meta = pd.read_pickle(save_dir/'train_meta.pkl')
test_meta = pd.read_pickle(save_dir/'test_meta.pkl')

train_pub_log = pd.read_pickle(save_dir/'train_pub_log.pkl')
train_pri_log = pd.read_pickle(save_dir/'train_pri_log.pkl')

# train_pub + test_whole_log
public_log = pd.read_pickle(save_dir/'public_log.pkl')
# %%
product_master = dataset.product_master
target_category = '麺類__カップ麺'
log = dataset.whole_log
meta = dataset.meta

jans = product_master[product_master['category_name']==target_category]['JAN'].unique()
# 商品追加&カップ麺のjanのlog
idx = (log['kind_1']=='商品') & (log['value_1'].isin(jans))
# session_idごとにsum
_log = log[idx].groupby('session_id')['n_items'].sum()

#'session_id', 'user_id', 'date'のユニーク値を抽出
df = meta.groupby(['session_id', 'user_id', 'date'])\
    .first().reset_index().sort_values(['user_id', 'session_id'])

#購入数をマージ
df = pd.merge(df, _log, on='session_id', how='left')[['session_id', 'user_id', 'n_items']]
df = df.set_index('session_id')
df = df.fillna(0)

# 各ユーザごとに累積和。そしてシフトした累積和を引く
cumsum = df.groupby('user_id')['n_items'].cumsum().fillna(0)
cumsum - cumsum.groupby(df['user_id']).shift(3).fillna(0)







# %%

dataset.public_log = public_log

feat_train, feat_test = pd.DataFrame(), pd.DataFrame()

feature_blocks = [
    *[CountEncodingBlock(column=c) for c in ['hour']],
    DateBlock(),
    PublicLogBlock(dataset),
    MetaInformationBlock(),
    UserHistoryBlock(dataset),
]

for block in feature_blocks:
    with timer(prefix='fit {} '.format(block)):
        out_i = block.fit(train_meta)
    assert len(train_meta) == len(out_i), block
    feat_train = pd.concat([feat_train, out_i], axis=1)

for block in feature_blocks:
    with timer(prefix='fit {} '.format(block)):
        out_i = block.transform(test_meta)

    assert len(test_meta) == len(out_i), block
    feat_test = pd.concat([feat_test, out_i], axis=1)
# %%

print(feat_train.columns)
feat_train.head(30)

#%%
import pprint
pprint.pprint(list(feat_train.columns))


# %%
