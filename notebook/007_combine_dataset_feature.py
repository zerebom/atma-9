#%%

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


# %%
from pathlib import Path
sys.path.append('../')
from src.dataset import RetailDataset, create_target_from_log,only_payment_session_record,create_payment,annot_category

from src.features.features import CountEncodingBlock, DateBlock, PublicLogBlock,MetaInformationBlock,UserHistoryBlock
from src.utils import timer

path = Path('../input')
dataset = RetailDataset(file_path=path,thres_sec=10*60)
dataset.prepare_data()

#%%

#%%
# １０分以上の買い物をしたsession
#支払ったログかつ、testじゃない
wo_test_log = dataset.whole_log
pay_wo_test_log = dataset.get_payment_log(wo_test_log)

is_item_record = pay_wo_test_log['kind_1'] == '商品'
max_payed_time = pay_wo_test_log[is_item_record].groupby('session_id')['spend_time'].max()
max_payed_time_over_10min = max_payed_time[max_payed_time > 10 * 60]

train_sessions = max_payed_time_over_10min.index.tolist()

# testにない+paymentあり+last_item_add > 10min
train_log = pay_wo_test_log[pay_wo_test_log['session_id'].isin(train_sessions)].reset_index(drop=True)
#(8162072, 18)
print(train_log.shape)
train_log.head()


#%%

meta = dataset.meta
time_elasped_count = meta['time_elapsed'].value_counts(normalize=True)
train_time_elapsed = np.random.choice(time_elasped_count.index.astype(int),
                                      p=time_elasped_count.values,
                                      size=len(train_sessions))

train_meta = pd.DataFrame({
    'session_id': list(train_sessions),
    'time_elapsed': train_time_elapsed
})

train_meta = pd.merge(train_meta,
                         meta.drop(columns=['time_elapsed']),
                         on='session_id',
                         how='left')
#(379625, 7)
print(train_meta.shape)
train_meta.head()

#%%

_df = pd.merge(train_log[['session_id', 'spend_time']], train_meta, on='session_id', how='left')
idx_show = _df['spend_time'] <= _df['time_elapsed'] * 60

train_pub_log = train_log[idx_show].reset_index(drop=True)
train_pri_log = train_log[~idx_show].reset_index(drop=True)

#(1788989, 18) (6373083, 18)
print(train_pub_log.shape,train_pri_log.shape)
#0.2807101366795317
print(len(train_pub_log)/len(train_pri_log))
#%%

test_input_log = dataset.get_test_input_log()
public_log = pd.concat([
    train_pub_log, test_input_log
], axis=0, ignore_index=True)

print(public_log.shape)

#%%
#263852/379625
print(train_pub_log['session_id'].nunique())
print(train_pri_log['session_id'].nunique())


#%%
train_private_df = train_pri_log.rename(columns={'JAN':'value_1'})
train_target_df,_ = create_target_from_log(train_pri_log,
                       product_master_df=dataset.product_master,
                       TARGET_IDS=dataset.target_category_ids,
                       only_payment=False)


#%%
feature_blocks = [
    *[CountEncodingBlock(column=c) for c in ['hour']],
    DateBlock(),
    PublicLogBlock(dataset),
    MetaInformationBlock(),
    UserHistoryBlock(dataset),
]

# %%
feat_train_df = pd.DataFrame()

for block in feature_blocks:
    with timer(prefix='fit {} '.format(block)):
        out_i = block.fit(train_meta)
    assert len(train_meta) == len(out_i), block
    feat_train_df = pd.concat([feat_train_df, out_i], axis=1)

# %%

test = dataset.test
test_meta_df = pd.merge(test, meta, on='session_id', how='left')
feat_test_df = pd.DataFrame()

for block in feature_blocks:
    with timer(prefix='fit {} '.format(block)):
        out_i = block.transform(test_meta_df)

    assert len(test_meta_df) == len(out_i), block
    feat_test_df = pd.concat([feat_test_df, out_i], axis=1)
#%%
assert len(train_target_df) == len(feat_train_df)


#%%

from src.model_utils import fit_and_predict,create_predict
TARGET_IDS = dataset.target_category_ids

oof_df, pred_df = pd.DataFrame(), pd.DataFrame()

for i in TARGET_IDS:
    oof, models = fit_and_predict(train_df=feat_train_df,
                                target_df=train_target_df,
                                target_id=i)

    # 予測モデルで推論実行
    with timer(prefix='predict {} '.format(i)):
        pred = create_predict(models, input_df=feat_test_df)

    oof_df[i] = oof
    pred_df[i] = pred

# %%
train_target_df




# %%
