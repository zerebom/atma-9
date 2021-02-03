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
from src.dataset import RetailDataset

# %%
# %%

INPUT_DIR = '../input/'
OUTPUT_DIR = './tutorial2_outputs'

os.makedirs(OUTPUT_DIR, exist_ok=True)
product_master_df = pd.read_csv(os.path.join(INPUT_DIR, 'product_master.csv'), dtype={ 'JAN': str })
log_df = pd.read_csv(os.path.join(INPUT_DIR, 'carlog.csv'), dtype={ 'value_1': str }, date_parser=['date'])
meta_df = pd.read_csv(os.path.join(INPUT_DIR, 'meta.csv'))
test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))

test_sessions = test_df['session_id'].unique()

#%%

display_action_df = pd.read_csv(os.path.join(INPUT_DIR, 'display_action_id.csv'))
display_action_names = display_action_df['display_name'] + '_' + display_action_df['action_name']
display_action2name = dict(zip(display_action_df['display_action_id'], display_action_names))


#%%
# 完全なデータを持っているログに絞る
test_sessions = test_df['session_id'].unique()
idx_test = log_df['session_id'].isin(test_sessions)
whole_log_df = log_df[~idx_test].reset_index(drop=True)

payment_session_df = only_payment_session_record(whole_log_df)

# %%
# train_sessions = payment_session_df['session_id'].unique()

# 商品購買の最後(max spend time)が10分より大きいセッションを取り出す
is_item_record = payment_session_df['kind_1'] == '商品'
max_payed_time = payment_session_df[is_item_record].groupby('session_id')['spend_time'].max()
max_payed_time_over_10min = max_payed_time[max_payed_time > 10 * 60]

train_sessions = max_payed_time_over_10min.index.tolist()
train_whole_log_df = payment_session_df[payment_session_df['session_id'].isin(train_sessions)].reset_index(drop=True)


# %%

time_elasped_count = meta_df['time_elapsed'].value_counts(normalize=True)
train_time_elapsed = np.random.choice(time_elasped_count.index.astype(int),
                                      p=time_elasped_count.values,
                                      size=len(train_sessions))

train_meta_df = pd.DataFrame({
    'session_id': train_sessions,
    'time_elapsed': train_time_elapsed
})

train_meta_df = pd.merge(train_meta_df,
                         meta_df.drop(columns=['time_elapsed']),
                         on='session_id',
                         how='left')


# %%

_df = pd.merge(train_whole_log_df[['session_id', 'spend_time']], train_meta_df, on='session_id', how='left')
idx_show = _df['spend_time'] <= _df['time_elapsed'] * 60

train_public_df = train_whole_log_df[idx_show].reset_index(drop=True)
train_private_df = train_whole_log_df[~idx_show].reset_index(drop=True)

# テストのログデータと合わせて推論時に見ても良いログ `public_log_df` として保存しておく
public_log_df = pd.concat([
    train_public_df, log_df[log_df['session_id'].isin(test_sessions)]
], axis=0, ignore_index=True)

# %%

test_meta_df = pd.merge(test_df, meta_df, on='session_id', how='left')

# %%

train_target_df, _  = create_target_from_log(train_private_df,
                                             product_master_df=product_master_df,
                                            only_payment=False)

# %%


#%%

feature_blocks = [
    *[CountEncodingBlock(column=c) for c in ['hour']],
    DateBlock(),
    PublicLogBlock(),
    MetaInformationBlock(),
    UserHistoryBlock(),
]

feat_train_df = pd.DataFrame()

for block in feature_blocks:
    with timer(prefix='fit {} '.format(block)):
        out_i = block.fit(train_meta_df)
    assert len(train_meta_df) == len(out_i), block
    feat_train_df = pd.concat([feat_train_df, out_i], axis=1)
#%%

feat_test_df = pd.DataFrame()

for block in feature_blocks:
    with timer(prefix='fit {} '.format(block)):
        out_i = block.transform(test_meta_df)

    assert len(test_meta_df) == len(out_i), block
    feat_test_df = pd.concat([feat_test_df, out_i], axis=1)
#%%



#%%

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