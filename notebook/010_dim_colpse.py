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
from src.features.features import CountEncodingBlock, DateBlock, PublicLogBlock, MetaInformationBlock, UserHistoryBlock
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

print(train_target.shape, feat_train.shape)



#%%
product_master = dataset.product_master
purchase_df = dataset.get_payment_log(dataset.whole_log).rename(columns={ 'value_1': 'JAN' })
category = annot_category(purchase_df, product_master)
idx_none_target = ~category.isin(dataset.target_category_ids)

bumon_name = pd.merge(purchase_df['JAN'],
                              product_master[['JAN', '部門名']], on='JAN', how='left')['部門名']

_df = pd.pivot_table(data=purchase_df[idx_none_target],
        index='user_id',
        columns=bumon_name[idx_none_target],
        values='n_items',
        aggfunc='sum')\
        .fillna(0)


_df.head()
#%%
user_category_ratio_df = _df.div(_df.sum(axis=1), axis=0).fillna(0)
user_category_ratio_arr = user_category_ratio_df.values
user_category_ratio_arr = np.clip(user_category_ratio_arr,0,1).astype(np.float32)


#%%
from openTSNE import TSNE
tsne = TSNE()
embedding = tsne.fit(user_category_ratio_arr)


# %%

vis_x = embedding[:, 0]
vis_y = embedding[:, 1]
max_idx = np.argmax(user_category_ratio_arr,axis=1)
plt.scatter(vis_x, vis_y,  c=max_idx, cmap=plt.cm.get_cmap("jet", 124), marker='.')
plt.colorbar(ticks=range(124))
plt.clim(-0.5, 123.5)
plt.show()


# %%
emb_2d_df = pd.DataFrame(user_category_ratio_df.reset_index()['user_id'])
emb_2d_df['dim1'] = np.array(embedding)[:,0]
emb_2d_df['dim2'] = np.array(embedding)[:,1]


# %%
emb_2d_df.to_csv('../input/tsne_prodcut_ratio_by_user_emb2d.csv',index=None)


# %%

_df = dataset.user_master[['age','gender']].astype('category')




# %%
