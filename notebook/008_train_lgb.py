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
from src.features.features import CountEncodingBlock, DateBlock, PublicLogBlock, MetaInformationBlock, UserHistoryBlock,SessionPriceBlock
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
    SessionPriceBlock(dataset),
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

#%%
#from src.model_utils import to_lgbm_params

def to_lgbm_params(params: dict):
    retval = dict(**params)
    retval.update({
        'n_estimators': 10000,
        'learning_rate': .1,
        'objective': 'binary',
        # 'metric': 'rmse',
        'importance_type': 'gain',
        # 'verbose': -1
    })
    return retval

TARGET_IDS = dataset.target_category_ids
oofs, preds = pd.DataFrame(), pd.DataFrame()
feature_importance_list = []
cv = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

params = {'reg_lambda': 12.4,
 'reg_alpha': 0.014,
 'feature_fraction': 0.89,
 'bagging_fraction': 0.98,
 'max_depth': 6,
 'num_leaves': 50,
 'min_child_weight': 9.68}

params = to_lgbm_params(params)

for i in TARGET_IDS:
    oof, models, feature_importance_df = fit_and_predict(train_df=feat_train,
                                  target_df=train_target,
                                  target_id=i,
                                  params=params,
                                  cv=cv)

    feature_importance_list.append(feature_importance_df)
    # 予測モデルで推論実行
    with timer(prefix='predict {} '.format(i)):
        pred = create_predict(models, input_df=feat_test)

    oofs[i] = oof
    preds[i] = pred

#%%

#feの順番を保存
fe_dic ={}
for idx,id in enumerate(TARGET_IDS):
    _df = feature_importance_list[idx].groupby('column')['feature_importance'].mean().sort_values(ascending=False).reset_index()

    fe_dic[id] = list(_df.loc[_df['column'].str.contains('ratio'),'column'])

import pickle
with open('../output/008/feature_importance_dic.pkl', 'wb') as f:
    pickle.dump(fe_dic, f)






# %%

roc_auc_score((train_target[TARGET_IDS] > 0).astype(
    int), oofs[TARGET_IDS], average='macro')

# %%
TARGET_CATEGORIES = dataset.target_category
category_id2code = dataset.get_category_id2code()

# %%
nfigs = len(oofs.columns)
ncols = 5
nrows = - (- nfigs // ncols)
fig, axes = plt.subplots(figsize=(5 * ncols, 3 * nrows),
                         ncols=ncols, nrows=nrows, sharex=True)
axes = np.ravel(axes)

with timer(prefix='plot oof-test-distributions '):
    for c, ax in zip(oofs.columns, axes):
        sns.histplot(data=pd.DataFrame({
            'OOF': oofs[c],
            'Test': preds[c]
        }), ax=ax, stat='density', common_norm=False,)
        ax.set_facecolor('white')
        ax.set_title(category_id2code[c])
        ax.grid()
    fig.tight_layout()
    savefig(fig, to='out-of-fold_vs_test')

# %%

nfigs = len(oofs.columns)
ncols = 5
nrows = - (- nfigs // ncols)
fig, axes = plt.subplots(figsize=(5 * ncols, 3 * nrows),
                         ncols=ncols, nrows=nrows, sharex=True)
axes = np.ravel(axes)

with timer(prefix='plot oof-test-distributions '):
    for c, ax in zip(oofs.columns, axes):
        sns.histplot(data=pd.DataFrame({
            'OOF': oofs[c],
            'Test': preds[c]
        }), ax=ax, stat='density', common_norm=False,)
        ax.set_title(category_id2code[c])
        ax.grid()
        ax.set_xlim(-.02, .2)
    fig.tight_layout()
    savefig(fig, to='out-of-fold_vs_test_拡大')

# %%
out_dir = Path('../output/008')
out_dir.mkdir(parents=True, exist_ok=True)
preds.to_csv(out_dir/'tutorial2_submission_v3.csv', index=False)

# %%
#optuna


#%%




