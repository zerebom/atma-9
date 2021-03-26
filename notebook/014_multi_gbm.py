#%%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgbm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy


from lightgbm import LGBMClassifier
from catboost import CatBoost
from catboost import Pool
import xgboost as xgb

#%%

from pathlib import Path
sys.path.append('../')

from src.dataset import RetailDataset, create_target_from_log, only_payment_session_record, create_payment, annot_category
from src.features.features import CountEncodingBlock, DateBlock, PublicLogBlock, MetaInformationBlock, UserHistoryBlock,SessionPriceBlock
from src.model_utils import fit_and_predict, create_predict
from src.utils import timer, savefig

#%%



def fit_catboost(X, y, cv=None, params: dict=None, verbose=500):

    if params is None:
        params = deepcopy(CAT_DEFAULT_PARAMS)

    if cv is None:
        cv = StratifiedKFold(n_splits=2, shuffle=True)
    models = []
    # training data の target と同じだけのゼロ配列を用意
    # float にしないと悲しい事件が起こるのでそこだけ注意
    oof_pred = np.zeros_like(y, dtype=np.float)

    for i, (idx_train, idx_valid) in enumerate(cv.split(X, y)):
        # この部分が交差検証のところです。データセットを cv instance によって分割します
        # training data を trian/valid に分割
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = CatBoost(params=params)

        with timer(prefix='fit fold={} '.format(i + 1)):
            clf_train = Pool(x_train, y_train)
            clf_val = Pool(x_valid, y_valid)
            clf.fit(clf_train, eval_set=[clf_val])

        pred_i = clf.predict(x_valid, prediction_type='Probability')[:, 1]
        oof_pred[idx_valid] = pred_i
        models.append(clf)

        print(f'Fold {i} AUC: {roc_auc_score(y_valid, pred_i):.4f}')

    score = roc_auc_score(y, oof_pred)
    print('FINISHED \ whole score: {:.4f}'.format(score))
    return oof_pred, models, score










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
#%%

print(train_target.shape, feat_train.shape)
print(feat_test.shape)

#%%

def fit_xgb(X, y, cv=None, params: dict=None, verbose=500):

    if params is None:
        params = deepcopy(XGB_DEFAULT_PARAMS)

    if cv is None:
        cv = StratifiedKFold(n_splits=2, shuffle=True)
    models = []
    # training data の target と同じだけのゼロ配列を用意
    # float にしないと悲しい事件が起こるのでそこだけ注意
    oof_pred = np.zeros_like(y, dtype=np.float)

    for i, (idx_train, idx_valid) in enumerate(cv.split(X, y)):
        # この部分が交差検証のところです。データセットを cv instance によって分割します
        # training data を trian/valid に分割
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        with timer(prefix='fit fold={} '.format(i + 1)):

            print(x_train.shape,y_train.shape)
            print(x_valid.shape,y_valid.shape)


            dtrain = xgb.DMatrix(x_train, label=y_train)
            dval = xgb.DMatrix(x_valid, label=y_valid)
            evals = [(dtrain, 'train'), (dval, 'eval')]

            clf = xgb.train(params, dtrain, evals=evals,
                            early_stopping_rounds=params['early_stopping_rounds'], num_boost_round=params['num_boost_round'], verbose_eval=verbose)

        pred_i = clf.predict(dval)
        oof_pred[idx_valid] = pred_i
        models.append(clf)

        print(f'Fold {i} AUC: {roc_auc_score(y_valid, pred_i):.4f}')

    score = roc_auc_score(y, oof_pred)
    print('FINISHED \ whole score: {:.4f}'.format(score))
    return oof_pred, models, score

def fit_and_predict(train_df,
                    target_df,
                    target_id,
                    params=None,
                    cv=None,
                    gbm_func=None):

    """対象の target_id の学習を行なう"""
    category_id2code = dataset.get_category_id2code()
    target_name = category_id2code[target_id]

    print('-' * 20 + ' start {} '.format(target_name) + '-' * 20)

    if target_id not in TARGET_IDS:
        raise ValueError('`target_id` は {} から選んでください'.format(','.join(str, TARGET_IDS)))

    y = target_df[target_id].values
    y = np.where(y > 0, 1, 0)

    # speedup (n_split = 2)
    if cv is None:
        cv = StratifiedKFold(n_splits=2, random_state=71, shuffle=True)

    # モデルの学習.
    oof, models, _ = gbm_func(train_df.values, y, cv=cv, verbose=500,params=params)

    # 特徴重要度の可視化

    return oof, models

TARGET_IDS = dataset.target_category_ids
oofs, preds = pd.DataFrame(), pd.DataFrame()
feature_importance_list = []
cv = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

CAT_DEFAULT_PARAMS = {
    'objective': 'Logloss',
    'task_type': 'GPU',
    'boosting_type': "Plain",
    'iterations': 50000,
    'early_stopping_rounds':150,
    'learning_rate': 0.1,
    'verbose': 500,
}

XGB_DEFAULT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.1,
    'max_depth': 5,
    'num_leaves': 60,
    'num_boost_round': 10000,
    'early_stopping_rounds': 100,
    'random_state': 42,
    'tree_method': 'gpu_hist'
}

params = XGB_DEFAULT_PARAMS

def create_predict(models, input_df,type='lgbb') -> np.ndarray:
    """与えられた機械学習モデルで予測する"""
    pred = None
    if type=='lgbm':
        pred = np.array([model.predict_proba(input_df.values)[:, 1] for model in models])

    elif type=='cat':
        test_pool = Pool(input_df.values)
        pred = np.array([model.predict(test_pool, prediction_type='Probability')[:, 1] for model in models])
    else:
        dtrain = xgb.DMatrix(input_df.values)
        pred = np.array([model.predict(dtrain) for model in models])


    pred = np.mean(pred, axis=0)
    return pred


for i in TARGET_IDS:
    oof, models = fit_and_predict(train_df=feat_train,
                                  target_df=train_target,
                                  target_id=i,
                                  params=params,
                                  cv=cv,gbm_func=fit_xgb)

    # 予測モデルで推論実行
    with timer(prefix='predict {} '.format(i)):
        pred = create_predict(models, input_df=feat_test,type='xgb')

    oofs[i] = oof
    preds[i] = pred


# %%


roc_auc_score((train_target[TARGET_IDS] > 0).astype(
    int), oofs[TARGET_IDS], average='macro')


# %%
out_dir = Path('../output/014')
out_dir.mkdir(parents=True, exist_ok=True)
preds.to_csv(out_dir/'xgb_submission.csv', index=False)


# %%
import glob
sub_df_list = [pd.read_csv(path).values for path in glob.glob('../output/008/*.csv')]
sub_df_list2 = [pd.read_csv(path).values for path in glob.glob('../output/014/*.csv')]

sub_df_list.extend(sub_df_list2)

# %%


mean_preds = preds.copy()
mean_preds.iloc[:,:] = np.array(sub_df_list).mean(axis=0)
mean_preds.to_csv(out_dir/'mean_submission_v3.csv', index=False)

# %%

preds

# %%
