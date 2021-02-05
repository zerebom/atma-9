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

from .utils import timer,savefig

#%%


def fit_lgbm(X, y, cv=None, params: dict=None, verbose=500):
    # パラメータがないときはデフォルトパラメータを使う
    if params is None:
        params = deepcopy(LGBM_DEFAULT_PARAMS)

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

        clf = lgbm.LGBMClassifier(**params)

        with timer(prefix='fit fold={} '.format(i + 1)):
            clf.fit(x_train, y_train,
                    eval_set=[(x_valid, y_valid)],
                    early_stopping_rounds=100,
                    eval_metric='logloss',
                    verbose=verbose)

        pred_i = clf.predict_proba(x_valid)[:, 1]
        oof_pred[idx_valid] = pred_i
        models.append(clf)

        print(f'Fold {i} AUC: {roc_auc_score(y_valid, pred_i):.4f}')

    score = roc_auc_score(y, oof_pred)
    print('FINISHED \ whole score: {:.4f}'.format(score))
    return oof_pred, models, score