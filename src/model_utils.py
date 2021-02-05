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



TARGET_CATEGORIES = [
    # お酒に関するもの
    'ビール系__RTD', 'ビール系__ビール系', 'ビール系__ノンアルコール',

    # お菓子に関するもの
    'スナック・キャンディー__スナック',
    'チョコ・ビスクラ__チョコレート',
    'スナック・キャンディー__ガム',
    'スナック・キャンディー__シリアル',
    'アイスクリーム__ノベルティー',
    '和菓子__米菓',

    # 飲料に関するもの
    '水・炭酸水__大型PET（炭酸水）',
    '水・炭酸水__小型PET（炭酸水）',
    '缶飲料__コーヒー（缶）',
    '小型PET__コーヒー（小型PET）',
    '大型PET__無糖茶（大型PET）',

    # 麺類
    '麺類__カップ麺',
]
INPUT_DIR = '../input/'
product_master_df = pd.read_csv(os.path.join(INPUT_DIR, 'product_master.csv'), dtype={ 'JAN': str })
cat2id = dict(zip(product_master_df['category_name'], product_master_df['category_id']))
TARGET_IDS = pd.Series(TARGET_CATEGORIES).map(cat2id).values.tolist()
category_id2code = dict(zip(TARGET_IDS, TARGET_CATEGORIES))

LGBM_DEFAULT_PARAMS = {
    'objective': 'binary',
    'learning_rate': .1,
    'max_depth': 6,
    'n_estimators': 1000,
    'colsample_bytree': .7,
    'importance_type': 'gain'
}

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



def fit_and_predict(train_df,
                    target_df,
                    target_id,
                    params=None,
                    cv=None):
    """対象の target_id の学習を行なう"""
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
    oof, models, _ = fit_lgbm(train_df.values, y, cv=cv, verbose=500,params=params)

    # 特徴重要度の可視化
    fig, ax, feature_importance_df = visualize_importance(models, train_df)
    ax.set_title('Importance: TARGET={}'.format(target_name))
    fig.tight_layout()
    savefig(fig, to=f'{target_name}_importance')
    plt.close(fig)

    return oof, models, feature_importance_df


def visualize_importance(models, feat_train_df):
    """lightGBM の model 配列の feature importance を plot する
    CVごとのブレを boxen plot として表現します.

    args:
        models:
            List of lightGBM models
        feat_train_df:
            学習時に使った DataFrame
    """
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df['feature_importance'] = model.feature_importances_
        _df['column'] = feat_train_df.columns
        _df['fold'] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

    order = feature_importance_df.groupby('column')\
        .sum()[['feature_importance']]\
        .sort_values('feature_importance', ascending=False).index[:100]

    fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .2)))
    ax.set_facecolor('white')
    sns.boxenplot(data=feature_importance_df, y='column', x='feature_importance',
                  orient='h',
                  order=order, ax=ax, palette='viridis')

    ax.tick_params(axis='x', rotation=90)
    ax.grid()
    fig.tight_layout()
    return fig, ax,feature_importance_df


def create_predict(models, input_df) -> np.ndarray:
    """与えられた機械学習モデルで予測する"""
    pred = np.array([model.predict_proba(input_df.values)[:, 1] for model in models])
    pred = np.mean(pred, axis=0)
    return pred

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