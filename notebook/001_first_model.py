# %%
import gc
import os
import random
import numpy as np
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
sys.path.append('/home/vmlab/higuchi/scorta')

import scorta
from scorta.eda.df import df_info
# %%
input_dir = '../input'
carlog = pd.read_csv(f'{input_dir}/carlog.csv',
                     dtype={'value_1': str}, parse_dates=['date'])

# %%

user_master = pd.read_csv(f'{input_dir}/user_master.csv')
product_master = pd.read_csv(f'{input_dir}/product_master.csv')
display_action = pd.read_csv(f'{input_dir}/display_action_id.csv')
meta = pd.read_csv(f'{input_dir}/meta.csv')
test = pd.read_csv(f'{input_dir}/test.csv')
sub_df = pd.read_csv(f'{input_dir}/atmaCup#9__sample_submission.csv')


df_list = [carlog, user_master, product_master,
           display_action, meta, test, sub_df]



# %%
idx = carlog['kind_1'] == '商品'
out_df = carlog[idx].reset_index(drop=True)
out_df = out_df.groupby(['session_id', 'value_1'])['n_items'].sum().reset_index()
out_df = out_df.rename(columns={'value_1': 'JAN'})


out_df['JAN'] = out_df['JAN'].astype(np.int64)
out_df = out_df.merge(product_master[['JAN', 'category_id']], on='JAN')
out_df = out_df.merge(meta, on='session_id')
out_df.head()

# %%
ses_cat_df = out_df.groupby(['session_id', 'category_id'])['n_items'].sum().unstack().reset_index()
user_cat_df = out_df.groupby(['user_id', 'category_id'])['n_items'].sum().unstack().reset_index()

# %%
pred_categories = [int(i) for i in list(sub_df.columns)]
user_cat_df = user_cat_df[['user_id']+pred_categories]
ses_cat_df = ses_cat_df[['session_id']+pred_categories]
ses_cat_df = ses_cat_df.fillna(0)


# %%
ans_df = (ses_cat_df.select_dtypes('float') > 0).astype(int)
ans_df['session_id'] = ses_cat_df['session_id']

# %%

train = meta.merge(user_cat_df, how='left', on='user_id')
test = train[~train['time_elapsed'].isnull()]
train = train[train['time_elapsed'].isnull()]


train = train.merge(ans_df, on='session_id', how='inner',suffixes=('', '_ans')).fillna(0)
test = test.merge(ans_df, on='session_id', how='left',suffixes=('', '_ans')).fillna(0)


# %%
ys = train.loc[:, train.columns.str.contains('_ans')]
test_ys = test.loc[:, test.columns.str.contains('_ans')]

ans_cols = list(ys.columns)

X = train.drop(
    columns=['session_id', 'user_id', 'date', 'time_elapsed']+ans_cols)
test_X = test.drop(columns=['session_id', 'user_id',
                            'date', 'time_elapsed']+ans_cols)



# %%
fold = KFold(n_splits=4, shuffle=True)
pred_ys = test_ys.copy()

params = {
    'num_leaves': 24,
    'max_depth': 6,
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05
}

for i in range(ys.shape[1]):
    y = ys.iloc[:, i]

    categorical_cols = list(train.select_dtypes('category').columns)
    oof = np.zeros(shape=(len(y)), dtype=np.float)
    models = []
    y_preds = []

    for fold_idx, (idx_tr, idx_val) in enumerate(fold.split(X)):
        train_X, val_X = X.loc[idx_tr, :], X.loc[idx_val, :]
        train_y, val_y = y[idx_tr], y[idx_val]

        lgb_train = lgb.Dataset(train_X,
                                train_y,
                                categorical_feature=categorical_cols)

        lgb_eval = lgb.Dataset(val_X,
                               val_y,
                               reference=lgb_train,
                               categorical_feature=categorical_cols)

        model = lgb.train(params, lgb_train, valid_sets=[
                          lgb_train, lgb_eval], verbose_eval=50, num_boost_round=1000, early_stopping_rounds=50)

        oof[idx_val] = model.predict(val_X, num_iteration=model.best_iteration)
        y_pred = model.predict(test_X)

        y_preds.append(y_pred)
        models.append(model)

    pred = sum(y_preds) / len(y_preds)
    pred[pred < 0] = 0
    pred_ys.iloc[:, i] = pred

# %%
sub_df.iloc[:, :] = pred_ys.values

# %%
sub_df.to_csv('../output/first_output.csv', index=None)


