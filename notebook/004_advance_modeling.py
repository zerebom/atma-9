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
pred_categories = [int(i) for i in list(sub_df.columns)]

ses_cat_df = out_df.groupby(['session_id', 'category_id'])['n_items'].sum().unstack().reset_index()
user_cat_df = out_df.groupby(['user_id', 'category_id'])['n_items'].sum().unstack().reset_index()

user_cat_df = user_cat_df[['user_id']+pred_categories]
ses_cat_df = ses_cat_df[['session_id']+pred_categories]

user_cat_df[pred_categories] = user_cat_df[pred_categories].fillna(0).astype(int)
ses_cat_df = ses_cat_df.fillna(0).astype(int)


# %%
ans_df = (ses_cat_df[pred_categories] > 0).astype(int)
ans_df['session_id'] = ses_cat_df['session_id']

# %%

idx = carlog['kind_1'] == '商品'
product_df = carlog[idx].reset_index(drop=True)
user_product_df = product_df.groupby('user_id')
product_df['price'] = product_df['n_items'] * product_df['unit_price']

session_price = product_df.groupby(['session_id','user_id'])['price'].sum().reset_index().groupby('user_id')['price']
session_price_agg_df = session_price.agg(['mean','max','std','sum']).reset_index().fillna(-1)
session_price_agg_df.columns = ['user_id','price_mean','price_max','price_std','price_sum']

user_df = session_price_agg_df.copy()
user_df['n_items_sum'] = user_product_df['n_items'].sum().values
user_df['spend_time_mean'] = user_product_df['spend_time'].mean().values
user_df = user_df.merge(user_master,how='left',on='user_id')


#%%
user_cat_df

#%%

all_data = meta.merge(user_cat_df, how='left', on='user_id')
all_data = all_data.merge(user_df,on='user_id',how='left')

test = all_data[~all_data['time_elapsed'].isnull()]
train = all_data[all_data['time_elapsed'].isnull()]







#%%
train = train.merge(ans_df, on='session_id', how='inner',suffixes=('', '_ans')).fillna(0)
test = test.merge(ans_df, on='session_id', how='left',suffixes=('', '_ans')).fillna(0)

#%%

ys = train.loc[:, train.columns.str.contains('_ans')]
test_ys = test.loc[:, test.columns.str.contains('_ans')]

ans_cols = list(ys.columns)

X = train.drop(
    columns=['session_id', 'user_id', 'date', 'time_elapsed']+ans_cols)
test_X = test.drop(columns=['session_id', 'user_id',
                            'date', 'time_elapsed']+ans_cols)

#%%
import optuna.integration.lightgbm as olgb

#%%



# %%
fold = KFold(n_splits=4, shuffle=True)
pred_ys = test_ys.copy()
oofs = ys.copy()

params = {
    'num_leaves': 48,
    'max_depth': 7,
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

    oofs.iloc[:, i] = oof
    pred = sum(y_preds) / len(y_preds)
    pred[pred < 0] = 0
    pred_ys.iloc[:, i] = pred

# %%
sub_df.iloc[:, :] = pred_ys.values

#%%
test_log = test.loc[:, test.columns.str.contains('_ans')].astype(int).values
#%%
test_log.max()


#%%
over_sub_df = sub_df.copy()
over_sub_df.iloc[:,:] = np.where(test_log == 0, sub_df.values,test_log)


#%%
over_sub_df.to_csv('../output/first_output_v3.csv', index=None)


#%%
sub_df.to_csv('../output/first_output_v2.csv', index=None)




#%%

from sklearn.metrics import roc_auc_score
roc_auc_score(ys.values,oofs.values, average='macro')

# scores = []


# # 各ラベルごとに AUC を計算
# for i in range(oofs.shape[1]):
#     score_i = roc_auc_score(ys.values[:, i], oofs.values[:, i])
#     scores.append(score_i)

# # 平均をとる
# auc_score = sum(scores) / len(scores)
# print(auc_score)
#%%
test_X

#%%

ans_df


#%%
ys.values[:, 0].dtype

#%%
sub_df.head()




# %%

np.min(ys.values)







# %%
print(ys.shape)
print(oofs.shape)

# %%
