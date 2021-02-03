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
from pathlib import Path
sys.path.append('../')
from src.dataset import RetailDataset

file_path = Path('../input')
dataset = RetailDataset(file_path,thres_sec=60)

# %%
train_y = dataset.get_train_target()

train_input_log = dataset.get_train_input_log()
test_input_log = dataset.get_test_input_log()
test_sessions = dataset.get_test_sessions()


# %%
#セッションごとの集計

def make_session_feature(input_log):
    n_actions = input_log.groupby(["session_id"]).size().rename("n_actions")
    n_add_items = input_log.query("kind_1 == '商品'").groupby(["session_id"]).size().rename("n_add_items")
    mean_spend_time = input_log.groupby(["session_id"])["spend_time"].mean()

    session_features = pd.concat([
        n_actions,
        n_add_items,
        mean_spend_time,
    ], axis=1)

    session_features.head()
    return session_features

# %%
## ユーザーごとのデータ
user_features = pd.merge(
    dataset.meta[["session_id", "user_id"]],
    dataset.user_master,
    on="user_id",
    how="left",
).drop(columns=["user_id"])
user_features.head()


# %%

train_session_ids = train_y.index
train_features = pd.DataFrame({"session_id": train_session_ids})
train_features = pd.merge(train_features, session_features, on="session_id", how="left")
train_features = pd.merge(train_features, user_features, on="session_id", how="left")

train_features.head()

# %%

train_features

# %%

# データ作成
# input_dir = '../input'
# carlog = pd.read_csv(f'{input_dir}/carlog.csv',
#                      dtype={'value_1': str}, parse_dates=['date'])


# user_master = pd.read_csv(f'{input_dir}/user_master.csv')
# product_master = pd.read_csv(f'{input_dir}/product_master.csv')
# display_action = pd.read_csv(f'{input_dir}/display_action_id.csv')
# meta = pd.read_csv(f'{input_dir}/meta.csv')
# test = pd.read_csv(f'{input_dir}/test.csv')
# sub_df = pd.read_csv(f'{input_dir}/atmaCup#9__sample_submission.csv')

# meta['time_elapsed_sec'] = meta['time_elapsed'] * 60
# product_master =product_master.rename(columns={"JAN": "jan_code"})

# df_list = {"cartlog":carlog, "user_master":user_master, "product_master":product_master,
#            "display_action":display_action, "meta":meta, "test":test, "sub_df":sub_df}

# for name,df in df_list.items():
#     df.to_pickle(f'../input/{name}.pkl')