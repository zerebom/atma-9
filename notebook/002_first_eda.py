#%%
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

#%%

df_list = {"carlog":carlog, "user_master":user_master, "product_master":product_master,
           "display_action":display_action, "meta":meta, "test":test, "sub_df":sub_df}



# %%
for df in df_list:
    display(df_info(df))

# %%

for name,df in df_list.items():
    df[:5000].to_csv(f'../input/mini_data/{name}.csv',index=None)

# %%
