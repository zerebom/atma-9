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

meta['time_elapsed'].fillna(-1,inplace=True)

# %%

time_by_u_df = meta.groupby(['user_id','time_elapsed'])['session_id'].count().unstack().fillna(0).astype(int)


time_by_u_df['test_ratio'] =  time_by_u_df.iloc[:,1:].sum(axis=1) / time_by_u_df.iloc[:,0]
time_by_u_df =time_by_u_df.replace([np.inf, -np.inf], 15)
time_by_u_df.head()

# %%
fig, (ax1, ax2) = plt.subplots(1 ,2,figsize=(10, 3))
time_by_u_df.query('test_ratio<1')['test_ratio'].plot.hist(ax=ax1)
time_by_u_df.query('test_ratio>=1')['test_ratio'].plot.hist(ax=ax2)
plt.show()




#%%


time_by_u_df['session_count'] =time_by_u_df.select_dtypes('int').sum(axis=1)


# %%

fig, (ax1, ax2) = plt.subplots(1 ,2,figsize=(10, 3))
sns.scatterplot(x='session_count', y='test_ratio',data=time_by_u_df.query('test_ratio>=1'),ax=ax1)
sns.scatterplot(x='session_count', y='test_ratio',data=time_by_u_df.query('0<test_ratio<1') ,ax=ax2)




#%%
time_by_u_df[time_by_u_df['test_ratio']==15].iloc[:,0:5].sum(axis=1).plot.hist()
print(time_by_u_df[time_by_u_df['test_ratio']==15].iloc[:,0:5].sum().sum())



# %%
time_by_u_df

# %%
meta

# %%
meta[meta['time_elapsed'] != -1.0]

# %%

meta[meta['time_elapsed'] != 0].sort_values('date').head(50)





# %%
