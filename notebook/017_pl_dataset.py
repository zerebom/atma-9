# %%
from pathlib import Path
import xgboost as xgb
from catboost import Pool
from catboost import CatBoost
from lightgbm import LGBMClassifier
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgbm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from src.dataset import RetailDataset, create_target_from_log, only_payment_session_record, create_payment, annot_category
from src.model_utils import fit_and_predict, create_predict
from src.utils import timer, savefig
from src.features.features import CountEncodingBlock, DateBlock, PublicLogBlock, MetaInformationBlock, UserHistoryBlock, SessionPriceBlock
import sys
sys.path.append('../')


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
print('a')

# %%
