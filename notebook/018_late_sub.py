# %%

import gc
import torch.nn.functional as Fj
import copy
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch
from typing import List
from dataclasses import dataclass
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
import sys
sys.path.append('../')


# %%
if 1:
    from src.dataset import RetailDataset, create_target_from_log, only_payment_session_record, create_payment, annot_category
    from src.model_utils import fit_and_predict, create_predict
    from src.utils import timer, savefig
    from src.features.features import CountEncodingBlock, DateBlock, PublicLogBlock, MetaInformationBlock, UserHistoryBlock, SessionPriceBlock
    from src.atma_saint import AtmaTransformer

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


def make_raw_train_paid_log(cartlog, meta):
    session_ids = meta.query("date < '2020-08-01'")["session_id"].unique()
    raw_train_log = cartlog[cartlog['session_id'].isin(session_ids)]

    paid_session_ids = raw_train_log.query("is_payment == 1")[
        "session_id"].unique()
    raw_train_paid_log = cartlog[cartlog['session_id'].isin(paid_session_ids)]

    return raw_train_paid_log


def make_train_target(raw_train_log, train_log, target_category_ids, product_master):
    def agg_payment(cartlog) -> pd.DataFrame:
        """セッションごと・商品ごとの購買個数を集計する"""
        # 購買情報は商品のものだけ.
        target_index = cartlog["kind_1"] == "商品"

        # JANコード (vale_1)ごとに商品の購入個数(n_items)を足し算
        agg = (
            cartlog.loc[target_index]
            .groupby(["session_id", "value_1"])["n_items"]
            .sum()
            .reset_index()
        )
        agg = agg.rename(columns={"value_1": "jan_code"})
        agg = agg.astype({"jan_code": int})
        return agg

    product_master['jan_code'] = product_master['jan_code'].astype(int)
    if "JAN" in product_master.columns:
        product_master = product_master.rename(columns={"JAN": "jan_code"})

    train_session_ids = raw_train_log['session_id'].unique()
    train_target_shape = np.zeros((len(train_session_ids),
                                   len(target_category_ids)))
    train_target = pd.DataFrame(
        train_target_shape,
        index=train_session_ids,
        columns=target_category_ids)
    train_target.index.name = "session_id"

    train_items_per_session_jan = agg_payment(train_log)
    train_items_per_session_target_jan = pd.merge(
        train_items_per_session_jan,
        product_master[["jan_code", "category_id"]],
        on="jan_code",
        how="inner",
    ).query("category_id in @target_category_ids")

    train_target_pos = (
        train_items_per_session_target_jan.groupby(["session_id", "category_id"])[
            "n_items"
        ]
        .sum()
        .unstack()
        .fillna(0)
        .astype(int)
    )

    train_target_pos[train_target_pos > 0] = 1
    train_target_pos[train_target_pos <= 0] = 0

    # return train_target, train_target_pos
    train_target.loc[train_target_pos.index] = train_target_pos.values
    train_target = train_target[target_category_ids]

    return train_target


def split_log_by_spend_time(log, spend_time):
    before_log = log.query(f'spend_time < {spend_time}').reset_index(drop=True)
    after_log = log.query(f'spend_time >= {spend_time}').reset_index(drop=True)
    return before_log, after_log


def make_strange_day(train_target_with_meta, target_category_ids):
    target_sum_day = \
        train_target_with_meta.\
        groupby('date')[target_category_ids].sum().sum(axis=1)

    few_day = target_sum_day[target_sum_day < 500]
    much_day = target_sum_day[target_sum_day > 2000]
    strange_day = pd.concat([much_day, few_day])
    return strange_day


# %%

product_master = dataset.product_master
product_master = product_master.rename(columns={'JAN': 'jan_code'})

for etime in [180, 300, 600]:
    plt.figure()
    raw_train_paid_log = make_raw_train_paid_log(dataset.cartlog, dataset.meta)
    train_log_b, train_log_a = split_log_by_spend_time(
        raw_train_paid_log, etime)

    train_target = make_train_target(raw_train_paid_log,
                                     train_log_a,
                                     dataset.target_category_ids,
                                     product_master)

    train_target_with_meta = pd.merge(dataset.meta,
                                      train_target,
                                      how='inner',
                                      on='session_id')

    strange_day = make_strange_day(
        train_target_with_meta, dataset.target_category_ids)
    print(len(strange_day))

    # 異常な日時を削除
    train_log_b = train_log_b[~train_log_b['date'].isin(strange_day.index)]
    train_log_a = train_log_a[~train_log_a['date'].isin(strange_day.index)]

    meta = dataset.meta
    strange_days_session_ids = meta[meta['date'].isin(
        strange_day.index)]['session_id'].unique()
    del_idx = (train_target.index & strange_days_session_ids)
    train_target.drop(index=del_idx, inplace=True)

    print(len(strange_days_session_ids), len(train_target.index))

    train_log_b.groupby('session_id')['hour'].count().plot.hist(bins=40)
    plt.show()
    break


# %%

jan_cid = product_master[['jan_code', 'category_id']]
jan_cid['jan_code'] = jan_cid['jan_code'].astype(str)

train_log_b = train_log_b.rename(columns={'value_1': 'jan_code'})
train_log_b['jan_code'] = train_log_b['jan_code'].astype(str)

seq_log = pd.merge(train_log_b, jan_cid, how='left', on='jan_code')
seq_log = seq_log[~seq_log['category_id'].isnull()].fillna(-1)
seq_log['category_id'] = seq_log['category_id'].astype(int)


# %%

seq_list = seq_log[['session_id', 'category_id']].groupby('session_id')[
    'category_id'].agg(list)

# %%


@dataclass
class Session:
    category_seq: List[int]
    labels: List[int]


sessions = {}
and_indices = (seq_list.index & train_target.index)
_category = seq_list[and_indices].values
_labels = train_target.loc[and_indices, :].values

for i, session_id in tqdm(enumerate(and_indices)):
    sessions[session_id] = Session(
        category_seq=_category[i],
        labels=_labels[i]
    )


# %%

class TFDataset(Dataset):
    def __init__(self, samples: "dict[int,Session]", min_seq: int = 5, max_seq: int = 64):
        super(TFDataset, self).__init__()
        self.max_seq = max_seq
        self.samples = samples

        self.sess_ids = []
        for sess_id in samples.keys():
            category_ids = samples[sess_id].category_seq
            if len(category_ids) < min_seq:
                continue
            self.sess_ids.append(sess_id)

    def __len__(self):
        return len(self.sess_ids)

    def __getitem__(self, index):
        sess_id = self.sess_ids[index]
        sample = self.samples[sess_id]

        labels = sample.labels
        category_ids = self._crop_and_pad_seq(sample.category_seq)

        return category_ids, labels

    def _crop_and_pad_seq(self, category_ids):
        seq_len = len(category_ids)
        cropped_category_ids = np.zeros(self.max_seq, dtype=int)

        if seq_len >= self.max_seq:
            cropped_category_ids[:] = category_ids[-self.max_seq:]
        else:
            cropped_category_ids[-seq_len:] = category_ids

        return cropped_category_ids


tf_dataset = TFDataset(sessions)
tf_dataset.__getitem__(0)


# %%


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 samples: "dict[int,Session]",
                 val_ratio: float = 0.2,
                 batch_size: int = 512):

        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.samples = samples
        super().__init__()

    def setup(self, min_seq: int = 5, max_seq: int = 64, stage=None):
        train_samples, val_samples = self._split_sessions()
        self.train_dataset = TFDataset(train_samples, min_seq, max_seq)
        self.val_dataset = TFDataset(val_samples, min_seq, max_seq)

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=8,
                          shuffle=shuffle)

    def val_dataloader(self, shuffle=False) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=8,
                          shuffle=shuffle)

    def _split_sessions(self, val_ratio=0.2):
        session_ids = list(self.samples.keys())
        n_sess = len(session_ids)
        n_vals = int(n_sess*val_ratio)

        tr_sess_ids = session_ids[:n_vals]
        val_sess_ids = session_ids[n_vals:]

        tr_samples = {idx: self.samples[idx] for idx in tr_sess_ids}
        val_samples = {idx: self.samples[idx] for idx in val_sess_ids}
        return tr_samples, val_samples


# %%
dm = DataModule(sessions)
dm.setup()

train_dl = dm.train_dataloader()
x, y = next(iter(train_dl))
print(x.shape, y.shape)


# %%


class StackedNMultiHeadAttention(nn.Module):
    def __init__(self, n_stacks, n_dims, n_heads, seq_len, n_multihead=1, dropout=0.2):
        super(StackedNMultiHeadAttention, self).__init__()
        self.n_stacks = n_stacks
        self.n_multihead = n_multihead
        self.n_dims = n_dims
        self.norm_layers = nn.LayerNorm(n_dims)
        # n_stacks has n_multiheads each
        multihead_layer = [nn.ModuleList(n_multihead*[nn.MultiheadAttention(embed_dim=n_dims,
                                                                            num_heads=n_heads,
                                                                            dropout=dropout), ]), ]
        self.multihead_layers = nn.ModuleList(n_stacks*multihead_layer)
        self.ffn = nn.ModuleList(n_stacks*[FFN(n_dims)])
        self.mask = torch.triu(torch.ones(seq_len, seq_len),
                               diagonal=1).to(dtype=torch.bool)

    def forward(self, input_q, input_k, input_v, encoder_output=None, break_layer=None):
        for stack in range(self.n_stacks):
            for multihead in range(self.n_multihead):
                norm_q = self.norm_layers(input_q)
                norm_k = self.norm_layers(input_k)
                norm_v = self.norm_layers(input_v)
                heads_output, _ = self.multihead_layers[stack][multihead](query=norm_q.permute(1, 0, 2),
                                                                          key=norm_k.permute(
                                                                              1, 0, 2),
                                                                          value=norm_v.permute(
                                                                              1, 0, 2),
                                                                          attn_mask=self.mask.to(config.device))
                heads_output = heads_output.permute(1, 0, 2)
                #assert encoder_output != None and break_layer is not None
                if encoder_output != None and multihead == break_layer:
                    assert break_layer <= multihead, " break layer should be less than multihead layers and postive integer"
                    input_k = input_v = encoder_output
                    input_q = input_q + heads_output
                else:
                    input_q = input_q + heads_output
                    input_k = input_k + heads_output
                    input_v = input_v + heads_output
            last_norm = self.norm_layers(heads_output)
            ffn_output = self.ffn[stack](last_norm)
            ffn_output = ffn_output + heads_output
        return ffn_output


class FFN(nn.Module):
    def __init__(self, in_feat):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(in_feat, in_feat)
        self.linear2 = nn.Linear(in_feat, in_feat)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        out = F.relu(self.drop(self.linear1(x)))
        out = self.linear2(out)
        return out


class EncoderEmbedding(nn.Module):
    def __init__(self, n_categories, n_dims, seq_len):
        super(EncoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.category_embed = nn.Embedding(n_categories, n_dims)
        self.position_embed = nn.Embedding(seq_len, n_dims)
        self.dev = 'cpu'

    def forward(self, categories):
        c = self.category_embed(categories)
        seq = torch.arange(self.seq_len, device=self.dev).unsqueeze(0)
        p = self.position_embed(seq)
        return p + c


class AtmaTransformer(pl.LightningModule):
    def __init__(self, n_labels=15, n_cat=772, n_dec=4, n_enc=4, n_head=8, n_emb=64, n_seq=64):
        super(AtmaTransformer, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.encoder_layer = StackedNMultiHeadAttention(n_stacks=n_dec,
                                                        n_dims=n_emb,
                                                        n_heads=n_head,
                                                        seq_len=n_seq,
                                                        n_multihead=1, dropout=0.2)
        self.encoder_embedding = EncoderEmbedding(
            n_categories=n_cat,
            n_dims=n_emb, seq_len=n_seq)

        self.GAP = nn.AdaptiveAvgPool1d((n_emb))
        self.fc = nn.Linear(n_emb, n_labels)
        self.dev = 'cpu'

    def forward(self, x, y):
        enc = self.encoder_embedding(categories=x.long().to(self.dev))
        encoder_output = self.encoder_layer(input_k=enc,
                                            input_q=enc,
                                            input_v=enc)

        out = self.GAP(encoder_output)
        print('out1',out.shape)
        out = self.fc(out)

        print('out2',out.shape)
        return out.squeeze()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 0.0001)

    def training_step(self, batch, batch_ids):
        input, ans, labels = batch
        out = self(input, ans)

        loss = self.loss(out.view(-1).float(), labels.view(-1).float())
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return {"loss": loss, "outs": out, "labels": labels}

    def validation_step(self, batch, batch_ids):
        input, ans, labels = batch
        out = self(input, ans)

        loss = self.loss(out.view(-1).float(), labels.view(-1).float())
        self.log("val_loss", loss, on_step=True, prog_bar=True)
        output = {"outs": out, "labels": labels}
        return output

    def validation_epoch_end(self, validation_ouput):
        out = torch.cat([i["outs"] for i in validation_ouput]).view(-1)
        labels = torch.cat([i["labels"] for i in validation_ouput]).view(-1)
        auc = roc_auc_score(labels.cpu().detach().numpy(),
                            out.cpu().detach().numpy())
        self.print("val auc", auc)


# %%
b, e, c = 8, 64, 15

criterion = nn.BCEWithLogitsLoss()

out = torch.empty([b, c])
ans = torch.ones([b, c])
criterion(out,ans)


# %%
x.max()


#%%

train_dl = dm.train_dataloader()
x, y = next(iter(train_dl))
print(x.shape, y.shape)

atf = AtmaTransformer()
atf.to('cpu')
p = atf(x, y)

#%%
atf = AtmaTransformer()
atf = atf.to('cuda')
x2 = x.to('cuda')
y2 = y.to('cuda')
p = atf(x2, y2)

#%%




# %%
train_target_with_meta = pd.merge(dataset.meta,
                                  train_target,
                                  how='inner',
                                  on='session_id')

fig, ax = plt.subplots(figsize=(20, 10))
train_target_with_meta.samples(
    'date')[dataset.target_category_ids].sum().sum(axis=1).plot(ax=ax)
plt.show()


# %%
if False:
    # 日付ごとのターゲット==1 の数を確認
    fig, ax = plt.subplots(figsize=(20, 10))
    train_target_with_meta.groupby(
        'date')[dataset.target_category_ids].sum().sum(axis=1).plot(ax=ax)
    plt.show()
