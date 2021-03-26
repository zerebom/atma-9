import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch import nn
import copy
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import gc


class config:
    device = torch.device("cuda")
    MAX_SEQ = 100
    EMBED_DIMS = 512
    ENC_HEADS = DEC_HEADS = 8
    NUM_ENCODER = NUM_DECODER = 4
    BATCH_SIZE = 64
    TRAIN_FILE = "../input/riiid-test-answer-prediction/train.csv"
    TOTAL_EXE = 13523
    TOTAL_CAT = 10000


class DKTDataset(Dataset):
    def __init__(self, samples, max_seq, start_token=0):
        super().__init__()
        self.samples = samples
        self.max_seq = max_seq
        self.start_token = start_token
        self.data = []
        for id in self.samples.index:
            exe_ids, answers, ela_time, categories = self.samples[id]
            if len(exe_ids) > max_seq:
                for l in range((len(exe_ids)+max_seq-1)//max_seq):
                    self.data.append(
                        (exe_ids[l:l+max_seq], answers[l:l+max_seq], ela_time[l:l+max_seq], categories[l:l+max_seq]))
            elif len(exe_ids) < self.max_seq and len(exe_ids) > 10:
                self.data.append((exe_ids, answers, ela_time, categories))
            else:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_ids, answers, ela_time, exe_category = self.data[idx]
        seq_len = len(question_ids)

        exe_ids = np.zeros(self.max_seq, dtype=int)
        ans = np.zeros(self.max_seq, dtype=int)
        elapsed_time = np.zeros(self.max_seq, dtype=int)
        exe_cat = np.zeros(self.max_seq, dtype=int)
        if seq_len < self.max_seq:
            exe_ids[-seq_len:] = question_ids
            ans[-seq_len:] = answers
            elapsed_time[-seq_len:] = ela_time
            exe_cat[-seq_len:] = exe_category
        else:
            exe_ids[:] = question_ids[-self.max_seq:]
            ans[:] = answers[-self.max_seq:]
            elapsed_time[:] = ela_time[-self.max_seq:]
            exe_cat[:] = exe_category[-self.max_seq:]

        input_rtime = np.zeros(self.max_seq, dtype=int)
        input_rtime = np.insert(elapsed_time, 0, self.start_token)
        input_rtime = np.delete(input_rtime, -1)

        input = {"input_ids": exe_ids, "input_rtime": input_rtime.astype(
            np.int), "input_cat": exe_cat}
        answers = np.append([0], ans[:-1])  # start token
        assert ans.shape[0] == answers.shape[0] and answers.shape[0] == input_rtime.shape[0], "both ans and label should be \
                                                                                            same len with start-token"
        return input, answers, ans


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
    def __init__(self, n_exercises, n_categories, n_dims, seq_len):
        super(EncoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.exercise_embed = nn.Embedding(n_exercises, n_dims)
        self.category_embed = nn.Embedding(n_categories, n_dims)
        self.position_embed = nn.Embedding(seq_len, n_dims)

    def forward(self, exercises, categories):
        e = self.exercise_embed(exercises)
        c = self.category_embed(categories)
        seq = torch.arange(self.seq_len, device=config.device).unsqueeze(0)
        p = self.position_embed(seq)
        return p + c + e


class DecoderEmbedding(nn.Module):
    def __init__(self, n_responses, n_dims, seq_len):
        super(DecoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.response_embed = nn.Embedding(n_responses, n_dims)
        self.time_embed = nn.Linear(1, n_dims, bias=False)
        self.position_embed = nn.Embedding(seq_len, n_dims)

    def forward(self, responses):
        e = self.response_embed(responses)
        seq = torch.arange(self.seq_len, device=config.device).unsqueeze(0)
        p = self.position_embed(seq)
        return p + e


# layers of encoders stacked onver, multiheads-block in each encoder is n.
# Stacked N MultiheadAttentions
class StackedNMultiHeadAttention(nn.Module):
    def __init__(self, n_stacks, n_dims, n_heads, seq_len, n_multihead=1, dropout=0.2):
        super(StackedNMultiHeadAttention, self).__init__()
        self.n_stacks = n_stacks
        self.n_multihead = n_multihead
        self.n_dims = n_dims
        self.norm_layers = nn.LayerNorm(n_dims)
        # n_stacks has n_multiheads each
        self.multihead_layers = nn.ModuleList(n_stacks*[nn.ModuleList(n_multihead*[nn.MultiheadAttention(embed_dim=n_dims,
                                                                                                         num_heads=n_heads,
                                                                                                         dropout=dropout), ]), ])
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


class PlusSAINTModule(pl.LightningModule):
    def __init__(self, n_dec=4, n_enc=4, n_head=8, n_emb=64, n_seq=64):
        super(PlusSAINTModule, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.encoder_layer = StackedNMultiHeadAttention(n_stacks=n_dec,
                                                        n_dims=n_emb,
                                                        n_heads=n_head,
                                                        seq_len=n_seq,
                                                        n_multihead=1, dropout=0.2)
        self.decoder_layer = StackedNMultiHeadAttention(n_stacks=config.NUM_ENCODER,
                                                        n_dims=n_emb,
                                                        n_heads=n_head,
                                                        seq_len=n_seq,
                                                        n_multihead=2, dropout=0.2)
        self.encoder_embedding = EncoderEmbedding(n_exercises=config.TOTAL_EXE,
                                                  n_categories=config.TOTAL_CAT,
                                                  n_dims=n_emb, seq_len=n_seq)
        self.decoder_embedding = DecoderEmbedding(
            n_responses=3, n_dims=n_emb, seq_len=n_seq)
        self.elapsed_time = nn.Linear(1, n_emb)
        self.fc = nn.Linear(n_emb, 1)

    # TODO: implement embdding layer and its output
    def forward(self, x, y):
        enc = self.encoder_embedding(exercises=x["input_ids"].long().to(
            config.device), categories=x['input_cat'].long().to('cuda'))
        dec = self.decoder_embedding(responses=y.long().to('cuda'))
        # elapsed_time = x["input_rtime"].unsqueeze(-1).float()
        # ela_time = self.elapsed_time(elapsed_time)

        # this encoder
        encoder_output = self.encoder_layer(input_k=enc,
                                            input_q=enc,
                                            input_v=enc)
        #this is decoder
        decoder_output = self.decoder_layer(input_k=dec,
                                            input_q=dec,
                                            input_v=dec,
                                            encoder_output=encoder_output,
                                            break_layer=1)
        # fully connected layer
        out = self.fc(decoder_output)
        return out.squeeze()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 0.0001)

    def training_step(self, batch, batch_ids):
        input, ans, labels = batch
        target_mask = (input["input_ids"] != 0)
        out = self(input, ans)
        loss = self.loss(out.view(-1).float(), labels.view(-1).float())
        out = torch.masked_select(out, target_mask)
        out = torch.sigmoid(out)
        labels = torch.masked_select(labels, target_mask)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return {"loss": loss, "outs": out, "labels": labels}

    def validation_step(self, batch, batch_ids):
        input, ans, labels = batch
        target_mask = (input["input_ids"] != 0)
        out = self(input, ans)
        loss = self.loss(out.view(-1).float(), labels.view(-1).float())
        out = torch.masked_select(out, target_mask)
        out = torch.sigmoid(out)
        labels = torch.masked_select(labels, target_mask)
        self.log("val_loss", loss, on_step=True, prog_bar=True)
        output = {"outs": out, "labels": labels}
        return output

    def validation_epoch_end(self, validation_ouput):
        out = torch.cat([i["outs"] for i in validation_ouput]).view(-1)
        labels = torch.cat([i["labels"] for i in validation_ouput]).view(-1)
        auc = roc_auc_score(labels.cpu().detach().numpy(),
                            out.cpu().detach().numpy())
        self.print("val auc", auc)


class AtmaTransfomer(pl.LightningModule):
    def __init__(self, n_dec=4, n_enc=4, n_head=8, n_emb=64, n_seq=64):
        super(AtmaTransfomer, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.encoder_layer = StackedNMultiHeadAttention(n_stacks=n_dec,
                                                        n_dims=n_emb,
                                                        n_heads=n_head,
                                                        seq_len=n_seq,
                                                        n_multihead=1, dropout=0.2)
        self.decoder_layer = StackedNMultiHeadAttention(n_stacks=config.NUM_ENCODER,
                                                        n_dims=n_emb,
                                                        n_heads=n_head,
                                                        seq_len=n_seq,
                                                        n_multihead=2, dropout=0.2)
        self.encoder_embedding = EncoderEmbedding(n_exercises=config.TOTAL_EXE,
                                                  n_categories=config.TOTAL_CAT,
                                                  n_dims=n_emb, seq_len=n_seq)
        self.decoder_embedding = DecoderEmbedding(
            n_responses=3, n_dims=n_emb, seq_len=n_seq)
        self.elapsed_time = nn.Linear(1, n_emb)
        self.fc = nn.Linear(n_emb, 1)

    # TODO: implement embdding layer and its output
    def forward(self, x, y):
        enc = self.encoder_embedding(exercises=x["input_ids"].long().to(
            'cuda'), categories=x['input_cat'].long().to('cuda'))
        dec = self.decoder_embedding(responses=y.long().to('cuda'))
        elapsed_time = x["input_rtime"].unsqueeze(-1).float()
        ela_time = self.elapsed_time(elapsed_time)
        dec = dec + ela_time
        # this encoder
        encoder_output = self.encoder_layer(input_k=enc,
                                            input_q=enc,
                                            input_v=enc)
        #this is decoder
        decoder_output = self.decoder_layer(input_k=dec,
                                            input_q=dec,
                                            input_v=dec,
                                            encoder_output=encoder_output,
                                            break_layer=1)
        # fully connected layer
        out = self.fc(decoder_output)
        return out.squeeze()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 0.0001)

    def training_step(self, batch, batch_ids):
        input, ans, labels = batch
        target_mask = (input["input_ids"] != 0)
        out = self(input, ans)
        loss = self.loss(out.view(-1).float(), labels.view(-1).float())
        out = torch.masked_select(out, target_mask)
        out = torch.sigmoid(out)
        labels = torch.masked_select(labels, target_mask)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return {"loss": loss, "outs": out, "labels": labels}

    def validation_step(self, batch, batch_ids):
        input, ans, labels = batch
        target_mask = (input["input_ids"] != 0)
        out = self(input, ans)
        loss = self.loss(out.view(-1).float(), labels.view(-1).float())
        out = torch.masked_select(out, target_mask)
        out = torch.sigmoid(out)
        labels = torch.masked_select(labels, target_mask)
        self.log("val_loss", loss, on_step=True, prog_bar=True)
        output = {"outs": out, "labels": labels}
        return output

    def validation_epoch_end(self, validation_ouput):
        out = torch.cat([i["outs"] for i in validation_ouput]).view(-1)
        labels = torch.cat([i["labels"] for i in validation_ouput]).view(-1)
        auc = roc_auc_score(labels.cpu().detach().numpy(),
                            out.cpu().detach().numpy())
        self.print("val auc", auc)


def get_dataloaders():
    dtypes = {'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16',
              'answered_correctly': 'int8', "content_type_id": "int8",
              "prior_question_elapsed_time": "float32", "task_container_id": "int16"}
    print("loading csv.....")
    train_df = pd.read_csv(config.TRAIN_FILE, usecols=[
                           1, 2, 3, 4, 5, 7, 8], dtype=dtypes)
    print("shape of dataframe :", train_df.shape)

    train_df = train_df[train_df.content_type_id == 0]
    train_df.prior_question_elapsed_time.fillna(0, inplace=True)
    train_df.prior_question_elapsed_time /= 3600
    # train_df.prior_question_elapsed_time.clip(lower=0,upper=300,inplace=True)
    train_df.prior_question_elapsed_time = train_df.prior_question_elapsed_time.astype(
        np.int)

    train_df = train_df.sort_values(
        ["timestamp"], ascending=True).reset_index(drop=True)
    n_skills = train_df.content_id.nunique()
    print("no. of skills :", n_skills)
    print("shape after exlusion:", train_df.shape)

    # grouping based on user_id to get the data supplu
    print("Grouping users...")
    group = train_df[["user_id", "content_id", "answered_correctly", "prior_question_elapsed_time", "task_container_id"]]\
        .groupby("user_id")\
        .apply(lambda r: (r.content_id.values, r.answered_correctly.values,
                          r.prior_question_elapsed_time.values, r.task_container_id.values))
    del train_df
    gc.collect()
    print("splitting")
    train, val = train_test_split(group, test_size=0.2)
    print("train size: ", train.shape, "validation size: ", val.shape)
    train_dataset = DKTDataset(train, max_seq=config.MAX_SEQ)
    val_dataset = DKTDataset(val, max_seq=config.MAX_SEQ)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              num_workers=8,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            num_workers=8,
                            shuffle=False)
    del train_dataset, val_dataset
    gc.collect()
    return train_loader, val_loader


train_loader, val_loader = get_dataloaders()
