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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import gc


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

    def forward(self, categories):
        c = self.category_embed(categories)
        seq = torch.arange(self.seq_len, device='cuda').unsqueeze(0)
        p = self.position_embed(seq)
        return p + c


# class DecoderEmbedding(nn.Module):
#     def __init__(self, n_responses, n_dims, seq_len):
#         super(DecoderEmbedding, self).__init__()
#         self.n_dims = n_dims
#         self.seq_len = seq_len
#         self.response_embed = nn.Embedding(n_responses, n_dims)
#         self.time_embed = nn.Linear(1, n_dims, bias=False)
#         self.position_embed = nn.Embedding(seq_len, n_dims)

#     def forward(self, responses):
#         e = self.response_embed(responses)
#         seq = torch.arange(self.seq_len, device='cuda').unsqueeze(0)
#         p = self.position_embed(seq)
#         return p + e


class AtmaTransformer(pl.LightningModule):
    def __init__(self,n_labels=15, n_cat=771,n_dec=4, n_enc=4, n_head=8, n_emb=64, n_seq=64):
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

        # self.decoder_embedding = DecoderEmbedding(
        #     n_responses=3, n_dims=n_emb, seq_len=n_seq)

        self.fc = nn.Linear(n_emb, n_labels)

    # TODO: implement embdding layer and its output
    def forward(self, x, y):
        enc = self.encoder_embedding(categories=x.long().to('cuda'))
        encoder_output = self.encoder_layer(input_k=enc,
                                            input_q=enc,
                                            input_v=enc)

        out = self.GAP(encoder_output)
        # fully connected layer
        out = self.fc(out)
        return out.squeeze()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 0.0001)

    def training_step(self, batch, batch_ids):
        input, ans, labels = batch
        # target_mask = (input["input_ids"] != 0)
        out = self(input, ans)
        loss = self.loss(out.view(-1).float(), labels.view(-1).float())
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return {"loss": loss, "outs": out, "labels": labels}

    def validation_step(self, batch, batch_ids):
        input, ans, labels = batch
        out = self(input, ans)
        loss = self.loss(out.view(-1).float(), labels.view(-1).float())
        out = torch.sigmoid(out)

        self.log("val_loss", loss, on_step=True, prog_bar=True)
        output = {"outs": out, "labels": labels}
        return output

    def validation_epoch_end(self, validation_ouput):
        out = torch.cat([i["outs"] for i in validation_ouput]).view(-1)
        labels = torch.cat([i["labels"] for i in validation_ouput]).view(-1)
        auc = roc_auc_score(labels.cpu().detach().numpy(),
                            out.cpu().detach().numpy())
        self.print("val auc", auc)
