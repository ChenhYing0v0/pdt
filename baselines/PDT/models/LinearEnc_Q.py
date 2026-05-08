import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
# from layers.Transformer_EncDec import Encoder_ori, LinearEncoder
from layers.My_EncDec import Encoder_ori, LinearEncoder
from normailzation.IterNormTempAuto import IterNormTempAuto
from normailzation.ItNormWhiteMatCal import ItNormWhiteMatCal


import sys


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # channels
        self.seq_len = configs.seq_len
        self.hidden_size = self.d_model = configs.d_model  # hidden_size
        self.d_ff = configs.d_ff  # d_ff
        self.T = configs.T
        self.concat = configs.concat
        self.Q_mode = configs.Q_mode
        self.alpha = configs.alpha
        self.embed_size = configs.embed_size  # embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.pred_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )

        # for final input and output
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

        # #############  transformer related  #########
        self.encoder = Encoder_ori(
            [
                LinearEncoder(
                    d_model=configs.d_model, d_ff=configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        # if self.concat:
        #     self.ortho_trans = nn.Sequential(
        #         nn.Linear(self.seq_len * self.embed_size * 2, self.d_model),
        #         self.encoder,
        #         nn.Linear(self.d_model, self.pred_len * self.embed_size)
        #     )
        # else:
        self.ortho_trans = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_model),
            self.encoder,
            nn.Linear(self.d_model, self.pred_len * self.embed_size)
        )

        if self.Q_mode == 'ItrNorm':
            self.norm_layer = IterNormTempAuto(
                seq_len=self.seq_len,
                T=self.T,
                eps=configs.eps,
                momentum=configs.momentum,
                affine=configs.affine,
            )
        elif self.Q_mode == 'lw':
            self.norm_layer = ItNormWhiteMatCal(
                seq_len=self.seq_len,
                T=self.T,
                eps=configs.eps
            )
            # 初始化一个可学习的tensor，与wm相同的尺寸（1，L，L）
            self.lw = nn.Parameter(torch.randn(1, self.seq_len, self.seq_len), requires_grad=True)
        self.catLinear = nn.Linear(self.seq_len * 2, self.seq_len)
        self.catlayernorm = nn.LayerNorm(self.seq_len)


    # dimension extension
    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        # x: [B, T, N] --> [B, N, T]
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        # B*N*T*1 x 1*D = B*N*T*D
        return x * embeddings

    def Fre_Trans(self, x):
        # [B, N, T, D]
        B, N, T, D = x.shape
        assert T == self.seq_len
        # [B, N, D, T]
        x = x.transpose(-1, -2)
        
        if self.Q_mode == 'ItrNorm':
            # Whitening
            x_w = x.reshape(B, N*D, T)  # [B, N*D, T]
            x_w = self.norm_layer(x_w)  # [B, N*D, T]
            x_w = x_w.reshape(B, N, D, T)  # [B, N, D, T]
        elif self.Q_mode == 'lw':
            x_ = x.reshape(B, N*D, T)  # [B, N*D, T]
            wm = self.norm_layer(x_)     # (1,L,L)
            # lw_new = self.alpha * wm + self.lw * (1-self.alpha)
            lw_new = self.lw
            x_reshaped = x_.permute(2, 0, 1).contiguous().view(1, self.seq_len, B*N*D)
            mean = x_reshaped.mean(-1, keepdim=True)
            xc = x_reshaped - mean      # (1,L,B*N*D)
            x_w = lw_new.matmul(xc)         # # (1,L,B*N*D)
            # 将 x_w 从 [1, L, B*N*D] reshape 为 [B, N, D, L]
            x_w = x_w.view(self.seq_len, B, N, D).permute(1, 2, 3, 0).contiguous()  # [B, N, D, L]


        if self.concat == 1:
            x = torch.cat([x, x_w], dim=-1) # [B, N, D, 2T]
            x = self.catLinear(x)
            x = self.catlayernorm(x)
        elif self.concat == 2:
            x = x+x_w
            x = self.catlayernorm(x)
        else:
            x = x_w

        # ########## transformer ####
        x = self.ortho_trans(x.flatten(-2)).reshape(B, N, D, self.pred_len)


        # [B, N, tau, D]
        x = x.transpose(-1, -2)
        return x

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape

        # revin norm
        x = self.revin_layer(x, mode='norm')
        x_ori = x

        # ###########  frequency (high-level) part ##########
        # input fre fine-tuning
        # [B, T, N]
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x_ori, self.embeddings)
        # [B, N, tau, D]
        x = self.Fre_Trans(x)

        # linear
        # [B, N, tau*D] --> [B, N, dim] --> [B, N, tau] --> [B, tau, N]
        out = self.fc(x.flatten(-2)).transpose(-1, -2)

        # dropout
        out = self.dropout(out)

        # revin denorm
        out = self.revin_layer(out, mode='denorm')

        return out
