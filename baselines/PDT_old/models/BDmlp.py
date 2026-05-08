import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from normailzation.IterNorm_temp import IterNormTemp



class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.embed_dim = configs.d_model
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.revin = configs.revin

        
        if configs.iter_norm:
            self.iter_norm = IterNormSequential(self.embed_dim, num_groups=configs.num_groups, T=configs.T, eps=configs.eps, momentum=configs.momentum, affine=configs.affine)
        else:
            self.iter_norm = None
        
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.channel_embed = nn.Linear(self.channels, self.embed_dim)
        self.channel_Linear = nn.Linear(self.embed_dim, self.channels)

        # scale = 1 / self.seq_len
        # self.Linear.weight = nn.Parameter(scale * torch.ones([self.pred_len, self.embed_dim]))


        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)
        # 新增：用于累积所有batch的init和协方差
        self._init_list = []
        self._cov_list = []

    def encoder(self, x):
        # x shape: [B, L, N]
        # 在N维度进行RevIN标准化
        if self.revin:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True)
            x = (x - mean) / (std + 1e-5)

        if self.iter_norm:
            init = self.iter_norm(x.permute(0, 2, 1)).permute(0, 2, 1)     #  B, L, N
        init = self.channel_embed(init)        # B, L, N -> B, L, D
        init = self.channel_Linear(init)    # B, L, D -> B, L, N
        
        init = init.permute(0, 2, 1)   # [B, N, L]

        output = self.Linear(init)      # B, N, L -> B, N, S

        x = output                      # B, N, S

        # 在N维度进行RevIN逆标准化
        if self.revin:
            x = x.permute(0, 2, 1)
            x = x * (std + 1e-5) + mean
            x = x.permute(0, 2, 1)

        return x.permute(0, 2, 1)   # B, S, N

    

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
