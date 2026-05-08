import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
from layers.Transformer_EncDec import Encoder_ori, LinearEncoder
# from layers.SelfAttention_Family import AttentionLayer, EnhancedAttention

import sys


"""
ROLinear:
Q_in before + R + no_freeze_R + no delta2
"""

def _load_npy(path, root_path=None, device='cpu'):
    p = path if os.path.isfile(path) else os.path.join(root_path or '', path)
    assert os.path.isfile(p), f'File not found: {path}'
    arr = np.load(p)
    return torch.from_numpy(arr).to(torch.float32).to(device)




class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # channels
        self.seq_len = configs.seq_len
        self.hidden_size = self.d_model = configs.d_model  # hidden_size
        self.d_ff = configs.d_ff  # d_ff

        self.Q_chan_indep = configs.Q_chan_indep

        q_path = configs.Q_MAT_file if self.Q_chan_indep else configs.q_mat_file
        # print(q_mat_dir)
        if not os.path.isfile(q_path):
            q_path = os.path.join(configs.root_path, q_path)
        # print(q_mat_dir)
        assert os.path.isfile(q_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        Q_in = _load_npy(q_path, device=device)  # [L,r] 或 [N,L,r]
        if self.Q_chan_indep:
            assert Q_in.ndim == 3 and Q_in.shape[0] == self.enc_in and Q_in.shape[1] == self.seq_len
            self.r = Q_in.shape[2]
        else:
            assert Q_in.ndim == 2 and Q_in.shape[0] == self.seq_len
            self.r = Q_in.shape[1]
        self.register_buffer('Q_in', Q_in)

        # R
        r_path = configs.R_MAT_file if self.Q_chan_indep else configs.r_mat_file
        if not os.path.isfile(r_path):
            r_path = os.path.join(configs.root_path, r_path)
        R = _load_npy(r_path, device=device)      # [r,H] 或 [N,r,H]
        if self.Q_chan_indep:
            assert R.ndim == 3 and R.shape[0] == self.enc_in and R.shape[1] == self.r and R.shape[2] == self.pred_len
        else:
            assert R.ndim == 2 and R.shape[0] == self.r and R.shape[1] == self.pred_len
        
        self.freeze_R = configs.freeze_R
        if self.freeze_R:
            self.register_buffer('R_fix', R)
        else:
            self.R_param = nn.Parameter(R.clone())
        
        # 先走freeze_R的版本
        # self.register_buffer('R_fix', R)
        q_out_mat_dir = configs.Q_OUT_MAT_file if self.Q_chan_indep else configs.q_out_mat_file
        if not os.path.isfile(q_out_mat_dir):
            q_out_mat_dir = os.path.join(configs.root_path, q_out_mat_dir)
        assert os.path.isfile(q_out_mat_dir)
        self.Q_out_mat = torch.from_numpy(np.load(q_out_mat_dir)).to(torch.float32).to(device)

        assert (self.Q_out_mat.ndim == 3 if self.Q_chan_indep else self.Q_out_mat.ndim == 2)
        assert (self.Q_out_mat.shape[0] == self.enc_in if self.Q_chan_indep else
                self.Q_out_mat.shape[0] == self.pred_len)

        self.patch_len = configs.temp_patch_len
        self.stride = configs.temp_stride

        # self.channel_independence = configs.channel_independence
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
                    d_model=configs.d_model, d_ff=configs.d_ff, CovMat=None,
                    dropout=configs.dropout, activation=configs.activation, token_num=self.enc_in,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            one_output=True,
            CKA_flag=configs.CKA_flag
        )
        self.ortho_trans = nn.Sequential(
            nn.Linear(self.r * self.embed_size, self.d_model),
            self.encoder,
            nn.Linear(self.d_model, self.r * self.embed_size)
        )

        # learnable delta
        # self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.r))
        self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, self.r)) 
        self.delta2 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.pred_len))

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
        # [B, N, r, D]
        B, N, r, D = x.shape
        assert r == self.r
        # [B, N, D, r]
        x = x.transpose(-1, -2)

        x_trans = x

        # orthogonal transformation
        # [B, N, D, T]
        # if self.Q_chan_indep:
        #     x_trans = torch.einsum('bndt,ntr->bndr', x, self.Q_in)
        # else:
        #     x_trans = torch.einsum('bndt,tr->bndr', x, self.Q_in) + self.delta1
        #     # x_trans = x + self.delta1
        #     # added on 25/1/30
        #     # x_trans = F.gelu(x_trans)
        #     # [B, N, D, r]
        # assert x_trans.shape[-1] == self.r

        # ########## transformer ####
        x_trans = self.ortho_trans(x_trans.flatten(-2)).reshape(B, N, D, self.r)

        # 暂时先用Identity
        if self.freeze_R:
            if self.Q_chan_indep:
                x = torch.einsum('bnrd,nrh->bndh', x_trans.transpose(-1, -2), self.R_fix)
            else:
                x = torch.einsum('bnrd,rh->bndh', x_trans.transpose(-1, -2), self.R_fix) + self.delta2
                # x = x_trans + self.delta2
                # added on 25/1/30
                # x = F.gelu(x)
        else:
            if self.Q_chan_indep:
                x = torch.einsum('bnrd,nrh->bndh', x_trans.transpose(-1, -2), self.R_param)
            else:
                x = torch.einsum('bnrd,rh->bndh', x_trans.transpose(-1, -2), self.R_param)
                # x = x_trans + self.delta2
                # added on 25/1/30
                # x = F.gelu(x)

        # [B, N, tau, D]
        x = x.transpose(-1, -2)
        return x

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape

        # revin norm
        x = self.revin_layer(x, mode='norm')
        x_ori = x.transpose(-1, -2)       # [B,N,T]

        if self.Q_chan_indep:
            z_k = torch.einsum('bnt,ntr->bnr', x_ori, self.Q_in) + self.delta1 # [B,N,r]
        else:
            z_k = torch.einsum('bnt,tr->bnr', x_ori, self.Q_in) + self.delta1    # [B,N,r]
        x_ori = z_k.transpose(-1, -2)  # B r N
        
        # [B, T, N]
        # embedding x: [B, N, r, D]
        x = self.tokenEmb(x_ori, self.embeddings)

        x = self.Fre_Trans(x)

        # linear
        # [B, N, tau*D] --> [B, N, dim] --> [B, N, tau] --> [B, tau, N]
        out = self.fc(x.flatten(-2)).transpose(-1, -2)

        # dropout
        out = self.dropout(out)

        # revin denorm
        out = self.revin_layer(out, mode='denorm')

        return out
