import os
import math
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
ROLinear (RRR + OLinear) with Top-k Linear Head:
      x --RevIN--> Q_in投影到 r 维 --tokenEmb--> [B,N,r,D]
        ├─ Key 分支：取前 k 个方向 -> Linear(k*D -> r)  (简单线性预测)
        └─ Ortho 分支：orthotrans 处理复杂特征 -> D-池化到 r
      α 融合: z = σ(α)*z_key + (1-σ(α))*z_ortho  -> (可选 LayerNorm(r))
      经 R (r->H) 风格 fc(H*D->H) -> RevIN 反变换
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

        # ------- Key 分支：对每个 D slice 共享的 k→r 线性映射 -------
        # self.k_top = int(getattr(configs, 'k_top', min(16, self.r)))
        # assert 1 <= self.k_top <= self.r
        # 映射矩阵形状 [r, k]（注意 F.linear 的 weight 语义）
        # self.key_map = nn.Linear(self.k_top, self.r)
        # nn.init.xavier_uniform_(self.key_map.weight, gain=0.5)

        # 融合门 α（标量）；如需“每个 r 一门”，可改成 nn.Parameter(torch.zeros(r)) 后用 sigmoid
        # alpha_init = float(getattr(configs, 'alpha_init', 0.5))
        # logit = math.log(alpha_init/(1-alpha_init))
        # self.alpha_logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

        # 融合后的稳态：先对 D 维做 LN（R 不感知 D，归一 D 比较自然）
        # self.use_norm_D = bool(getattr(configs, 'use_norm_D', 1))
        # if self.use_norm_D:
        #     self.norm_D = nn.LayerNorm(self.embed_size)
        

        self.fc = nn.Sequential(
            nn.Linear(self.pred_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )
        # if getattr(configs, 'init_fc_avg', 1):
        #     with torch.no_grad():
        #         w1, b1 = self.fc[0].weight, self.fc[0].bias
        #         w2, b2 = self.fc[2].weight, self.fc[2].bias
        #         H, D = self.pred_len, self.embed_size
        #         w1.zero_(); b1.zero_()
        #         for h in range(H):
        #             for d in range(D): w1[h, h*D + d] = 1.0 / D
        #         if self.d_ff == H:
        #             w2.zero_(); b2.zero_()
        #             for h in range(H): w2[h, h] = 1.0

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
        x_tok = self.tokenEmb(x_ori, self.embeddings)

        # x = self.Fre_Trans(x)
        # ------ Ortho 分支：保持到 [B,N,r,D] ------
        B, N, r, D = x_tok.shape
        assert r == self.r
        x_trans = x_tok.transpose(-1, -2)                        # [B,N,D,r]
        x_ortho = self.ortho_trans(x_trans.flatten(-2)).reshape(B, N, D, self.r)          # [B,N,d_model] -> [B,N,r*D]      # [B,N,D,r]

        # ------ Key 分支：Top-k -> [B,N,r,D] （对每个 D slice 共享 k→r 线性）------
        # x_k = x_tok[:, :, :self.k_top, :]                           # [B,N,k,D]
        # # 让 k 作为最后一维应用 Linear(k->r)：把 D 提到倒数第二维
        # x_k_dk = x_k.permute(0,1,3,2)                               # [B,N,D,k]
        # z_key_rd = self.key_map(x_k_dk)           # [B,N,D,r]


        # ------ α 融合（保留 D）------
        # alpha = torch.sigmoid(self.alpha_logit)                     # 标量
        # z_fused = alpha * z_key_rd + (1.0 - alpha) * x_ortho        # [B,N,D,r]
        # if self.use_norm_D:
        #     z_fused = self.norm_D(z_fused)                          # 对最后一维 D 做 LN
        
        z_fused = x_ortho      # 只用单router分支
        
        if self.freeze_R:
            R = self.R_fix
        else:
            R = self.R_param
        if self.Q_chan_indep:
            yD = torch.einsum('bnrd,nrh->bndh', z_fused.transpose(-1, -2), R)         # [B,N,D,H]
        else:
            yD = torch.einsum('bnrd,rh->bndh', z_fused.transpose(-1, -2), R)          # [B,N,D,H]
        yD = yD.transpose(-1, -2)                                   # [B,N,H,D]


        # linear
        # [B, N, tau*D] --> [B, N, dim] --> [B, N, tau] --> [B, tau, N]
        out = self.fc(yD.flatten(-2)).transpose(-1, -2)

        # dropout
        out = self.dropout(out)

        # revin denorm
        out = self.revin_layer(out, mode='denorm')

        return out
