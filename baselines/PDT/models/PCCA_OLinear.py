# PCCA_OLinear.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
from layers.Transformer_EncDec import Encoder_ori, LinearEncoder
from layers.CAR import CAR_Module_Temporal, GFM_Module_Temporal, PLA_DiagGate_Temporal

class Model(nn.Module):
    """
    OLinear-compatible model using Predictive CCA (fixed) transforms.
    - Expects q_mat (input side) computed by PredictiveCCA_gen.py
    - Expects q_out_mat (output side), often identity of shape (H,H) or (N,H,H)
    Shapes & einsum follow OLinear so training scripts remain the same.
    """
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.hidden_size = self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.Q_chan_indep = configs.Q_chan_indep
        self.k_dim = configs.k_dim

        self.enable_chan_align = configs.enable_chan_align
        self.chan_align_type = configs.chan_align_type
        assert self.chan_align_type in ["layernorm", "linear"], "chan_align_type must be 'layernorm' or 'linear'"

        # === load PCCA q_mat (input) ===
        q_mat_path = configs.Q_MAT_file if self.Q_chan_indep else configs.q_mat_file
        if not os.path.isfile(q_mat_path):
            q_mat_path = os.path.join(configs.root_path, q_mat_path)
        assert os.path.isfile(q_mat_path), f"Q mat not found: {q_mat_path}"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Q_mat = torch.from_numpy(np.load(q_mat_path)).to(torch.float32).to(device)

        # shape checks follow OLinear conventions
        if self.Q_chan_indep:
            assert self.Q_mat.ndim == 3 and self.Q_mat.shape[0] == self.enc_in \
                   and self.Q_mat.shape[1] == self.seq_len and self.Q_mat.shape[2] == self.k_dim
        else:
            assert self.Q_mat.ndim == 2 and self.Q_mat.shape[0] == self.seq_len \
                   and self.Q_mat.shape[1] == self.k_dim

        # === load output-side transform (often identity) ===
        q_out_path = configs.Q_OUT_MAT_file if self.Q_chan_indep else configs.q_out_mat_file
        if not os.path.isfile(q_out_path):
            q_out_path = os.path.join(configs.root_path, q_out_path)
        assert os.path.isfile(q_out_path), f"Q out mat not found: {q_out_path}"
        self.Q_out_mat = torch.from_numpy(np.load(q_out_path)).to(torch.float32).to(device)

        if self.Q_chan_indep:
            assert self.Q_out_mat.ndim == 3 and self.Q_out_mat.shape[0] == self.enc_in \
                   and self.Q_out_mat.shape[1] == self.pred_len and self.Q_out_mat.shape[2] == self.k_dim
        else:
            assert self.Q_out_mat.ndim == 2 and self.Q_out_mat.shape[0] == self.pred_len \
                   and self.Q_out_mat.shape[1] == self.k_dim

        # === token emb / backbone stay the same as OLinear ===
        self.patch_len = configs.temp_patch_len
        self.stride = configs.temp_stride
        self.embed_size = configs.embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.gate_module = configs.gate_module

        # self.fc = nn.Sequential(
        #     nn.Linear(self.pred_len * self.embed_size, self.d_ff),
        #     nn.GELU(),
        #     nn.Linear(self.d_ff, self.pred_len)
        # )
        self.fuse = nn.Linear(self.embed_size, 1, bias=True)

        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

        # 实例化针对时间特征的CAR模块
        if configs.gate_module == "CAR":
            self.car_temporal_module = CAR_Module_Temporal(num_temporal_features=self.k_dim, reduction_ratio=4)
        elif configs.gate_module == "GFM":
            self.gfm_module = GFM_Module_Temporal(num_temporal_features=self.k_dim)
        elif configs.gate_module == "PLA":
            self.pla_module = PLA_DiagGate_Temporal(
                k_dim=self.k_dim,
                film_hidden=getattr(configs, "pla_film_hidden", None),
                eps=getattr(configs, "pla_eps", 0.2),
                dropout=getattr(configs, "pla_dropout", 0.0),
            )

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
            nn.Linear(self.k_dim * self.embed_size, self.d_model),
            self.encoder,
            nn.Linear(self.d_model, self.k_dim * self.embed_size)
        )


        # learnable deltas (kept for drop-in compatibility; set requires_grad=True)
        # self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.k_dim))
        # self.delta2 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.pred_len))
        self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, self.k_dim))      # [1,N,k]，加在 Q_in 后
        # self.delta2 = nn.Parameter(torch.zeros(1, self.enc_in, self.pred_len))   # [1,N,τ]，加在 Q_out 后
        self.delta_k = nn.Parameter(torch.zeros(1, self.enc_in, self.k_dim))


    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        x = x.transpose(-1, -2)  # [B,N,T]
        x = x.unsqueeze(-1)      # [B,N,T,1]
        return x * embeddings    # [B,N,T,D]

    def Fre_Trans(self, x):
        # x: [B,N,T,D] -> transpose time and embed for time-transform
        B, N, T, D = x.shape
        # assert T == self.seq_len
        x = x.transpose(-1, -2)  # [B,N,D,T]

        x_trans = x

        # input-side transform by PCCA Q (fixed)
        # if self.Q_chan_indep:
        #     x_trans = torch.einsum('bndt,ntk->bndk', x, self.Q_mat) + self.delta1
        # else:
        #     x_trans = torch.einsum('bndt,tk->bndk', x, self.Q_mat) + self.delta1
        # assert x_trans.shape[-1] == self.k_dim

        # backbone mapping seq_len*D -> pred_len*D (unchanged)
        # x_trans: [B,N,D,k]  —— 投影后
        # 应用CAR模块，对变换后的T维时间特征进行重校准
        # 输入 x_trans 形状: [B, N, D, T], 输出形状: [B, N, D, T]
        if self.gate_module == "CAR":
            x_trans = self.car_temporal_module(x_trans)
        elif self.gate_module == "GFM":
            x_trans = self.gfm_module(x_trans)
        elif self.gate_module == "PLA":
            x_trans = self.pla_module(x_trans)
        else:
            x_trans = x_trans

        
        x_trans = self.ortho_trans(x_trans.flatten(-2)).reshape(B, N, D, self.k_dim)

        x_trans = x_trans + self.delta_k.unsqueeze(2)

        # output-side transform (often identity)
        if self.Q_chan_indep:
            x = torch.einsum('nhk,bndk->bndh', self.Q_out_mat, x_trans)
        else:
            x = torch.einsum('hk,bndk->bndh', self.Q_out_mat, x_trans)

        x = x.transpose(-1, -2)  # [B,N,tau,D]
        return x

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [B, T, N]
        x = self.revin_layer(x, mode='norm')
        # new
        x_nt = x.transpose(-1, -2)
        if self.Q_chan_indep:
            z_k = torch.einsum('bnt,ntk->bnk', x_nt, self.Q_mat) + self.delta1 # [B,N,k]
        else:
            z_k = torch.einsum('bnt,tk->bnk', x_nt, self.Q_mat) + self.delta1    # [B,N,k]
        x = z_k.transpose(-1, -2)  # B K N

        x_emb = self.tokenEmb(x, self.embeddings)          # [B,N,T,D]
        x_feat = self.Fre_Trans(x_emb)                     # [B,N,tau,D]
        # out = self.fc(x_feat.flatten(-2)).transpose(-1, -2)  # [B,tau,N]
        out = self.fuse(x_feat).squeeze(-1)
        out = out.transpose(-1, -2) 
        out = self.dropout(out)
        out = self.revin_layer(out, mode='denorm')
        return out
