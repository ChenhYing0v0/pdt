import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.RevIN import RevIN
from layers.Transformer_EncDec import Encoder_ori, LinearAttn  # NormLin 变量融合


# --------------------------
# gate（三选一）：在 Q_in 后，输入/输出: [B, N, k]
# --------------------------

class CAR_Gate_1D(nn.Module):
    def __init__(self, k_dim: int, reduction_ratio: int = 16):
        super().__init__()
        b = max(1, k_dim // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(k_dim, b, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(b, k_dim, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):           # x: [B,N,k]
        w = self.mlp(x.mean(dim=1)) # [B,k]
        return x * w.unsqueeze(1)   # [B,N,k]


class GFM_Gate_1D(nn.Module):
    def __init__(self, k_dim: int):
        super().__init__()
        self.gate = nn.Linear(k_dim, k_dim)
    def forward(self, x):           # x: [B,N,k]
        return x * torch.sigmoid(self.gate(x))


class PLA_DiagGate_1D(nn.Module):
    def __init__(self, k_dim: int, film_hidden: int = None, eps: float = 0.2, dropout: float = 0.0):
        super().__init__()
        self.k = k_dim
        self.eps = float(eps)
        h = film_hidden if film_hidden is not None else max(8, k_dim // 4)
        self.mlp = nn.Sequential(
            nn.Linear(2 * k_dim, h), nn.GELU(),
            nn.Linear(h, k_dim)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):               # x: [B,N,k]
        mean_k = x.mean(dim=1)          # [B,k]
        std_k  = x.std (dim=1, unbiased=False)
        stats  = torch.cat([mean_k, std_k], dim=-1)      # [B,2k]
        dgamma = torch.tanh(self.mlp(stats)) * self.eps  # [B,k]
        gamma  = 1.0 + dgamma
        return self.drop(x * gamma.unsqueeze(1))         # [B,N,k]


# --------------------------
# PLinear
# --------------------------

class Model(nn.Module):
    """
    PLinear: RevIN → Q_in → gate → embedding(Linear k→d) → temp_mlp(d→d) → variate_attn(NormLin)
             → time_proj(d→τ) → Q_out → head(τ→τ) → denorm
    - Q_in/Q_out 形状与 einsum 路由对齐 PCCA_OLinear 的实现 
    - variate_attn 采用 OLinear 的 LinearEncoder（NormLin） 
    """
    def __init__(self, configs):
        super().__init__()
        # 尺寸与配置
        self.pred_len = configs.pred_len      # τ
        self.enc_in   = configs.enc_in        # N
        self.seq_len  = configs.seq_len       # T
        self.k_dim    = configs.k_dim         # k
        self.embed_sz = configs.d_model    # d
        self.d_model  = configs.d_model
        assert self.embed_sz == self.d_model, "建议 embed_size == d_model，便于直接接入 NormLin"

        self.dropout   = configs.dropout
        self.d_ff      = configs.d_ff
        self.e_layers  = configs.e_layers
        self.activation = configs.activation
        self.Q_chan_indep = configs.Q_chan_indep
        self.gate_module = configs.gate_module  # {"CAR","GFM","PLA"}

        # RevIN
        self.revin_layer = RevIN(self.enc_in, affine=True)

        # ===== 加载 Q_in / Q_out =====
        q_mat_path = configs.Q_MAT_file if self.Q_chan_indep else configs.q_mat_file
        if not os.path.isfile(q_mat_path):
            q_mat_path = os.path.join(configs.root_path, q_mat_path)
        assert os.path.isfile(q_mat_path), f"Q_in not found: {q_mat_path}"

        q_out_path = configs.Q_OUT_MAT_file if self.Q_chan_indep else configs.q_out_mat_file
        if not os.path.isfile(q_out_path):
            q_out_path = os.path.join(configs.root_path, q_out_path)
        assert os.path.isfile(q_out_path), f"Q_out not found: {q_out_path}"

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Q_mat     = torch.from_numpy(np.load(q_mat_path)).to(torch.float32).to(device)
        self.Q_out_mat = torch.from_numpy(np.load(q_out_path)).to(torch.float32).to(device)

        if self.Q_chan_indep:
            # Q_in: [N,T,k]；Q_out: [N,τ,τ]
            assert self.Q_mat.ndim == 3 and self.Q_mat.shape[:2] == (self.enc_in, self.seq_len) and self.Q_mat.shape[2] == self.k_dim
            assert self.Q_out_mat.ndim == 3 and self.Q_out_mat.shape[:2] == (self.enc_in, self.pred_len) and self.Q_out_mat.shape[2] == self.pred_len
        else:
            # Q_in: [T,k]；Q_out: [τ,τ]
            assert self.Q_mat.ndim == 2 and self.Q_mat.shape == (self.seq_len, self.k_dim)
            assert self.Q_out_mat.ndim == 2 and self.Q_out_mat.shape == (self.pred_len, self.pred_len)

        # ===== gate（[B,N,k]）=====
        if self.gate_module == "CAR":
            self.gate = CAR_Gate_1D(self.k_dim, reduction_ratio=getattr(configs, "car_reduction", 16))   # 参考 CAR 思路  :contentReference[oaicite:6]{index=6}
        elif self.gate_module == "GFM":
            self.gate = GFM_Gate_1D(self.k_dim)                                                          # 参考 GFM 思路  :contentReference[oaicite:7]{index=7}
        elif self.gate_module == "PLA":
            self.gate = PLA_DiagGate_1D(self.k_dim,
                                        film_hidden=getattr(configs, "pla_film_hidden", None),
                                        eps=getattr(configs, "pla_eps", 0.2),
                                        dropout=getattr(configs, "pla_dropout", 0.0))
        else:
            self.gate = nn.Identity()

        # ===== embedding：Linear(k→d) =====
        self.embed = nn.Linear(self.k_dim, self.embed_sz)

        # ===== temp_mlp：仅在 d 内做两层（不造 τ）=====
        act = nn.ReLU() if self.activation.lower() == "relu" else nn.GELU()
        self.temp_mlp = nn.Sequential(
            nn.Linear(self.embed_sz, self.d_ff),
            act,
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.embed_sz),
            nn.Dropout(self.dropout)
        )

        # ===== variate_attn(NormLin)：跨变量 N（对 [B,N,d]）=====
        self.var_encoder = Encoder_ori(
            [
                LinearAttn(
                    d_model=self.d_model, d_ff=self.d_ff, CovMat=None,
                    dropout=self.dropout, activation=self.activation, token_num=self.enc_in,
                ) for _ in range(self.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
            one_output=True,
            CKA_flag=getattr(configs, "CKA_flag", False)
        )

        # ===== time_proj：d → τ（产生时间轴）=====
        self.time_proj = nn.Linear(self.d_model, self.pred_len)

        # ===== head：沿 τ 的轻量 refinement（τ→τ），再转为 [B,τ,N] =====
        # self.head = nn.Sequential(
        #     nn.Linear(self.pred_len, self.d_ff),  # 1x1 over τ（可换成 Identity）
        #     nn.GELU(),
        #     nn.Linear(self.d_ff, self.pred_len)
        # )
        self.head = nn.Sequential(
            nn.Linear(self.pred_len, self.pred_len),
        )
        self.dropout_layer = nn.Dropout(self.dropout)
        self.norm1 = nn.LayerNorm(self.d_model)

        # 与 PCCA_OLinear 一致的可学偏移项（如不需要可保持全 0）
        self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, self.k_dim))    # 作用在 Q_in 后的 [B,N,k]
        self.delta2 = nn.Parameter(torch.zeros(1, self.enc_in, self.pred_len)) # 作用在 Q_out 后的 [B,N,τ]

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):  # x: [B, T, N]
        B, T, N = x.shape
        assert T == self.seq_len and N == self.enc_in

        # 1) RevIN
        x = self.revin_layer(x, mode='norm')          # [B,T,N]

        # 2) Q_in（T→k）
        x = x.transpose(-1, -2)                       # [B,N,T]
        if self.Q_chan_indep:
            z_k = torch.einsum('bnt,ntk->bnk', x, self.Q_mat)   # [B,N,k]
        else:
            z_k = torch.einsum('bnt,tk->bnk', x, self.Q_mat)    # [B,N,k]
        z_k = z_k + self.delta1                                   # [B,N,k]

        # 3) gate
        z_k = self.gate(z_k)                                      # [B,N,k]

        # 4) embedding（k→d）
        z_d = self.embed(z_k)                                     # [B,N,d]

        # 5) temp_mlp（d→d）
        z_dy = self.temp_mlp(z_d)                                  # [B,N,d]

        ## 改
        # z_dy = self.norm1(z_dy+z_d)


        # 6) variate_attn（NormLin，跨 N）
        z_dy = self.var_encoder(z_dy)                               # [B,N,d]

        # 7) time_proj（d→τ）：产生时间轴
        y_tau = self.time_proj(z_dy)                               # [B,N,τ]

        # 8) Q_out（沿 τ 轴）
        if self.Q_chan_indep:
            y_tau = torch.einsum('bnt,ntv->bnv', y_tau, self.Q_out_mat)  # [B,N,τ]
        else:
            y_tau = torch.einsum('bnt,tv->bnv', y_tau, self.Q_out_mat)   # [B,N,τ]
        y_tau = y_tau + self.delta2                                     # [B,N,τ]

        # 9) head（τ→τ），并转回 [B,τ,N]
        y_tau = self.head(y_tau)                                        # [B,N,τ]
        out = y_tau.transpose(-1, -2)                                   # [B,τ,N]
        out = self.dropout_layer(out)

        # 10) denorm
        out = self.revin_layer(out, mode='denorm')                      # [B,τ,N]
        return out
