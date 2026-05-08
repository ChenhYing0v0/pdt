import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.RevIN import RevIN
from layers.Transformer_EncDec import Encoder_ori, LinearAttn   # NormLin 变量融合（跨N）



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


class Model(nn.Module):
    """
    RevIN → Q_in → tokenEmb → gate → temp_mlp(on D) → variate_attn(NormLin, per-k)
          → time_proj(k→τ) → Q_out(τ→τ) → head → denorm
    让 [B,N,k,D] 贯穿到 time_proj 之前，符合 OLinear 的“token × embed”布局与调用规范。  :contentReference[oaicite:2]{index=2}
    """
    def __init__(self, configs):
        super().__init__()
        # 基本尺寸
        self.pred_len = configs.pred_len         # τ
        self.enc_in   = configs.enc_in           # N
        self.seq_len  = configs.seq_len          # T
        self.k_dim    = configs.k_dim            # k
        self.embed_sz = configs.embed_size       # D
        self.d_model  = configs.d_model          # 与 D 保持一致，方便直接喂 NormLin
        # assert self.embed_sz == self.d_model, "建议 embed_size == d_model，便于直接接入 NormLin"

        self.dropout  = configs.dropout
        self.d_ff     = configs.d_ff
        self.e_layers = configs.e_layers
        self.activation = configs.activation
        self.Q_chan_indep = configs.Q_chan_indep
        self.gate_module = configs.gate_module   # {"CAR","GFM","PLA"}

        # RevIN
        self.revin_layer = RevIN(self.enc_in, affine=True)

        # ===== 加载 Q_in / Q_out（与 PCCA_OLinear.py 的断言/形状一致）=====
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

        # ===== tokenEmb（OLinear 风格）：对 k-token 做“幅度扩展”，得到 [B,N,k,D] =====
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_sz))

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

        self.proj_1 = nn.Linear(self.k_dim * self.embed_sz, self.d_model)
        self.proj_2 = nn.Linear(self.d_model, self.pred_len * self.embed_sz)
        # ===== temp_mlp：仅在 D 上（逐 (B,N,k) 的 1×1 MLP）=====
        act = nn.ReLU() if self.activation.lower() == "relu" else nn.GELU()
        self.temp_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff), act, nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_model), nn.Dropout(self.dropout)
        )
        self.norm1 = nn.LayerNorm(self.d_model)

        # ===== variate_attn（NormLin）：对每个 k 切片做跨变量融合 =====
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

        # ===== time_proj：k → τ（保持 D 不变，便于随后走 Q_out）=====
    

        # ===== head：与现工程一致（flatten τ×D → τ 并轻量 refinement）=====
        self.head = nn.Sequential(
            nn.Linear(self.pred_len * self.embed_sz, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )
        self.dropout_layer = nn.Dropout(self.dropout)

        # 轻量偏移（与 PCCA_OLinear 一致），如不需要可保持为 0
        self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, self.k_dim))      # [1,N,k]，加在 Q_in 后
        self.delta2 = nn.Parameter(torch.zeros(1, self.enc_in, self.pred_len))   # [1,N,τ]，加在 Q_out 后

    # ------- OLinear 同款 tokenEmb：把 [B,N,k] 扩成 [B,N,k,D] -------
    def tokenEmb(self, z_bnk):  # z_bnk: [B,N,k]
        # 按 OLinear：在“时间轴上”加一个长度为 D 的可学习幅度向量（广播乘）
        z = z_bnk.unsqueeze(-1)                # [B,N,k,1]
        return z * self.embeddings             # [B,N,k,D]

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):  # x: [B,T,N]
        B, T, N = x.shape
        assert T == self.seq_len and N == self.enc_in

        # 1) RevIN
        x = self.revin_layer(x, mode='norm')                  # [B,T,N]

        # 2) Q_in：T→k（按 PCCA_OLinear 的 einsum 路由）
        x_nt = x.transpose(-1, -2)                            # [B,N,T]
        if self.Q_chan_indep:
            z_k = torch.einsum('bnt,ntk->bnk', x_nt, self.Q_mat)   # [B,N,k]
        else:
            z_k = torch.einsum('bnt,tk->bnk', x_nt, self.Q_mat)    # [B,N,k]
        z_k = z_k + self.delta1                                     # [B,N,k]
        z_k = self.gate(z_k)                                      # [B,N,k]                         

         # 3) tokenEmb：得到 [B,N,k,D]
        z_kd = self.tokenEmb(z_k)  

        # ---- tempMLP ---- （）
        # 5) temp_mlp（仅在 D 上）：逐 (B,N,k) 做 1×1 MLP
        B_, N_, K_, D_ = z_kd.shape

        z_kd = z_kd.permute(0, 1, 3, 2).flatten(-2) # [B,N,k,D] -> [B,N,k*D]
        z_kd = self.proj_1(z_kd) # [B,N,k*D] -> [B,N,d_model]

        y_kd = self.temp_mlp(z_kd)   # [B,N,d_model]
        y_kd = self.norm1(y_kd+z_kd)    # [B,N,d_model]

        y_kd = self.var_encoder(y_kd)    # [B,N,d_model]

        y_dp = self.proj_2(y_kd).reshape(B_, N_, D_, self.pred_len) # [B,N,D,τ]

        # 8) Q_out：沿 τ 轴（与 PCCA_OLinear 相同的路由）
        if self.Q_chan_indep:
            y_dt = torch.einsum('bndt,ntv->bndv', y_dp, self.Q_out_mat)  # [B,N,D,τ]
        else:
            y_dt = torch.einsum('bndt,tv->bndv', y_dp, self.Q_out_mat)   # [B,N,D,τ]
        y_dt = y_dt + self.delta2.unsqueeze(2)                            # [B,N,D,τ] + [1,N,1,τ]

        # 9) head：flatten τ×D → τ，输出 [B,τ,N]
        y_td = y_dt.transpose(-1, -2)                                     # [B,N,τ,D]
        out  = self.head(y_td.flatten(-2)).transpose(-1, -2)              # [B,τ,N]
        out  = self.dropout_layer(out)

        # 10) denorm
        out = self.revin_layer(out, mode='denorm')                        # [B,τ,N]
        return out
