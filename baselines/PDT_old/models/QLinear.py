import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN  # 与 OLinear 保持一致
import math

def _load_npy(path, root_path=None, device='cpu'):
    p = path if os.path.isfile(path) else os.path.join(root_path or '', path)
    assert os.path.isfile(p), f'File not found: {path}'
    arr = np.load(p)
    ten = torch.from_numpy(arr).to(torch.float32).to(device)
    return ten

# ---------- 更通用的样本上下文提取 ----------
class StatContext(nn.Module):
    """
    从 x_n: [B, L, N] 提取每通道的上下文向量 c: [B, N, C]
    mode='stats2' -> [mean, std]
    mode='stats8' -> [mean, std, skew, kurt, q25, q50, q75, max_abs]
    mode='conv'   -> depthwise conv + GAP -> C 维
    """
    def __init__(self, L, mode='stats2', C=8):
        super().__init__()
        self.mode = mode
        self.C = C
        if mode == 'conv':
            self.conv = nn.Conv1d(in_channels=1, out_channels=C, kernel_size=5, padding=2, groups=1, bias=True)
        assert mode in ['stats2','stats8','conv']

    def forward(self, x):  # x: [B,L,N]
        B,L,N = x.shape
        if self.mode == 'stats2':
            mu  = x.mean(dim=1)                       # [B,N]
            std = x.std(dim=1).clamp_min(1e-6)        # [B,N]
            return torch.stack([mu, std], dim=-1)     # [B,N,2]

        if self.mode == 'stats8':
            mu  = x.mean(dim=1)
            std = x.std(dim=1).clamp_min(1e-6)
            xc  = (x - mu.unsqueeze(1)) / std.unsqueeze(1)
            skew = (xc**3).mean(dim=1)
            kurt = (xc**4).mean(dim=1) - 3.0
            def _q(x, q):
                # x: [B,L,N] over dim=1
                if hasattr(torch, 'quantile'):
                    return torch.quantile(x, q, dim=1)  # [B,N]
                L = x.size(1)
                k = int(round(q*(L+1)))
                k = max(1, min(L, k))
                return x.kthvalue(k, dim=1).values
            q25 = _q(x, 0.25); q50 = _q(x, 0.50); q75 = _q(x, 0.75)
            mabs = x.abs().amax(dim=1)
            return torch.stack([mu,std,skew,kurt,q25,q50,q75,mabs], dim=-1)  # [B,N,8]

        # conv
        x1 = x.transpose(1,2).unsqueeze(2)   # [B,N,1,L]
        x1 = x1.reshape(B*N,1,L)             # depthwise per-channel
        h  = self.conv(x1)                   # [B*N,C,L]
        h  = h.mean(dim=-1).reshape(B,N,-1)  # [B,N,C]
        return h

class QinAdapter(nn.Module):
    """FiLM： context(B,N,C) → (gamma, beta) ∈ R^{B×N×r}"""
    def __init__(self, r, C, hidden=64, zero_init=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, hidden), nn.GELU(),
            nn.Linear(hidden, 2*r)
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, ctx):  # [B,N,C]
        gb = self.net(ctx)           # [B,N,2r]
        gamma, beta = gb.chunk(2, dim=-1)  # [B,N,r]
        return gamma.tanh(), beta


# ---------- 轻量 Mixer（不丢 D 维），并在块尾加 Norm ----------
class RDMixer(nn.Module):
    """
    输入 z_tok: [B,N,r,D]
    先沿 D 混合(1x1)，再沿 r 混合(1x1)，皆为 PreNorm 残差；块尾再加 LayerNorm(D)。
    """
    def __init__(self, r, D, hid_d=0, hid_r=0, p_drop=0.0):
        super().__init__()
        self.p_drop = p_drop
        self.norm_d = nn.LayerNorm(D)
        self.norm_r = nn.LayerNorm(r)
        # D-mix
        self.use_d = hid_d and hid_d>0
        if self.use_d:
            self.d_mlp = nn.Sequential(nn.Linear(D, hid_d), nn.GELU(), nn.Linear(hid_d, D))
            nn.init.zeros_(self.d_mlp[-1].weight); nn.init.zeros_(self.d_mlp[-1].bias)
        # r-mix
        self.use_r = hid_r and hid_r>0
        if self.use_r:
            self.r_mlp = nn.Sequential(nn.Linear(r, hid_r), nn.GELU(), nn.Linear(hid_r, r))
            nn.init.zeros_(self.r_mlp[-1].weight); nn.init.zeros_(self.r_mlp[-1].bias)

        self.out_norm = nn.LayerNorm(D)

    def forward(self, z):  # [B,N,r,D]
        if self.use_d:
            z = z + F.dropout(self.d_mlp(self.norm_d(z)), p=self.p_drop, training=self.training)
        # 交换轴以在 r 上做 LayerNorm/MLP
        if self.use_r:
            z = z.transpose(-1,-2)  # [B,N,D,r]
            z = z + F.dropout(self.r_mlp(self.norm_r(z)), p=self.p_drop, training=self.training)
            z = z.transpose(-1,-2)  # [B,N,r,D]
        z = self.out_norm(z)
        return z

# ---------- 学习式 D-池化（而非平均） ----------
class DPool(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.pool = nn.Linear(D, 1, bias=False)
    def forward(self, z):   # [B,N,r,D]
        return self.pool(z).squeeze(-1)  # [B,N,r]

class LoRA(nn.Module):
    """低秩增量 AB，用于在 R 上做轻微修正"""
    def __init__(self, in_dim, out_dim, rank=0, scale=1.0):
        super().__init__()
        self.rank = rank
        self.scale = scale
        if rank and rank > 0:
            self.A = nn.Linear(in_dim, rank, bias=False)
            self.B = nn.Linear(rank, out_dim, bias=False)
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)

    def forward(self, z):
        if self.rank and self.rank > 0:
            return self.B(self.A(z)) * self.scale
        return 0.0

class Model(nn.Module):
    """
    RRR Baseline:
    ReVIN -> Qin -> QinAdapter -> tokenEmbed -> (mean over D + tiny MLP) -> Head(R) -> Denorm
    兼容 OLinear 的输入/输出约定与设备处理方式。
    """
    def __init__(self, configs):
        super().__init__()
        self.pred_len = H = configs.pred_len
        self.enc_in   = N = configs.enc_in
        self.seq_len  = L = configs.seq_len

        # self.embed_size = getattr(configs, 'embed_size', 1)  # 与 OLinear 一致的乘法式嵌入
        self.dropout = nn.Dropout(getattr(configs, 'dropout', 0.0))
        self.Q_chan_indep = getattr(configs, 'Q_chan_indep', False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ==== 载入 RRR 矩阵 ====
        # Qin: [L,r] 或 [N,L,r]
        q_path = configs.Q_MAT_file if self.Q_chan_indep else configs.q_mat_file
        Q_in = _load_npy(q_path, root_path=configs.root_path, device=device)
        assert Q_in.ndim in (2,3)
        if self.Q_chan_indep:
            assert Q_in.shape[0] == N and Q_in.shape[1] == L
            r = Q_in.shape[2]
        else:
            assert Q_in.shape[0] == L
            r = Q_in.shape[1]
        self.r = r
        self.register_buffer('Q_in', Q_in)  # 跟随 to(device)

        # R: [r,H] 或 [N,r,H]
        r_path = getattr(configs, 'R_MAT_file', None) if self.Q_chan_indep \
                 else getattr(configs, 'r_mat_file', None)
        assert r_path is not None, "请在 configs 中提供 r_mat_file / R_MAT_file（RRR 的 R 矩阵）"
        R = _load_npy(r_path, root_path=configs.root_path, device=device)
        if self.Q_chan_indep:
            assert R.shape == (N, r, H)
        else:
            assert R.shape == (r, H)
        # Head：以 R 为初始化（可冻结或细调）
        self.freeze_R = getattr(configs, 'freeze_R', False)
        if self.freeze_R:
            self.register_buffer('R_fix', R)
        else:
            self.R_param = nn.Parameter(R.clone())

        # 可选 LoRA 修正（默认不开）
        self.use_lora = getattr(configs, 'r_lora_rank', 0) > 0
        if self.use_lora:
            import math
            rank = int(getattr(configs, 'r_lora_rank', 0))
            scale = float(getattr(configs, 'r_lora_scale', 1.0))
            # 针对共享/通道独立分别定义
            if self.Q_chan_indep:
                self.lora = nn.ModuleList([LoRA(r, H, rank=rank, scale=scale) for _ in range(N)])
            else:
                self.lora = LoRA(r, H, rank=rank, scale=scale)
        else:
            self.lora = None

        # ==== RevIN ====
        self.revin = RevIN(N, affine=True)  # 与 OLinear 相同的用法与 I/O 形状 :contentReference[oaicite:3]{index=3}

        # 1) 上下文
        self.ctx = StatContext(L=self.seq_len,
                            mode=getattr(configs, 'adapter_context', 'stats2'),
                            C=getattr(configs, 'adapter_C', 8))
        C = (2 if self.ctx.mode=='stats2' else (8 if self.ctx.mode=='stats8' else getattr(configs,'adapter_C',8)))
        self.q_adapter = QinAdapter(r=self.r, C=C,
                                    hidden=getattr(configs,'q_adapter_hidden',64), zero_init=True)
        
        # 2) tokenEmb 之后不再 mean，而是 Mixer + 学习式 D-池化
        self.embed_size = getattr(configs, 'embed_size', 1)
        # self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size) / max(1, self.embed_size**0.5))
        self.mixer = RDMixer(r=self.r, D=self.embed_size,
                            hid_d=getattr(configs,'mix_hid_d',0),
                            hid_r=getattr(configs,'mix_hid_r',0),
                            p_drop=getattr(configs,'dropout',0.0))
        self.dpool = DPool(D=self.embed_size)

        # 3) 头部前的统一 FiLM（可选；与 Qin 同风格）
        self.use_head_film = getattr(configs, 'use_head_film', False)
        if self.use_head_film:
            self.head_adapter = QinAdapter(r=self.r, C=C,
                                        hidden=getattr(configs,'q_adapter_hidden',64), zero_init=True)

    # 乘法式 token 嵌入： [..., r] -> [..., r, D]
    def token_emb(self, z):
        if self.embed_size <= 1:
            return z.unsqueeze(-1)
        return z.unsqueeze(-1) * self.embeddings  # broadcast

    def _proj_with_Q(self, x):
        # x: [B,L,N]  →  z: [B,N,r]
        if self.Q_chan_indep:
            # einsum: (B,L,N) x (N,L,r) -> (B,N,r)
            z = torch.einsum('bln,nlr->bnr', x, self.Q_in)
        else:
            # einsum: (B,L,N) x (L,r) -> (B,N,r)
            z = torch.einsum('bln,lr->bnr', x, self.Q_in)
        return z

    def _head_apply(self, z):
        # z: [B,N,r] → y: [B,N,H]
        if self.freeze_R:
            if self.Q_chan_indep:
                y = torch.einsum('bnr,nrh->bnh', z, self.R_fix)
            else:
                y = torch.einsum('bnr,rh->bnh', z, self.R_fix)
        else:
            if self.Q_chan_indep:
                y = torch.einsum('bnr,nrh->bnh', z, self.R_param)
            else:
                y = torch.einsum('bnr,rh->bnh', z, self.R_param)

        # LoRA 增量（可选）
        if self.lora is not None:
            if self.Q_chan_indep:
                y_add = []
                for n in range(z.size(1)):
                    y_add.append(self.lora[n](z[:, n, :]))  # [B,H]
                y = y + torch.stack(y_add, dim=1)          # [B,N,H]
            else:
                y = y + self.lora(z)                        # [B,N,H]
        return y

    def forward(self, x, *args, **kwargs):
        # x: [B,L,N]
        B, L, N = x.shape
        assert L == self.seq_len and N == self.enc_in

        # 1) RevIN norm
        x_n = self.revin(x, mode='norm')  # [B,L,N]

        # 2) Qin 投影
        z = self._proj_with_Q(x_n)  # [B,N,r]

        # Qin FiLM（样本自适应）
        ctx = self.ctx(x_n)                          # [B,N,C]
        gamma, beta = self.q_adapter(ctx)            # [B,N,r]
        z = (1+gamma) * z + beta

        # 4) tokenEmbed（乘法式）
        z_tok = self.token_emb(z)                      # [B,N,r,D]
        

        # 5) 简单表达（均值 D + 轻 MLP）
        z_tok = self.mixer(z_tok)                    # [B,N,r,D]
        z_feat = self.dpool(z_tok)                   # [B,N,r]

        # 头部前再做一次统一 FiLM（可选）
        if self.use_head_film:
            g2, b2 = self.head_adapter(ctx)          # [B,N,r]
            z_feat = (1+g2) * z_feat + b2            # [B,N,r]

        # 6) 预测头（R 或 R+LoRA）
        y = self._head_apply(z_feat)                   # [B,N,H]
        y = y.transpose(1, 2)                          # [B,H,N]

        # 7) RevIN denorm
        y = self.revin(y, mode='denorm')
        return y
