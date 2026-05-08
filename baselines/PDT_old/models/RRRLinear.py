# models/RRRLinear.py
import os
import numpy as np
import torch
import torch.nn as nn

def _load_npy(path, root_path=None, device='cpu'):
    p = path if os.path.isfile(path) else os.path.join(root_path or '', path)
    assert os.path.isfile(p), f'File not found: {path}'
    arr = np.load(p)
    return torch.from_numpy(arr).to(torch.float32).to(device)

class Model(nn.Module):
    """
    RRR 纯线性零训练模型：
      y_hat = (x - mu_X) @ B + mu_Y
    其中 B = Qin @ R。mu_X/mu_Y 从 train_loader 统计（与离线 build_xy 的列均值对齐）。

    输入:  x: [B, L, N]
    输出:  y: [B, H, N]
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len  = configs.seq_len     # L
        self.pred_len = configs.pred_len    # H
        self.enc_in   = configs.enc_in      # N

        self.Q_chan_indep = getattr(configs, 'Q_chan_indep', False)
        device = torch.device("cuda:0" if torch.cuda.is_available() and configs.use_gpu else "cpu")

        # ---- 载入 Qin, R 并构造 B ----
        if self.Q_chan_indep:
            Q_path = getattr(configs, 'Q_MAT_file', None); R_path = getattr(configs, 'R_MAT_file', None)
            assert Q_path and R_path, "逐通道需要提供 Q_MAT_file, R_MAT_file"
            Q = _load_npy(Q_path, root_path=configs.root_path, device=device)   # [N, L, r]
            R = _load_npy(R_path, root_path=configs.root_path, device=device)   # [N, r, H]
            assert Q.shape[0] == self.enc_in and Q.shape[1] == self.seq_len
            assert R.shape[0] == self.enc_in and R.shape[2] == self.pred_len
            B = torch.einsum('nlr,nrh->nlh', Q, R)                              # [N, L, H]
            self.register_buffer('B', B)
        else:
            q_path = getattr(configs, 'q_mat_file', None); r_path = getattr(configs, 'r_mat_file', None)
            assert q_path and r_path, "共享版需要提供 q_mat_file, r_mat_file"
            Q = _load_npy(q_path, root_path=configs.root_path, device=device)   # [L, r]
            R = _load_npy(r_path, root_path=configs.root_path, device=device)   # [r, H]
            assert Q.shape[0] == self.seq_len and R.shape[1] == self.pred_len
            B = (Q @ R)                                                         # [L, H]
            self.register_buffer('B', B)

        # ---- 训练集统计的列均值（lazy 准备）----
        self.register_buffer('muX', None)   # [L] or [N, L]
        self.register_buffer('muY', None)   # [H] or [N, H]
        self.register_buffer('mu_ch', None)   # [N]
        self.register_buffer('std_ch', None)  # [N]
        self.register_buffer('muX_col', None) # [L]
        self.register_buffer('muY_col', None) # [H]
        self._stats_ready = False

    @torch.no_grad()
    def prepare_all_stats(self, train_loader, device=None):
        dev = device or next(self.parameters()).device
        L,H,N = self.seq_len, self.pred_len, self.enc_in

        # 1) 通道全局均值/方差（train）
        sum1 = torch.zeros(N, device=dev)
        sum2 = torch.zeros(N, device=dev)
        count = 0
        for bx, _, _, _ in train_loader:
            x = bx.to(dev).float()  # [B,L,N]
            sum1 += x.sum(dim=(0,1))             # over B,L
            sum2 += (x*x).sum(dim=(0,1))
            count += x.shape[0]*x.shape[1]
        mu_ch  = sum1 / count                    # [N]
        var_ch = (sum2 / count) - mu_ch**2
        std_ch = var_ch.clamp_min(1e-8).sqrt()   # [N]
        self.mu_ch = mu_ch
        self.std_ch = std_ch

        # 2) 在 z-score 后统计窗口列均值（与离线一致）
        sumX = torch.zeros(L, device=dev)
        sumY = torch.zeros(H, device=dev)
        countX = 0
        countY = 0
        for bx, by, _, _ in train_loader:
            x = bx.to(dev).float()               # [B,L,N]
            y = by[:, -H:, :].to(dev).float()    # [B,H,N]
            x_std = (x - mu_ch.view(1,1,-1)) / std_ch.view(1,1,-1)
            y_std = (y - mu_ch.view(1,1,-1)) / std_ch.view(1,1,-1)

            # 按“列”（沿 batch 和 channel）平均（与离线 build_xy 的列均值等价）
            sumX += x_std.sum(dim=(0,2))         # [L]
            sumY += y_std.sum(dim=(0,2))         # [H]
            countX += x_std.shape[0]*x_std.shape[2]
            countY += y_std.shape[0]*y_std.shape[2]
        self.muX_col = sumX / max(1,countX)      # [L]
        self.muY_col = sumY / max(1,countY)      # [H]

    def forward(self, batch_x, batch_x_mark=None, dec_inp=None, batch_y_mark=None):
        """
        兼容现有训练框架：忽略 *_mark 与 dec_inp。
        """
        assert all(t is not None for t in [self.mu_ch, self.std_ch, self.muX_col, self.muY_col])
        x = batch_x.float()                             # [B, L, N]
        # z-score
        x_std = (x - self.mu_ch.view(1,1,-1)) / self.std_ch.view(1,1,-1)
        # 列中心化
        x_c   = x_std - self.muX_col.view(1,-1,1)
        # 线性
        y_std = torch.einsum('bln,lh->bnh', x_c, self.B) + self.muY_col.view(1,1,-1)
        # 反 z-score
        y = y_std * self.std_ch.view(1, -1, 1) + self.mu_ch.view(1, -1, 1)
        return y.transpose(1, 2).contiguous()  # [B, H, N]