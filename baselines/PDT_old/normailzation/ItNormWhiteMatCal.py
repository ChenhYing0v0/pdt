# ItNormWhiteMatCal.py
# 仅计算白化矩阵的版本，不进行输入变换
# 基于 IterNormTempAuto 的核心逻辑，在时间维度 L 上计算白化矩阵

import torch
import torch.nn as nn

class ItNormWhiteMatCal(nn.Module):
    """
    仅计算白化矩阵的迭代归一化模块 (Temporal Data)。

    Args:
        seq_len (int): 序列长度 (L 维度)，将在此维度上进行解相关。
        T (int, optional): Newton-Schulz 迭代的次数。默认: 5。
        eps (float, optional): 添加到协方差矩阵对角线的小值，用于稳定性。默认: 1e-5。
    """
    def __init__(self, seq_len, T=5, eps=1e-5, *args, **kwargs):
        super(ItNormWhiteMatCal, self).__init__()
        self.seq_len = seq_len
        self.T = T
        self.eps = eps

    def forward(self, X: torch.Tensor):
        """
        计算白化矩阵 wm。
        
        Args:
            X (torch.Tensor): 输入张量，形状为 (B, C, L)
            
        Returns:
            torch.Tensor: 白化矩阵，形状为 (1, L, L)
        """
        assert X.dim() == 3, f"输入必须是3D张量 (B, C, L)，但得到的是 {X.dim()}D"
        assert X.size(2) == self.seq_len, f"输入序列长度 {X.size(2)} 与模块的 seq_len {self.seq_len} 不匹配"
        
        # 记录原始维度
        B, C, L = X.shape
        
        # 将 (B, C, L) 重塑为 (g, d, m) 其中 g=1, d=L, m=B*C
        # 这是为了在时间维度 L 上进行解相关
        x_reshaped = X.permute(2, 0, 1).contiguous().view(1, L, B * C)
        g, d, m = x_reshaped.shape
        
        # 1. 计算批次统计量
        # 计算中心化激活
        mean = x_reshaped.mean(-1, keepdim=True)  # Shape: (1, L, 1)
        xc = x_reshaped - mean
        
        # 计算协方差矩阵 Sigma，形状为 (1, L, L)
        I = torch.eye(d, device=X.device, dtype=X.dtype).unsqueeze(0) # Shape: (1, d, d)
        
        # Sigma = (1/m) * xc @ xc^T
        Sigma = torch.baddbmm(I.mul(self.eps), xc, xc.transpose(1, 2), beta=1., alpha=1./m)
        
        # 计算 Sigma 迹的倒数
        rTr = (Sigma * I).sum((1, 2), keepdim=True).reciprocal_()
        
        # 归一化协方差矩阵
        Sigma_N = Sigma * rTr

        # 2. 通过 Newton-Schulz 迭代计算 Sigma^{-1/2}
        P = I.clone()
        for _ in range(self.T):
            P_cubed = torch.matrix_power(P, 3)
            P = torch.baddbmm(P.mul(1.5), P_cubed, Sigma_N, beta=1., alpha=-0.5)
        
        # 得到最终的白化矩阵
        wm = P.mul(rTr.sqrt()) # Shape: (1, L, L)
        
        return wm

    def extra_repr(self):
        return 'seq_len={seq_len}, T={T}, eps={eps}'.format(**self.__dict__) 