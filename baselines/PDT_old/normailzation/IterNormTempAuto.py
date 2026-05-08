# IterNormTempAuto.py
# 一个完全依赖 PyTorch Autograd 的版本，用于处理时序数据
# 它保留了 IterNorm_temp 的核心逻辑：在时间维度 L 上进行白化，所有通道共享一个白化矩阵

import torch
import torch.nn as nn
from torch.nn import Parameter

class IterNormTempAuto(nn.Module):
    """
    Iterative Normalization for Temporal Data (Autograd Version).

    Args:
        seq_len (int): The length of the sequence (L dimension), which will be decorrelated.
        T (int, optional): Number of iterations for Newton-Schulz iteration. Default: 5.
        eps (float, optional): A small value added to the diagonal of covariance for stability. Default: 1e-5.
        momentum (float, optional): The momentum for updating running statistics. Default: 0.1.
        affine (bool, optional): If True, this module has learnable affine parameters. Default: True.
    """
    def __init__(self, seq_len, T=5, eps=1e-5, momentum=0.1, affine=True, *args, **kwargs):
        super(IterNormTempAuto, self).__init__()
        self.seq_len = seq_len
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        # 仿射变换参数 (weight 和 bias)
        # 形状为 (1, 1, L)，可以广播到 (B, C, L)
        if self.affine:
            self.weight = Parameter(torch.ones(1, 1, self.seq_len))
            self.bias = Parameter(torch.zeros(1, 1, self.seq_len))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # 运行中的统计量，用于推理
        # running_mean 的形状是 (1, L, 1)
        self.register_buffer('running_mean', torch.zeros(1, self.seq_len, 1))
        # running_wm 是白化矩阵，形状是 (1, L, L)
        self.register_buffer('running_wm', torch.eye(self.seq_len).unsqueeze(0))
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        assert X.dim() == 3, f"Input must be a 3D tensor (B, C, L), but got {X.dim()}D"
        assert X.size(2) == self.seq_len, f"Input sequence length {X.size(2)} does not match module's seq_len {self.seq_len}"
        
        # 记录原始维度
        B, C, L = X.shape
        
        # 将 (B, C, L) 重塑为 (g, d, m) 其中 g=1, d=L, m=B*C
        # 这是为了在时间维度 L 上进行解相关
        x_reshaped = X.permute(2, 0, 1).contiguous().view(1, L, B * C)
        g, d, m = x_reshaped.shape
        
        if self.training:
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
            
            # 3. 在无梯度的上下文中更新全局统计量
            with torch.no_grad():
                self.running_mean.copy_(self.momentum * mean.detach() + (1. - self.momentum) * self.running_mean)
                self.running_wm.copy_(self.momentum * wm.detach() + (1. - self.momentum) * self.running_wm)
        else:
            # 推理模式，直接使用全局统计量
            mean = self.running_mean
            wm = self.running_wm
            xc = x_reshaped - mean
            
        # 4. 应用白化变换
        xn = wm.matmul(xc) # (1, L, L) @ (1, L, B*C) -> (1, L, B*C)
        
        # 5. 将形状恢复为 (B, C, L)
        Xn = xn.view(L, B, C).permute(1, 2, 0).contiguous()
        
        # 6. 应用仿射变换
        if self.affine:
            return Xn * self.weight + self.bias
        else:
            return Xn

    def extra_repr(self):
        return '{seq_len}, T={T}, eps={eps}, momentum={momentum}, affine={affine}'.format(**self.__dict__)