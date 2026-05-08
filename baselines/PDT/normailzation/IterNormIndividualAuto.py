# IterNormIndividualAuto.py
# 一个完全依赖 PyTorch Autograd 的版本

import torch
import torch.nn as nn
from torch.nn import Parameter

class IterNormIndividualAuto(nn.Module):
    def __init__(self, num_channels, seq_len, T=5, eps=1e-3, momentum=0.1, affine=True, *args, **kwargs):
        super(IterNormIndividualAuto, self).__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.T = T
        # 增大eps以保证数值稳定性
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        if self.affine:
            self.weight = Parameter(torch.ones(1, self.num_channels, self.seq_len))
            self.bias = Parameter(torch.zeros(1, self.num_channels, self.seq_len))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(self.num_channels, self.seq_len, 1))
        initial_wm = torch.eye(self.seq_len).expand(self.num_channels, self.seq_len, self.seq_len).clone()
        self.register_buffer('running_wm', initial_wm)
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        assert X.dim() == 3, f"Input must be a 3D tensor (B, C, L), but got {X.dim()}D"
        assert X.size(1) == self.num_channels
        assert X.size(2) == self.seq_len
        
        # 将原 autograd.Function.forward 的逻辑直接嵌入此处
        B, C, L = X.shape
        x_reshaped = X.permute(1, 2, 0).contiguous() # (C, L, B)
        
        if self.training:
            # 1. 计算批次统计量
            mean = x_reshaped.mean(-1, keepdim=True)  # (C, L, 1)
            xc = x_reshaped - mean
            
            I = torch.eye(L, device=X.device, dtype=X.dtype).expand(C, L, L)
            Sigma = torch.baddbmm(I.mul(self.eps), xc, xc.transpose(1, 2), beta=1., alpha=1./B)
            
            rTr = (Sigma * I).sum((1, 2), keepdim=True).reciprocal_()
            Sigma_N = Sigma * rTr
            
            # 迭代计算白化矩阵
            P = I.clone()
            for _ in range(self.T):
                P_cubed = torch.matrix_power(P, 3)
                P = torch.baddbmm(P.mul(1.5), P_cubed, Sigma_N, beta=1., alpha=-0.5)
            
            wm = P.mul(rTr.sqrt()) # (C, L, L)
            
            # 2. 更新全局统计量
            # 使用 .detach() 来更新 running_mean 和 running_wm，
            # 因为这些统计量的更新过程不应计入反向传播的计算图中。
            with torch.no_grad():
                self.running_mean.copy_(self.momentum * mean + (1. - self.momentum) * self.running_mean)
                self.running_wm.copy_(self.momentum * wm + (1. - self.momentum) * self.running_wm)
        else:
            # 推理模式，直接使用全局统计量
            mean = self.running_mean
            wm = self.running_wm
            xc = x_reshaped - mean
            
        # 3. 应用白化变换
        xn = wm.matmul(xc) # (C, L, L) @ (C, L, B) -> (C, L, B)
        Xn = xn.permute(2, 0, 1).contiguous() # (B, C, L)
        
        # 4. 应用仿射变换
        if self.affine:
            return Xn * self.weight + self.bias
        else:
            return Xn

    def extra_repr(self):
        return 'num_channels={num_channels}, seq_len={seq_len}, T={T}, eps={eps}, momentum={momentum}, affine={affine}'.format(**self.__dict__)