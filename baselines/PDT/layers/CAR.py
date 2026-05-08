# 模块代码：CAR_Module_Temporal
import torch
import torch.nn as nn
import torch.nn.functional as F

class CAR_Module_Temporal(nn.Module):
    """
    针对时间特征维度的通道级注意力重校准 (Channel-wise Attentive Recalibration for Temporal Features)。
    将Q_in变换后的T个时间特征维度视为“通道”，并学习它们的样本级重要性。
    """
    def __init__(self, num_temporal_features, reduction_ratio=16):
        """
        初始化模块.
        :param num_temporal_features: 时间特征的数量 (即 T 或 configs.seq_len).
        :param reduction_ratio: 瓶颈层的降维比例。
        """
        super(CAR_Module_Temporal, self).__init__()
        bottleneck_features = max(1, num_temporal_features // reduction_ratio)

        # Squeeze 操作: 在 (N, D) 维度上进行全局平均池化
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Excitation 操作: 两个全连接层，作用于T维度
        self.excitation = nn.Sequential(
            nn.Linear(num_temporal_features, bottleneck_features, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_features, num_temporal_features, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播.
        :param x: 输入张量，期望形状为 [Batch, N_vars, D_emb, T_features].
        :return: 经过重校准的张量，形状与输入x相同.
        """
        b, n, d, t = x.shape

        # Permute to treat T as the channel dimension: [B, N, D, T] -> [B, T, N, D]
        x_permuted = x.permute(0, 3, 1, 2)

        # Squeeze: [B, T, N, D] -> [B, T, 1, 1]
        y = self.squeeze(x_permuted).view(b, t)

        # Excitation: [B, T] -> [B, T]
        y = self.excitation(y)

        # Recalibration: [B, T] -> [B, T, 1, 1] and multiply
        y = y.view(b, t, 1, 1)
        x_recalibrated = x_permuted * y.expand_as(x_permuted)
        
        # Permute back to original shape: [B, T, N, D] -> [B, N, D, T]
        output = x_recalibrated.permute(0, 2, 3, 1)
        
        return output


# 模块代码：GFM_Module_Temporal
# 建议将此模块代码保存为一个单独的文件（如 layers/GFM.py）或直接添加到 OLinear.py 的顶部



class GFM_Module_Temporal(nn.Module):
    """
    针对时间特征维度的门控特征调制 (Gated Feature Modulation for Temporal Features)。
    为Q_in变换后的T维特征图中的每个点学习一个独立的门控权重，以实现细粒度的信息流控制。
    """
    def __init__(self, num_temporal_features):
        """
        初始化模块.
        :param num_temporal_features: 时间特征的数量 (即 T 或 configs.seq_len).
        """
        super(GFM_Module_Temporal, self).__init__()
        # 定义一个线性层，用于从输入特征生成门控信号。
        # 它作用于最后一个维度（T维时间特征）。
        self.gate_generator = nn.Linear(num_temporal_features, num_temporal_features)

    def forward(self, x):
        """
        前向传播.
        :param x: 输入张量，期望形状为 [Batch, N_vars, D_emb, T_features].
        :return: 经过门控调制的张量，形状与输入x相同.
        """
        # 输入形状: [B, N, D, T]
        
        # 1. 生成门控信号
        # 通过线性变换和Sigmoid激活函数生成与x形状完全相同的门控张量
        # Sigmoid将门控值约束在 (0, 1) 范围内
        gate = torch.sigmoid(self.gate_generator(x))
        
        # 2. 应用门控调制
        # 将学习到的门控权重与原始输入进行逐元素相乘
        modulated_x = x * gate
        
        # （可选）添加残差连接以稳定训练，允许模块学习恒等映射
        # output = x + modulated_x
        # 这里我们返回纯粹的调制结果，与CAR模块保持一致
        output = modulated_x
        
        return output


class PLA_DiagGate_Temporal(nn.Module):
    """
    纯对角“近恒等”门：
      - 仅逐-k 通道缩放（正值），不引入跨-k 混合
      - 恒等初始化（初始即不改变输入）
      - 幅度受限：gamma ∈ (1-ε, 1+ε)
    约定输入/输出: [B, N, D, k]
    """
    def __init__(self, k_dim, film_hidden=None, eps=0.2, dropout=0.0):
        super().__init__()
        self.k = k_dim
        self.eps = float(eps)
        h = film_hidden if film_hidden is not None else max(8, self.k // 4)
        self.drop = nn.Dropout(dropout)

        # 用 (mean_k, std_k) 作为样本统计，生成逐-k 的缩放因子增量 Δγ
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.k, h), nn.GELU(),
            nn.Linear(h, self.k)
        )
        # —— 恒等初始化：末层置零 → Δγ=0 → γ=1
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):  # x: [B,N,D,k]
        B, N, D, K = x.shape
        assert K == self.k

        # 统计沿 (N,D) 聚合：得到每个样本在 k 轴上的均值与方差尺度
        mean_k = x.mean(dim=(1, 2))                      # [B,k]
        std_k  = x.std (dim=(1, 2), unbiased=False)      # [B,k]
        stats  = torch.cat([mean_k, std_k], dim=-1)      # [B,2k]

        # 生成 Δγ 并限制幅度：Δγ = ε * tanh(·) ∈ (-ε, ε)
        delta_gamma = torch.tanh(self.mlp(stats)) * self.eps   # [B,k]
        gamma = 1.0 + delta_gamma                              # [B,k] 正值且近1

        # 广播并仅做逐-k 乘法；无偏置、无混合
        y = x * gamma.view(B, 1, 1, K)
        return self.drop(y)

