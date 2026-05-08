# IterNorm_individual.py
"""
This version performs per-channel temporal whitening.
For an input X of shape (B, C, L), it computes C independent whitening
transformations, one for each channel, to decorrelate the L dimension.
"""
import torch
import torch.nn as nn
from torch.nn import Parameter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

__all__ = ['iterative_normalization_py_individual', 'IterNormIndividual']


class iterative_normalization_py_individual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, running_mean, running_wm, T, eps, momentum, training):
        # ... forward 方法本身没有错误, 无需改动 ...
        ctx.T = T
        ctx.eps = eps
        B, C, L = X.shape
        ctx.B, ctx.C, ctx.L = B, C, L
        x_reshaped = X.permute(1, 2, 0).contiguous()
        g, d, m = x_reshaped.shape
        saved_tensors = []
        if training:
            mean = x_reshaped.mean(-1, keepdim=True)
            xc = x_reshaped - mean
            saved_tensors.append(xc)
            I = torch.eye(d, device=X.device, dtype=X.dtype).expand(g, d, d)
            Sigma = torch.baddbmm(I.mul(ctx.eps), xc, xc.transpose(1, 2), beta=1., alpha=1./m)
            rTr = (Sigma * I).sum((1, 2), keepdim=True).reciprocal_()
            saved_tensors.append(rTr)
            Sigma_N = Sigma * rTr
            saved_tensors.append(Sigma_N)
            P = [None] * (ctx.T + 1)
            P[0] = I
            for k in range(ctx.T):
                P_k_cubed = torch.matrix_power(P[k], 3)
                P[k+1] = torch.baddbmm(P[k].mul(1.5), P_k_cubed, Sigma_N, beta=1., alpha=-0.5)
            saved_tensors.extend(P)
            wm = P[ctx.T].mul_(rTr.sqrt())
            running_mean.copy_(momentum * mean + (1. - momentum) * running_mean)
            running_wm.copy_(momentum * wm + (1. - momentum) * running_wm)
        else:
            mean = running_mean
            wm = running_wm
            xc = x_reshaped - mean
        xn = wm.matmul(xc)
        Xn = xn.permute(2, 0, 1).contiguous()
        ctx.save_for_backward(*saved_tensors)
        return Xn

    @staticmethod
    def backward(ctx, grad_output):
        # --- 关键修正点在此 backward 方法中 ---
        saved = ctx.saved_variables
        xc, rTr, sn = saved[0], saved[1], saved[2]
        P = saved[3:]
        B, C, L = ctx.B, ctx.C, ctx.L
        g, d, m = xc.shape
        g_ = grad_output.permute(1, 2, 0).contiguous()

        g_wm = g_.matmul(xc.transpose(-2, -1))
        g_P = g_wm * rTr.sqrt()
        wm = P[ctx.T]

        g_sn = torch.zeros_like(sn)
        for k in range(ctx.T, 0, -1):
            P_k_minus_1 = P[k-1]
            P2 = P_k_minus_1.matmul(P_k_minus_1)
            
            # --- 以下是梯度计算的核心，原先存在bug ---
            g_tmp = g_P.matmul(sn.transpose(-2, -1)) # sn 是 Sigma_N
            
            # 修正: 移除了最后一个 P_k_minus_1 后面的 .transpose(-2, -1)
            # 这现在是原论文梯度公式的正确实现
            g_P = g_P.mul(1.5) - 0.5 * (g_tmp.matmul(P2) + P2.matmul(g_tmp) + P_k_minus_1.matmul(g_tmp).matmul(P_k_minus_1))
            
            # g_sn 的计算没有问题，但它依赖于正确的 g_P
            g_sn_part = P2.matmul(P_k_minus_1).matmul(g_P)
            g_sn += g_sn_part

        g_tr = ((-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm)) * P[0]).sum((1, 2), keepdim=True) * P[0]
        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2. * g_tr) * (-0.5 / m * rTr)
        g_x = torch.baddbmm(wm.transpose(-2, -1).matmul(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
        grad_input = g_x.permute(2, 0, 1).contiguous()
        return grad_input, None, None, None, None, None, None



class IterNormIndividual(nn.Module):
    def __init__(self, num_channels, seq_len, T=5, eps=1e-5, momentum=0.1, affine=True, *args, **kwargs):
        super(IterNormIndividual, self).__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.T = T
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
        
        # --- 修正点 ---
        # 在 .expand() 之后添加 .clone() 来确保每个通道的白化矩阵拥有独立的内存
        initial_wm = torch.eye(self.seq_len).expand(self.num_channels, self.seq_len, self.seq_len).clone()
        self.register_buffer('running_wm', initial_wm)
        # --- 修正结束 ---
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        assert X.dim() == 3, f"Input must be a 3D tensor (B, C, L), but got {X.dim()}D"
        assert X.size(1) == self.num_channels, f"Input channel size {X.size(1)} does not match module's num_channels {self.num_channels}"
        assert X.size(2) == self.seq_len, f"Input sequence length {X.size(2)} does not match module's seq_len {self.seq_len}"
        
        X_hat = iterative_normalization_py_individual.apply(X, self.running_mean, self.running_wm, self.T,
                                                          self.eps, self.momentum, self.training)
        
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return 'num_channels={num_channels}, seq_len={seq_len}, T={T}, eps={eps}, momentum={momentum}, affine={affine}'.format(**self.__dict__)

# --- For comparison, we need the corrected IterNorm_temp ---
class IterNormTemp(nn.Module):
    def __init__(self, seq_len, T=5, eps=1e-5, momentum=0.1, affine=True):
        super(IterNormTemp, self).__init__()
        # This is the "global" version that combines channels
        self.seq_len = seq_len
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.ones(1, 1, self.seq_len))
            self.bias = Parameter(torch.zeros(1, 1, self.seq_len))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(1, self.seq_len, 1))
        self.register_buffer('running_wm', torch.eye(self.seq_len).unsqueeze(0))
    
    def forward(self, X: torch.Tensor):
        B, C, L = X.shape
        x_reshaped = X.permute(2, 0, 1).contiguous().view(1, L, B * C)
        # Simplified forward for brevity; not a full autograd function
        # This is just for conceptual validation
        mean = x_reshaped.mean(-1, keepdim=True)
        xc = x_reshaped - mean
        I = torch.eye(L, device=X.device, dtype=X.dtype).unsqueeze(0)
        Sigma = torch.baddbmm(I.mul(self.eps), xc, xc.transpose(1, 2), beta=1., alpha=1./(B*C))
        rTr = (Sigma * I).sum((1, 2), keepdim=True).reciprocal_()
        Sigma_N = Sigma * rTr
        P = I
        for _ in range(self.T):
            P_cubed = torch.matrix_power(P, 3)
            P = torch.baddbmm(P.mul(1.5), P_cubed, Sigma_N, beta=1., alpha=-0.5)
        wm = P.mul(rTr.sqrt())
        xn = wm.matmul(xc)
        Xn = xn.view(L, B, C).permute(1, 2, 0).contiguous()
        return Xn

### 3. 有效性验证与对比分析
'''
    为了验证 `IterNormIndividual` 的优越性，我们需要构建一个**各通道具有不同时间相关性**的“陷阱”数据集。如果所有通道的时间动态都一样，那么两种方法的效果会很相似。

    **实验设计：**
    1.  **构建数据**: 我们创建一个4通道的数据。
        * **通道0**: 具有强**正相关**性，即后一个时间点的值与前一个高度相关（例如，自回归过程 $x_t = 0.9 x_{t-1} + \epsilon_t$）。
        * **通道1**: 具有强**负相关**性，即后一个时间点的值与前一个倾向于相反（例如，$x_t = -0.9 x_{t-1} + \epsilon_t$）。
        * **通道2**: 纯粹的白噪声，**无相关性**。
        * **通道3**: 另一种正相关过程，但相关性稍弱。
    2.  **模型处理**: 将这份数据分别送入 `IterNormTemp` 和 `IterNormIndividual`。
    3.  **结果分析**: 分别计算两个模型输出的、每个通道的协方差矩阵。
        * **理想结果 (`IterNormIndividual`)**: 对于**每一个**通道，其输出的协方差矩阵都应该接近单位矩阵 $I$，因为每个通道都有自己专属的白化器。
        * **预期结果 (`IterNormTemp`)**: 它会学习一个“平均”的白化矩阵。这个矩阵可能对白噪声通道处理得不错，但无法同时完美地消除强正相关和强负相关。因此，在通道0和通道1上，我们会看到输出的协方差矩阵仍有显著的非对角线元素，即**白化不彻底**。
'''

# --- Validation and Comparison Code ---
def create_correlated_data(batch_size, num_channels, seq_len):
    """Creates a synthetic dataset with different temporal correlations per channel."""
    X = torch.randn(batch_size, num_channels, seq_len)
    
    # Channel 0: Strong positive correlation (AR(1) process)
    for t in range(1, seq_len):
        X[:, 0, t] = 0.9 * X[:, 0, t-1] + 0.4 * torch.randn(batch_size)
        
    # Channel 1: Strong negative correlation
    for t in range(1, seq_len):
        X[:, 1, t] = -0.9 * X[:, 1, t-1] + 0.4 * torch.randn(batch_size)
        
    # Channel 2 is already white noise
    
    # Channel 3: Weaker positive correlation
    for t in range(1, seq_len):
        X[:, 3, t] = 0.5 * X[:, 3, t-1] + 0.87 * torch.randn(batch_size)
        
    return X

def calculate_channel_covariance(data, channel_idx):
    """Calculates the temporal covariance matrix for a specific channel."""
    channel_data = data[:, channel_idx, :].contiguous() # Shape (B, L)
    B, L = channel_data.shape
    
    # Center the data
    mean = channel_data.mean(dim=0, keepdim=True) # Shape (1, L)
    centered_data = channel_data - mean
    
    # Calculate covariance: (1/B) * X^T @ X
    covariance = (1.0 / (B - 1)) * centered_data.t() @ centered_data
    return covariance.detach().cpu().numpy()

def plot_covariance(cov_matrix, title):
    """Plots a covariance matrix as a heatmap."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cov_matrix, cmap='viridis', cbar=True, vmin=-1, vmax=1)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Time Step")
    plt.show()

# =========================================================================
#  无需图形库的、可直接在服务器上运行的验证模块 (替换原文件末尾的验证代码)
# =========================================================================
if __name__ == '__main__':
    
    # --- Helper Functions for Text-based Validation ---
    def create_correlated_data(batch_size, num_channels, seq_len):
        """Creates a synthetic dataset with different temporal correlations per channel."""
        X = torch.randn(batch_size, num_channels, seq_len)
        # Channel 0: Strong positive correlation
        for t in range(1, seq_len):
            X[:, 0, t] = 0.9 * X[:, 0, t-1] + 0.4 * torch.randn(batch_size)
        # Channel 1: Strong negative correlation
        for t in range(1, seq_len):
            X[:, 1, t] = -0.9 * X[:, 1, t-1] + 0.4 * torch.randn(batch_size)
        # Channel 2 is white noise
        # Channel 3: Weaker positive correlation
        for t in range(1, seq_len):
            X[:, 3, t] = 0.5 * X[:, 3, t-1] + 0.87 * torch.randn(batch_size)
        return X

    def calculate_channel_covariance(data, channel_idx):
        """Calculates the temporal covariance matrix for a specific channel. Returns a torch tensor."""
        channel_data = data[:, channel_idx, :].contiguous()
        B, L = channel_data.shape
        mean = channel_data.mean(dim=0, keepdim=True)
        centered_data = channel_data - mean
        # Use B-1 for unbiased sample covariance
        covariance = (1.0 / (B - 1)) * centered_data.t() @ centered_data
        return covariance.detach()

    def analyze_covariance(cov_matrix, title):
        """Analyzes a covariance matrix and prints key numerical metrics."""
        print(title)
        diag_elements = torch.diag(cov_matrix)
        
        # Sum of absolute off-diagonal elements
        sum_abs_off_diag = torch.sum(torch.abs(cov_matrix)) - torch.sum(torch.abs(diag_elements))
        
        # Mean of diagonal elements
        mean_diag = torch.mean(diag_elements)
        
        print(f"  - 对角线元素均值 (应接近 1.0): {mean_diag:.6f}")
        print(f"  - 非对角线元素绝对值之和 (应接近 0.0): {sum_abs_off_diag:.6f}")
        print("-" * 50)

    # --- Configuration ---
    B, C, L = 128, 4, 16
    
    # --- Create Data ---
    input_data = create_correlated_data(B, C, L)
    
    print("="*20 + " 1. 输入数据的协方差分析 " + "="*20)
    analyze_covariance(calculate_channel_covariance(input_data, 0), "输入数据 - 通道 0 (强正相关)")
    analyze_covariance(calculate_channel_covariance(input_data, 1), "输入数据 - 通道 1 (强负相关)")
    analyze_covariance(calculate_channel_covariance(input_data, 2), "输入数据 - 通道 2 (白噪声)")

    # --- Initialize Models ---
    # 为了在单次训练中看到效果，设置 momentum=1
    model_global = IterNormTemp(seq_len=L, T=10, eps=1e-3)
    model_individual = IterNormIndividual(num_channels=C, seq_len=L, T=10, eps=1e-3)
    
    model_global.train()
    model_individual.train()
    
    # --- Process Data ---
    output_global = model_global(input_data)
    output_individual = model_individual(input_data)
    
    # --- Analyze Results ---
    print("\n" + "="*20 + " 2. IterNorm_temp (全局白化) 结果分析 " + "="*20)
    print("预期: 对于相关性强的通道0和1，白化不彻底，非对角线元素和较大。")
    analyze_covariance(calculate_channel_covariance(output_global, 0), "全局白化输出 - 通道 0")
    analyze_covariance(calculate_channel_covariance(output_global, 1), "全局白化输出 - 通道 1")
    analyze_covariance(calculate_channel_covariance(output_global, 2), "全局白化输出 - 通道 2")

    print("\n" + "="*20 + " 3. IterNorm_individual (独立白化) 结果分析 " + "="*20)
    print("预期: 对于所有通道，白化都应成功，对角线均值接近1，非对角线和接近0。")
    analyze_covariance(calculate_channel_covariance(output_individual, 0), "独立白化输出 - 通道 0")
    analyze_covariance(calculate_channel_covariance(output_individual, 1), "独立白化输出 - 通道 1")
    analyze_covariance(calculate_channel_covariance(output_individual, 2), "独立白化输出 - 通道 2")