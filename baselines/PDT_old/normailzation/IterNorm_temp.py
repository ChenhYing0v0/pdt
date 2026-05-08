# IterNorm_temp.py
"""
Reference:  Iterative Normalization: Beyond Standardization towards Efficient Whitening, CVPR 2019
This version is adapted for temporal data (B, C, L) to decorrelate along the L dimension.
"""
import torch
import torch.nn as nn
from torch.nn import Parameter

__all__ = ['iterative_normalization_py', 'IterNormTemp']


class iterative_normalization_py(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, running_mean, running_wmat, T, eps, momentum, training):
        """
        Forward pass for temporal iterative normalization.
        Input X has shape (B, C, L). We decorrelate along the L dimension.
        """
        ctx.T = T
        ctx.eps = eps

        # Reshape (B, C, L) to (g, d, m) where g=1, d=L, m=B*C
        # This treats temporal steps as features to be decorrelated.
        B, C, L = X.shape
        ctx.B, ctx.C, ctx.L = B, C, L
        # permute(2, 0, 1) -> (L, B, C)
        # view(1, L, B * C) -> (g, d, m)
        x_reshaped = X.permute(2, 0, 1).contiguous().view(1, L, B * C)
        
        g, d, m = x_reshaped.shape # g=1, d=L, m=B*C
        saved_tensors = []
        
        if training:
            # Calculate centered activation
            mean = x_reshaped.mean(-1, keepdim=True)  # Shape: (1, L, 1)
            xc = x_reshaped - mean
            saved_tensors.append(xc)

            # Calculate covariance matrix Sigma of shape (1, L, L)
            P = [None] * (ctx.T + 1)
            I = torch.eye(d, device=X.device, dtype=X.dtype).unsqueeze(0) # Shape: (1, d, d)
            P[0] = I

            # Sigma = (1/m) * xc @ xc^T
            Sigma = torch.baddbmm(I.mul(ctx.eps), xc, xc.transpose(1, 2), beta=1., alpha=1./m)
            
            # Reciprocal of trace of Sigma
            rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            saved_tensors.append(rTr)
            
            Sigma_N = Sigma * rTr
            # print('Sigma_N:', Sigma_N.size())
            saved_tensors.append(Sigma_N)

            # Newton-Schulz iteration to compute Sigma^{-1/2}
            for k in range(ctx.T):
                P[k + 1] = torch.baddbmm(P[k].mul(1.5), torch.matrix_power(P[k], 3), Sigma_N, alpha=-0.5, beta=1.)
            
            saved_tensors.extend(P)
            
            # Whitening matrix W = Sigma^{-1/2}
            wm = P[ctx.T].mul_(rTr.sqrt())
            
            # Update running statistics
            running_mean.copy_(momentum * mean + (1. - momentum) * running_mean)
            running_wmat.copy_(momentum * wm + (1. - momentum) * running_wmat)
        else:
            mean = running_mean
            wm = running_wmat
            xc = x_reshaped - mean
            
        # Whiten the data
        xn = wm.matmul(xc)
        
        # Reshape back to (B, C, L)
        # xn has shape (1, L, B*C)
        Xn = xn.view(L, B, C).permute(1, 2, 0).contiguous()
        
        ctx.save_for_backward(*saved_tensors)
        return Xn

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for temporal iterative normalization.
        """
        saved = ctx.saved_tensors
        xc, rTr, sn = saved[0], saved[1], saved[2]
        P = saved[3:]
        
        B, C, L = ctx.B, ctx.C, ctx.L
        g, d, m = xc.shape # g=1, d=L, m=B*C

        # Reshape grad_output from (B, C, L) to (1, L, B*C) to match xc
        g_ = grad_output.permute(2, 0, 1).contiguous().view(g, d, m)

        # Gradient calculation (following the original paper's derivation)
        g_wm = g_.matmul(xc.transpose(-2, -1))
        g_P = g_wm * rTr.sqrt()
        wm = P[ctx.T]

        g_sn = 0
        P_k_minus_1_T = P[ctx.T-1].transpose(-2, -1)
        P_k_minus_1_sq = P_k_minus_1_T.matmul(P[ctx.T-1])

        for k in range(ctx.T, 0, -1):
            if k < ctx.T:
                P_k_minus_1_T = P[k-1].transpose(-2, -1)
                P_k_minus_1_sq = P_k_minus_1_T.matmul(P[k-1])

            g_sn_part = P_k_minus_1_sq.matmul(P[k-1]).matmul(g_P)
            g_sn += g_sn_part

            g_tmp = g_P.matmul(sn.transpose(-2, -1))
            g_P = g_P.mul(1.5) \
                - 0.5 * (g_tmp.matmul(P_k_minus_1_sq) +
                         P_k_minus_1_sq.matmul(g_tmp) +
                         P_k_minus_1_T.matmul(g_tmp).matmul(P[k-1]))
        
        g_tr = ((-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm)) * P[0]).sum((1, 2), keepdim=True) * P[0]
        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2. * g_tr) * (-0.5 / m * rTr)
        
        g_x = torch.baddbmm(wm.transpose(-2, -1).matmul(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)

        # Reshape gradient back to (B, C, L)
        grad_input = g_x.view(L, B, C).permute(1, 2, 0).contiguous()

        return grad_input, None, None, None, None, None, None, None


class IterNormTemp(nn.Module):
    """
    Iterative Normalization for Temporal Data.

    Args:
        seq_len (int): The length of the sequence (L dimension), which will be decorrelated.
        T (int, optional): Number of iterations for Newton-Schulz iteration. Default: 5.
        eps (float, optional): A small value added to the diagonal of covariance for stability. Default: 1e-5.
        momentum (float, optional): The momentum for updating running statistics. Default: 0.1.
        affine (bool, optional): If True, this module has learnable affine parameters. Default: True.
    """
    def __init__(self, seq_len, T=5, eps=1e-5, momentum=0.1, affine=True, *args, **kwargs):
        super(IterNormTemp, self).__init__()
        self.seq_len = seq_len
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        # Affine parameters (weight and bias) are applied per time-step.
        # Shape (1, 1, L) is broadcastable to (B, C, L).
        if self.affine:
            self.weight = Parameter(torch.ones(1, 1, self.seq_len))
            self.bias = Parameter(torch.zeros(1, 1, self.seq_len))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # Running statistics for inference
        # running_mean has shape (1, L, 1)
        self.register_buffer('running_mean', torch.zeros(1, self.seq_len, 1))
        # running_wm is the whitening matrix, shape (1, L, L)
        self.register_buffer('running_wm', torch.eye(self.seq_len).unsqueeze(0))
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        assert X.dim() == 3, f"Input must be a 3D tensor (B, C, L), but got {X.dim()}D"
        assert X.size(2) == self.seq_len, f"Input sequence length {X.size(2)} does not match module's seq_len {self.seq_len}"
        
        X_hat = iterative_normalization_py.apply(X, self.running_mean, self.running_wm, self.T,
                                                 self.eps, self.momentum, self.training)
        
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return '{seq_len}, T={T}, eps={eps}, momentum={momentum}, affine={affine}'.format(**self.__dict__)


if __name__ == '__main__':
    # --- Configuration ---
    batch_size = 32
    num_channels = 7
    seq_length = 16 # The dimension to be decorrelated
    
    # --- Module Initialization ---
    # We set momentum=1 to see the immediate effect of whitening in one pass
    ItN = IterNormTemp(seq_len=seq_length, T=8, momentum=1, affine=False)
    print(ItN)
    
    # --- Training Mode Test ---
    print("\n--- Training Mode ---")
    ItN.train()
    
    # Input tensor of shape (B, C, L)
    x = torch.randn(batch_size, num_channels, seq_length)
    x.requires_grad_()
    
    # Forward pass
    y = ItN(x)
    
    # Verification: The covariance matrix of the output's temporal dimension should be close to identity
    # Reshape y from (B, C, L) to (L, B*C)
    z = y.permute(2, 0, 1).contiguous().view(seq_length, -1)
    
    # Calculate covariance: Cov = (1/m) * z @ z^T
    covariance = torch.matmul(z, z.t()) / z.size(1)

    print('covariance:\n', covariance)
    
    print("Covariance matrix of the temporal dimension (should be close to Identity):")
    # Print diagonal and a few off-diagonal elements for brevity
    print("Diagonal elements (should be ~1.0):\n", torch.diag(covariance))
    print("Sum of absolute off-diagonal elements (should be close to 0):", (torch.sum(torch.abs(covariance)) - torch.sum(torch.abs(torch.diag(covariance)))).item())
    
    # Backward pass test
    y.sum().backward()
    print('Gradient shape w.r.t x:', x.grad.size())
    
    # --- Evaluation Mode Test ---
    print("\n--- Evaluation Mode ---")
    ItN.eval()
    
    # Create new input data
    x_eval = x
    y_eval = ItN(x_eval)
    
    # Verification in eval mode
    z_eval = y_eval.permute(2, 0, 1).contiguous().view(seq_length, -1)
    covariance_eval = torch.matmul(z_eval, z_eval.t()) / z_eval.size(1)

    print("Covariance matrix in eval mode (using running stats):")
    print("Sum of absolute off-diagonal elements:", (torch.sum(torch.abs(covariance_eval)) - torch.sum(torch.abs(torch.diag(covariance_eval)))).item())