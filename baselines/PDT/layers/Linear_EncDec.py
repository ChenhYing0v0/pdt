from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidThreshold(nn.Module):
    def __init__(self, threshold: float = 0.5, sharpness_k: float = 10.0) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.tensor(threshold))
        self.register_buffer("sharpness_k", torch.tensor(sharpness_k))

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.sharpness_k * (probabilities - self.threshold))


class Mahalanobis_mask(nn.Module):
    def __init__(self, seq_len: int, threshold: float = 0.25, sharpness_k: float = 200.0) -> None:
        super().__init__()
        self.feature_weights = nn.Parameter(torch.ones(seq_len), requires_grad=True)
        self.thresholder = SigmoidThreshold(threshold=threshold, sharpness_k=sharpness_k)

    def calculate_prob_distance(self, inputs: torch.Tensor) -> torch.Tensor:
        x1 = inputs.unsqueeze(2)
        x2 = inputs.unsqueeze(1)
        diff = x1 - x2
        weights = F.softplus(self.feature_weights)
        dist = torch.sum(weights * (diff ** 2), dim=-1)

        exp_dist = 1 / (dist + 1e-10)
        off_diag_mask = (1 - torch.eye(exp_dist.shape[-1], device=exp_dist.device)).unsqueeze(0)
        exp_dist = exp_dist * off_diag_mask

        exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True)
        probs = exp_dist / (exp_max.detach() + 1e-10)

        diag = torch.eye(probs.shape[-1], device=probs.device).unsqueeze(0)
        return (probs * off_diag_mask + diag) * 0.99

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probabilities = self.calculate_prob_distance(inputs)
        return self.thresholder(probabilities).unsqueeze(1)


class Encoder_ori(nn.Module):
    def __init__(
        self,
        attn_layers,
        conv_layers=None,
        norm_layer=None,
        one_output: bool = False,
        CKA_flag: bool = False,
    ) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.one_output = one_output

    def forward(self, x: torch.Tensor, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        if self.one_output:
            return x
        return x, attns


class LinearEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        CovMat=None,
        dropout: float = 0.1,
        activation: str = "relu",
        token_num: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        if token_num is None:
            raise ValueError("token_num is required for LinearEncoder.")

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        init_weight = torch.eye(token_num) + torch.randn(token_num, token_num)
        self.weight_mat = nn.Parameter(init_weight[None, :, :])
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm2 = nn.LayerNorm(d_model)

    def _compute_attention_base(self) -> torch.Tensor:
        return F.softplus(self.weight_mat)

    def get_attention_matrix(self) -> torch.Tensor:
        return F.normalize(self._compute_attention_base(), p=1, dim=-1).detach()

    def forward(self, x: torch.Tensor, attn_mask=None, **kwargs):
        batch_size = x.shape[0]
        values = self.v_proj(x)
        attn = self._compute_attention_base()

        if attn_mask is not None:
            if attn_mask.dim() == 4:
                attn_mask = attn_mask.squeeze(1)
            attn = attn * attn_mask
        else:
            attn = attn.expand(batch_size, -1, -1)

        attn = F.normalize(attn, p=1, dim=-1)
        context = torch.bmm(attn, values)
        x = x + self.dropout(self.out_proj(context))
        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, -2))))
        y = self.dropout(self.conv2(y).transpose(-1, -2))
        out = self.norm2(x + y)
        return out, attn
