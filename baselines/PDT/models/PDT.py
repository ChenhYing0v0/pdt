from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from layers.Linear_EncDec import Encoder_ori, LinearEncoder, Mahalanobis_mask
from layers.RevIN import RevIN


def _load_npy(path: str, root_path: str | None = None, device: str = "cpu") -> torch.Tensor:
    resolved = path if os.path.isfile(path) else os.path.join(root_path or "", path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"File not found: {path}")
    array = np.load(resolved)
    return torch.from_numpy(array).to(torch.float32).to(device)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.hidden_size = self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.k_top = configs.k_top
        self.Q_chan_indep = configs.Q_chan_indep

        q_path = configs.Q_MAT_file if self.Q_chan_indep else configs.q_mat_file
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        q_in = _load_npy(q_path, root_path=configs.root_path, device=device)
        if self.Q_chan_indep:
            assert q_in.ndim == 3 and q_in.shape[0] == self.enc_in and q_in.shape[1] == self.seq_len
            self.r = q_in.shape[2]
        else:
            assert q_in.ndim == 2 and q_in.shape[0] == self.seq_len
            self.r = q_in.shape[1]
        self.register_buffer("Q_in", q_in)

        r_path = configs.R_MAT_file if self.Q_chan_indep else configs.r_mat_file
        r_mat = _load_npy(r_path, root_path=configs.root_path, device=device)
        if self.Q_chan_indep:
            assert r_mat.ndim == 3 and r_mat.shape == (self.enc_in, self.r, self.pred_len)
        else:
            assert r_mat.ndim == 2 and r_mat.shape == (self.r, self.pred_len)

        rk_path = configs.Rk_MAT_file if self.Q_chan_indep else configs.rk_mat_file
        rk_mat = _load_npy(rk_path, root_path=configs.root_path, device=device)
        if self.Q_chan_indep:
            assert rk_mat.ndim == 3 and rk_mat.shape == (self.enc_in, self.k_top, self.pred_len)
        else:
            assert rk_mat.ndim == 2 and rk_mat.shape == (self.k_top, self.pred_len)
        self.Rk_param = nn.Parameter(rk_mat.clone())

        self.freeze_R = configs.freeze_R
        if self.freeze_R:
            self.register_buffer("R_fix", r_mat)
        else:
            self.R_param = nn.Parameter(r_mat.clone())

        q_out_path = configs.Q_OUT_MAT_file if self.Q_chan_indep else configs.q_out_mat_file
        if q_out_path is not None:
            q_out = _load_npy(q_out_path, root_path=configs.root_path, device=device)
            self.register_buffer("Q_out_mat", q_out)
        else:
            self.Q_out_mat = None

        self.mask_generator = Mahalanobis_mask(
            self.k_top,
            threshold=configs.mask_threshold,
            sharpness_k=configs.mask_sharpness_k,
        )

        self.embed_size = configs.embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.fc = nn.Sequential(
            nn.Linear(self.pred_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len),
        )

        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)
        self.encoder = Encoder_ori(
            [
                LinearEncoder(
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    token_num=self.enc_in,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            one_output=True,
            CKA_flag=bool(configs.CKA_flag),
        )
        self.linear_head = nn.Linear(self.r * self.embed_size, self.d_model)
        self.linear_tail = nn.Linear(self.d_model, self.r * self.embed_size)

        self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, self.r))
        self.delta2 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.pred_len))
        self.delta3 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.pred_len))

        self.alpha = configs.alpha_init
        self.norm_D = nn.LayerNorm(self.embed_size)
        self._last_channel_mask: torch.Tensor | None = None

    def tokenEmb(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        x = x.transpose(-1, -2).unsqueeze(-1)
        return x * embeddings

    def get_alpha(self) -> torch.Tensor:
        return torch.tensor(float(self.alpha))

    def save_linear_encoder_A(self, output_dir: str) -> None:
        layer = self.encoder.attn_layers[0]
        matrix = layer.get_attention_matrix().detach().cpu().numpy()
        np.save(Path(output_dir) / "linear_encoder_A.npy", matrix)

    def save_channel_mask(self, output_dir: str) -> None:
        if self._last_channel_mask is not None:
            np.save(Path(output_dir) / "channel_mask.npy", self._last_channel_mask.numpy())

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x = self.revin_layer(x, mode="norm")
        x_ori = x.transpose(-1, -2)

        if self.Q_chan_indep:
            z_k = torch.einsum("bnt,ntr->bnr", x_ori, self.Q_in) + self.delta1
        else:
            z_k = torch.einsum("bnt,tr->bnr", x_ori, self.Q_in) + self.delta1
        x_ori = z_k.transpose(-1, -2)

        channel_mask = self.mask_generator(x_ori.transpose(-1, -2)[:, :, : self.k_top])
        self._last_channel_mask = channel_mask.detach().cpu()

        x = self.tokenEmb(x_ori, self.embeddings)
        batch_size, channel_num, rank_dim, embed_dim = x.shape
        assert rank_dim == self.r
        x_trans = x.transpose(-1, -2)

        x_trans = self.linear_head(x_trans.flatten(-2))
        x_trans = self.encoder(x=x_trans, attn_mask=channel_mask)
        x_trans = self.linear_tail(x_trans).reshape(batch_size, channel_num, embed_dim, self.r)

        x_k = x[:, :, : self.k_top, :]
        x_k_dk = x_k.permute(0, 1, 3, 2)
        if self.Q_chan_indep:
            z_key_rd = torch.einsum("bnkd,nkh->bndh", x_k_dk.transpose(-1, -2), self.Rk_param) + self.delta3
        else:
            z_key_rd = torch.einsum("bnkd,kh->bndh", x_k_dk.transpose(-1, -2), self.Rk_param) + self.delta3

        if self.freeze_R:
            if self.Q_chan_indep:
                x = torch.einsum("bnrd,nrh->bndh", x_trans.transpose(-1, -2), self.R_fix)
            else:
                x = torch.einsum("bnrd,rh->bndh", x_trans.transpose(-1, -2), self.R_fix) + self.delta2
        else:
            if self.Q_chan_indep:
                x = torch.einsum("bnrd,nrh->bndh", x_trans.transpose(-1, -2), self.R_param)
            else:
                x = torch.einsum("bnrd,rh->bndh", x_trans.transpose(-1, -2), self.R_param)

        x = self.alpha * z_key_rd + (1.0 - self.alpha) * x
        x = self.norm_D(x.transpose(-1, -2))
        out = self.fc(x.flatten(-2)).transpose(-1, -2)
        out = self.dropout(out)
        return self.revin_layer(out, mode="denorm")
