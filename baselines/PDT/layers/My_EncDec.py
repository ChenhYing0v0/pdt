import torch
import torch.nn as nn
import torch.nn.functional as F


import random
from typing import List

from utils.CKA import CudaCKA





class Encoder_ori(nn.Module):
    def __init__(self, inner_layer, norm_layer=None):
        super(Encoder_ori, self).__init__()
        self.inner_layers = nn.ModuleList(inner_layer)
        self.norm = norm_layer


    def forward(self, x):
        # x [B, nvars, D]
        X0 = None  # to make Pycharm happy
        layer_len = len(self.inner_layers)
        for i, inner_layer in enumerate(self.inner_layers):
            x = inner_layer(x)


        if isinstance(x, tuple) or isinstance(x, List):
            x = x[0]

        if self.norm is not None:
            x = self.norm(x)

        return x



class LinearEncoder(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="relu", **kwargs):
        super(LinearEncoder, self).__init__()

        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        # x.shape: b, l, d_model
        

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)

        return output