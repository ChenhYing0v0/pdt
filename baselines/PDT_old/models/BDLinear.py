import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from normailzation.IterNormTempAuto import IterNormTempAuto
from normailzation.IterNormIndividualAuto import IterNormIndividualAuto




class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.embed_dim = configs.d_model
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.revin = configs.revin
        self.add_module = configs.add_module
        self.norm_type = configs.norm_type

        # 记录T的值
        self.T = getattr(configs, 'T', None)
        # 分阶段保存
        self.save_cov = configs.save_cov
        if configs.save_cov:
            self._init_list_train = []
            self._cov_list_train = []
            self._init_list_eval = []
            self._cov_list_eval = []
            self.is_test = False  # 标记当前是否为test阶段

        if self.individual:
            if configs.iter_norm:
                self.iter_norm = nn.ModuleList([
                    IterNormTempAuto(self.seq_len, T=configs.T, eps=configs.eps, momentum=configs.momentum, affine=configs.affine)
                    for _ in range(self.channels)
                ])
            else:
                self.iter_norm = None
            self.Linear = nn.ModuleList()
            if self.add_module == 'embed':
                self.embed = nn.ModuleList([
                    nn.Linear(self.seq_len, self.embed_dim)
                    for _ in range(self.channels)
                ])
                self.Linear = nn.ModuleList([
                    nn.Linear(self.embed_dim, self.pred_len)
                    for _ in range(self.channels)
                ])
            else:
                self.Linear = nn.ModuleList([
                    nn.Linear(self.seq_len, self.pred_len)
                    for _ in range(self.channels)
                ])
            # for i in range(self.channels):
            #     if self.add_module == 'embed':
            #         self.Linear[i].weight = nn.Parameter(
            #             (1 / self.embed_dim) * torch.ones([self.pred_len, self.embed_dim]))
            #     else:
            #         self.Linear[i].weight = nn.Parameter(
            #             (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        else:
            if configs.iter_norm:
                if self.norm_type == 'mlp':
                    self.iter_norm = IterNormTempAuto(self.seq_len, T=configs.T, eps=configs.eps, momentum=configs.momentum, affine=configs.affine)
                elif self.norm_type == 'ind':
                    self.iter_norm = IterNormIndividualAuto(num_channels=self.channels, seq_len=self.seq_len, T=configs.T, eps=configs.eps, momentum=configs.momentum, affine=configs.affine)
            else:
                self.iter_norm = None
            
            if self.add_module == 'embed':
                self.embed = nn.Linear(self.seq_len, self.embed_dim)
                self.Linear = nn.Linear(self.embed_dim, self.pred_len)
            else:
                self.Linear = nn.Linear(self.seq_len, self.pred_len)

            # scale = 1 / self.seq_len
            # self.Linear.weight = nn.Parameter(scale * torch.ones([self.pred_len, self.embed_dim]))


        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)
        # 新增：用于累积所有batch的init和协方差
        self._init_list = []
        self._cov_list = []

    def encoder(self, x):
        # x shape: [B, L, N]
        # 在N维度进行RevIN标准化
        if self.revin:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True)
            x = (x - mean) / (std + 1e-5)

        init = x.permute(0, 2, 1)   # [B, N, L]
        
        if self.individual:
            output = torch.zeros(
                [init.size(0), init.size(1), self.pred_len], dtype=init.dtype
            ).to(init.device)
            for i in range(self.channels):
                cur = init[:, i, :]
                if self.iter_norm:
                    cur = self.iter_norm[i](cur.unsqueeze(1)).squeeze(1)  # [B, L] -> [B, 1, L] -> [B, L, 1] -> [B, L]
                if self.add_module == 'embed':
                    cur = self.embed[i](cur)
                cur = self.Linear[i](cur)
                output[:, i, :] = cur
        else:
            if self.iter_norm:
                init = self.iter_norm(init)     # input shape: [B, N, D]
            # 只在train和test阶段保存
            if self.save_cov:
                if self.training:
                    self._save_init_for_covariance_check(init, stage='train')
                elif getattr(self, 'is_test', False):
                    self._save_init_for_covariance_check(init, stage='test')
            # valid阶段不保存
            if self.add_module == 'embed':
                init = self.embed(init)     # [B, N, D]
            output = self.Linear(init)

        x = output

        # 在N维度进行RevIN逆标准化
        if self.revin:
            x = x.permute(0, 2, 1)
            x = x * (std + 1e-5) + mean
            x = x.permute(0, 2, 1)

        return x.permute(0, 2, 1)

    def _save_init_for_covariance_check(self, init, stage="train"):
        """将每个batch的init和协方差累积到列表，训练结束后统一保存，stage可选'train'或'eval'"""
        try:
            init_np = init.detach().cpu().numpy()  # shape: (B, N, L)
            B, N, L = init_np.shape
            init_reshaped = init_np.reshape(-1, L)  # [B*N, L]
            cov_matrix = np.cov(init_reshaped.T)  # [L, L]
            if stage == "train":
                self._init_list_train.append(init_np)
                self._cov_list_train.append(cov_matrix)
            elif stage == "test":
                self._init_list_eval.append(init_np)
                self._cov_list_eval.append(cov_matrix)
            else:
                print(f"未知stage: {stage}")
        except Exception as e:
            print(f"保存init值时出错: {e}")

    def save_all_inits_and_covs(self, attach_dir="attach", stage="train"):
        """训练或评估结束后调用，将所有batch的init和协方差保存为npy文件，stage可选'train'或'eval'"""
        try:
            if not os.path.exists(attach_dir):
                os.makedirs(attach_dir)
            if stage == "train":
                init_list = self._init_list_train
                cov_list = self._cov_list_train
            elif stage == "test":
                init_list = self._init_list_eval
                cov_list = self._cov_list_eval
            else:
                print(f"未知stage: {stage}")
                return
            if not init_list:
                print(f"没有可保存的{stage}阶段init数据！")
                return
            # 只保留shape与第一个batch一致的部分
            first_shape = init_list[0].shape
            valid_indices = [i for i, arr in enumerate(init_list) if arr.shape == first_shape]
            valid_inits = [init_list[i] for i in valid_indices]
            valid_covs = [cov_list[i] for i in valid_indices]
            # 根据是否使用iterNorm决定文件名后缀
            if self.iter_norm is None:
                suffix = "noIterNorm"
            else:
                suffix = "iterNorm"
            # 文件名中加入T的值和stage
            T_str = f"_T{self.T}" if self.T is not None else ""
            stage_str = f"_{stage}"
            # 保存所有batch的init
            all_inits = np.stack(valid_inits, axis=0)  # shape: (num_batches, B, N, L)
            np.save(os.path.join(attach_dir, f"init_after_{suffix}{T_str}{stage_str}_all.npy"), all_inits)
            # 保存所有batch的协方差矩阵
            all_covs = np.stack(valid_covs, axis=0)  # shape: (num_batches, L, L)
            np.save(os.path.join(attach_dir, f"covariance_matrix_{suffix}{T_str}{stage_str}_all.npy"), all_covs)
        except Exception as e:
            print(f"保存所有batch的init和协方差时出错: {e}")

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
