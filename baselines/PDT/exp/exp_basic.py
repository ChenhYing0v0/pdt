from __future__ import annotations

import os

import torch
from torch.utils.tensorboard import SummaryWriter

from models import PDT
from utils.tools import ensure_path


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "PDT": PDT,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.writer = None

        self.epoch = 0
        self.step = 0

        self.output_pred = args.output_pred
        self.output_vis = args.output_vis

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) \
                if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
            print(f"GPU {self.args.gpu}: {torch.cuda.get_device_name(self.args.gpu)}")
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _create_writer(self, log_dir):
        ensure_path(log_dir)
        return SummaryWriter(log_dir)

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
