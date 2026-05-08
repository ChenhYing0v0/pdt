# exp/exp_rrr_linear.py
import os, time, warnings
import numpy as np
import torch
import torch.nn as nn

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics_torch import create_metric_collector
from utils.tools import ensure_path

warnings.filterwarnings('ignore')

class Exp_RRR_Linear(Exp_Basic):
    """
    A1：RRR 纯线性零训练实验。
    步骤：
      1) 用 train_loader 统计窗口列均值 muX/muY（与离线一致）
      2) 直接在 vali/test 上评估，不做优化更新
    """
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len

    def _build_model(self):
        # model 注册名需在主程序处加入：model_dict['RRRLinearZero'] = models.RRRLinear
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def train(self, setting):
        # 取 3 个 loader
        train_data, train_loader = self._get_data('train')
        vali_data,  vali_loader  = self._get_data('val')
        test_data,  test_loader  = self._get_data('test')

        # 创建路径 & 记录器
        path = os.path.join(self.args.checkpoints, setting)
        ensure_path(path)
        res_path = os.path.join(self.args.results, setting)
        ensure_path(res_path)
        self.writer = self._create_writer(res_path)

        # 1) 统计窗口列均值（与离线一致）
        #    注意：这里默认 data_provider 已做了“按训练集的 z-score”预处理，
        #    我们只需对列做均值对齐。
        if isinstance(self.model, nn.DataParallel):
            self.model.module.prepare_all_stats(train_loader, device=self.device)
        else:
            self.model.prepare_all_stats(train_loader, device=self.device)

        # 2) 直接验证/测试
        criterion = nn.MSELoss()
        vali_loss = self.vali(vali_data, vali_loader, criterion)
        test_loss = self.vali(test_data, test_loader, criterion)

        print(f"[RRR-LinearZero] Vali Loss: {vali_loss:.7f} | Test Loss: {test_loss:.7f}")
        # 保存“检查点”（虽然模型无可学习参数）
        torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 与你的标准验证逻辑完全一致（忽略 dec_inp/marks）
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        return float(np.mean(total_loss))

    def test(self, setting, test=0):
        # 复用你现有 Exp_Long_Term_Forecast.test 的结构，简化实现：
        test_data, test_loader = self._get_data('test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))
        self.model.eval()

        metric_collector = create_metric_collector(device=self.device)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                metric_collector.update(outputs, batch_y)

        m = metric_collector.compute()
        print('{}\t| mse:{}, mae:{}'.format(self.pred_len, m["mse"], m["mae"]))
        return
