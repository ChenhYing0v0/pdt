from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features


class Dataset_ETT_hour(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
        add_noise=False,
        noise_amp=0.1,
        noise_freq_percentage=0.05,
        noise_seed=2023,
        noise_type="sin",
        data_percentage=1.0,
        **kwargs,
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in {"train", "val", "test"}
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self._read_data()

    def _read_data(self) -> None:
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in {"M", "MS"}:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.iloc[border1s[0]:border2s[0]].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]].iloc[border1:border2].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp["date"])
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp["date"].dt.month
            df_stamp["day"] = df_stamp["date"].dt.day
            df_stamp["weekday"] = df_stamp["date"].dt.weekday
            df_stamp["hour"] = df_stamp["date"].dt.hour
            data_stamp = df_stamp.drop(columns=["date"]).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq).transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len
        return (
            self.data_x[s_begin:s_end],
            self.data_y[r_begin:r_end],
            self.data_stamp[s_begin:s_end],
            self.data_stamp[r_begin:r_end],
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
        seasonal_patterns=None,
        add_noise=False,
        noise_amp=0.1,
        noise_freq_percentage=0.05,
        noise_seed=2023,
        noise_type="sin",
        data_percentage=1.0,
        **kwargs,
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in {"train", "val", "test"}
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self._read_data()

    def _read_data(self) -> None:
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in {"M", "MS"}:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.iloc[border1s[0]:border2s[0]].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]].iloc[border1:border2].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp["date"])
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp["date"].dt.month
            df_stamp["day"] = df_stamp["date"].dt.day
            df_stamp["weekday"] = df_stamp["date"].dt.weekday
            df_stamp["hour"] = df_stamp["date"].dt.hour
            df_stamp["minute"] = df_stamp["date"].dt.minute // 15
            data_stamp = df_stamp.drop(columns=["date"]).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq).transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len
        return (
            self.data_x[s_begin:s_end],
            self.data_y[r_begin:r_end],
            self.data_stamp[s_begin:s_end],
            self.data_stamp[r_begin:r_end],
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="electricity.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
        add_noise=False,
        noise_amp=0.1,
        noise_freq_percentage=0.05,
        noise_seed=2023,
        noise_type="sin",
        data_percentage=1.0,
        **kwargs,
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in {"train", "val", "test"}
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.add_noise = add_noise
        self.noise_amp = noise_amp
        self.noise_freq_percentage = noise_freq_percentage
        self.noise_seed = noise_seed
        self.noise_type = noise_type
        self.data_percentage = data_percentage
        self.root_path = root_path
        self.data_path = data_path
        self._read_data()

    def _read_data(self) -> None:
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.set_type == 0:
            print("Head lines of raw dataframe:")
            print(df_raw.head(5))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0 if (num_train + num_vali) % 2 == 0 else 1]
        border1s += [num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0 and self.data_percentage < 1.0:
            print(f"Shrink the train data to {self.data_percentage * 100}%")
            border1 = border2 - int((border2 - border1) * self.data_percentage)

        if self.features in {"M", "MS"}:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        if self.add_noise and self.noise_amp > 0:
            tmp_data = df_data.iloc[border1s[0]:border2s[1]].copy()
            freq_domain = np.fft.rfft(tmp_data, axis=0)
            if self.noise_type == "normal":
                freq_domain += self.noise_amp
            elif self.noise_type == "sin":
                data_len = border2s[1] - border1s[0]
                noise_freq = int(self.noise_freq_percentage * (data_len // 2 + 1))
                freq_domain[-noise_freq:] += self.noise_amp
            else:
                raise NotImplementedError(f"Unsupported noise type: {self.noise_type}")
            noise_data = np.fft.irfft(freq_domain, axis=0).real
            df_data.iloc[border1s[0]:border2s[1]] = noise_data

        if self.scale:
            self.scaler.fit(df_data.iloc[border1s[0]:border2s[0]].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]].iloc[border1:border2].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp["date"])
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp["date"].dt.month
            df_stamp["day"] = df_stamp["date"].dt.day
            df_stamp["weekday"] = df_stamp["date"].dt.weekday
            df_stamp["hour"] = df_stamp["date"].dt.hour
            data_stamp = df_stamp.drop(columns=["date"]).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq).transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len
        return (
            self.data_x[s_begin:s_end],
            self.data_y[r_begin:r_end],
            self.data_stamp[s_begin:s_end],
            self.data_stamp[r_begin:r_end],
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
