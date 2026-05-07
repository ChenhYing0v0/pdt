from __future__ import annotations

from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute


DATASETS = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
}


def data_provider(args, flag: str):
    if args.task_name != "long_term_forecast":
        raise ValueError("Cleaned PDT baseline only supports long_term_forecast.")

    dataset_cls = DATASETS[args.data]
    timeenc = 1 if args.embed == "timeF" else 0
    shuffle = flag != "test"
    batch_size = 1 if flag == "test" else args.batch_size

    dataset = dataset_cls(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        add_noise=args.add_noise,
        noise_amp=args.noise_amp,
        noise_freq_percentage=args.noise_freq_percentage,
        noise_seed=args.noise_seed,
        noise_type=args.noise_type,
        data_percentage=args.data_percentage,
    )
    print(flag, len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=True,
    )
    return dataset, dataloader
