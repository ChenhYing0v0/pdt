from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args

try:
    import setproctitle
except ImportError:  # pragma: no cover - optional dependency
    setproctitle = None


BASELINE_ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_value: str | None) -> str | None:
    if not path_value:
        return path_value
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((BASELINE_ROOT / path).resolve())


def prepare_output_paths(args: argparse.Namespace) -> argparse.Namespace:
    output_dir = args.output_dir or str(BASELINE_ROOT / "exp_results" / (args.run_id or args.model_id))
    args.output_dir = str(Path(output_dir).resolve())
    args.checkpoints = str((Path(args.output_dir) / "checkpoints").resolve())
    args.results = str((Path(args.output_dir) / "results").resolve())
    args.test_results = str((Path(args.output_dir) / "test_results").resolve())
    args.log_path = str((Path(args.output_dir) / "result_long_term_forecast.txt").resolve())
    Path(args.checkpoints).mkdir(parents=True, exist_ok=True)
    Path(args.results).mkdir(parents=True, exist_ok=True)
    Path(args.test_results).mkdir(parents=True, exist_ok=True)
    return args


def resolve_runtime_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.root_path = resolve_path(args.root_path)
    for field in [
        "q_mat_file",
        "Q_MAT_file",
        "q_out_mat_file",
        "Q_OUT_MAT_file",
        "r_mat_file",
        "R_MAT_file",
        "rk_mat_file",
        "Rk_MAT_file",
    ]:
        setattr(args, field, resolve_path(getattr(args, field)))
    return args


def build_setting(args: argparse.Namespace, run_index: int) -> str:
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}"
        f"_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}"
        f"_el{args.e_layers}_df{args.d_ff}_rr{args.r_rank}_k{args.k_top}_{args.des}_{run_index}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDT long-term forecasting entry")

    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, required=True, default=1)
    parser.add_argument("--model_id", type=str, required=True, default="test")
    parser.add_argument("--model", type=str, required=True, default="PDT")

    parser.add_argument("--data", type=str, required=True, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="dataset/ETT-small")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--add_noise", action="store_true", default=False)
    parser.add_argument("--noise_amp", type=float, default=0.0)
    parser.add_argument("--noise_freq_percentage", type=float, default=0.05)
    parser.add_argument("--noise_seed", type=int, default=2023)
    parser.add_argument("--noise_type", type=str, default="sin")
    parser.add_argument("--data_percentage", type=float, default=1.0)

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--inverse", action="store_true", default=False)

    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--distil", action="store_false", default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", action="store_true", default=False)
    parser.add_argument("--channel_independence", type=int, default=0)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--des", type=str, default="protocol")
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--use_amp", action="store_true", default=False)

    parser.add_argument("--rec_lambda", type=float, default=1.0)
    parser.add_argument("--auxi_lambda", type=float, default=0.0)
    parser.add_argument("--auxi_loss", type=str, default="MAE")
    parser.add_argument("--auxi_mode", type=str, default="fft")
    parser.add_argument("--auxi_type", type=str, default="complex")
    parser.add_argument("--module_first", type=int, default=1)
    parser.add_argument("--leg_degree", type=int, default=2)

    parser.add_argument("--q_mat_file", type=str, default=None)
    parser.add_argument("--Q_MAT_file", type=str, default=None)
    parser.add_argument("--q_out_mat_file", type=str, default=None)
    parser.add_argument("--Q_OUT_MAT_file", type=str, default=None)
    parser.add_argument("--Q_chan_indep", type=int, default=0)
    parser.add_argument("--CKA_flag", type=int, default=0)
    parser.add_argument("--embed_size", type=int, default=16)
    parser.add_argument("--loss_mode", type=str, default="L1", choices=["L1", "L2"])

    parser.add_argument("--r_mat_file", type=str, default=None)
    parser.add_argument("--R_MAT_file", type=str, default=None)
    parser.add_argument("--rk_mat_file", type=str, default=None)
    parser.add_argument("--Rk_MAT_file", type=str, default=None)
    parser.add_argument("--freeze_R", type=int, default=0)
    parser.add_argument("--r_rank", type=int, default=96)
    parser.add_argument("--k_top", type=int, default=16)
    parser.add_argument("--alpha_init", type=float, default=0.1)
    parser.add_argument("--mask_threshold", type=float, default=0.25)
    parser.add_argument("--mask_sharpness_k", type=float, default=100.0)
    parser.add_argument("--temp_stride", type=int, default=8)
    parser.add_argument("--temp_patch_len", type=int, default=16)

    parser.add_argument("--output_pred", action="store_true", default=False)
    parser.add_argument("--output_vis", action="store_true", default=False)
    parser.add_argument("--save_cov", type=int, default=0)

    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--metric_policy", type=str, default="forecasting_v1")
    parser.add_argument("--selection_policy", type=str, default="best_val_mse")
    parser.add_argument("--skip_predictions", action="store_true", default=False)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.task_name != "long_term_forecast":
        raise ValueError("PDT baseline in this repo only supports long_term_forecast.")
    if args.model != "PDT":
        raise ValueError("This cleaned baseline only exposes the PDT model.")

    args = prepare_output_paths(args)
    args = resolve_runtime_paths(args)
    set_seed(args.seed)

    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(device_id) for device_id in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print_args(args)

    if setproctitle is not None:
        setproctitle.setproctitle(args.task_name)

    if args.is_training:
        for run_index in range(args.itr):
            exp = Exp_Long_Term_Forecast(args)
            setting = build_setting(args, run_index)
            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)
            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        exp = Exp_Long_Term_Forecast(args)
        setting = build_setting(args, 0)
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

    print("实际生效的 CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
