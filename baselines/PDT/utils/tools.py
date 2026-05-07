from __future__ import annotations

import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend("agg")


def ensure_path(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def adjust_learning_rate(optimizer, epoch, args, scheduler=None, printout: bool = True) -> None:
    lr_adjust = {}
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** (epoch - 3))}
    elif args.lradj == "constant":
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == "TST":
        if scheduler is None:
            raise ValueError("scheduler is required when lradj=TST")
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj in {"cosine", "card"}:
        min_lr = 0.0
        warmup_epochs = 0
        lr = min_lr + (args.learning_rate - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (args.train_epochs - warmup_epochs))
        )
        lr_adjust = {epoch: lr}

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if printout:
            print(f"Updating learning rate to {lr}")


class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0, save_every_epoch: bool = False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_every_epoch = save_every_epoch

    def __call__(self, val_loss, model, path, epoch=None) -> None:
        if np.isnan(val_loss):
            self.early_stop = True
            return

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return

        self.best_score = score
        self.save_checkpoint(val_loss, model, path, epoch)
        self.counter = 0
        self.early_stop = False

    def save_checkpoint(self, val_loss, model, path, epoch=None) -> None:
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")

        checkpoint_path = os.path.join(path, "checkpoint.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"The size of checkpoint is {convert_size(os.path.getsize(checkpoint_path))}.")

        delete_txt_files_in_folder(path)
        with open(os.path.join(path, f"Epoch_{epoch}.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"Current Epoch: {epoch}")

        if self.save_every_epoch:
            if epoch is None:
                suffix = f"checkpoint_val_loss_{val_loss:.5f}.pth"
            else:
                suffix = f"checkpoint_epoch_{epoch:d}_val_loss_{val_loss:.5f}.pth"
            shutil.copy(checkpoint_path, os.path.join(path, suffix))

        self.val_loss_min = val_loss


def convert_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f}PB"


def delete_txt_files_in_folder(path: str) -> None:
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            os.remove(os.path.join(path, filename))


def visual(true, preds=None, name: str = "./pic/test.pdf", imp: bool = False) -> None:
    folder_name = os.path.dirname(name)
    if folder_name:
        os.makedirs(folder_name, exist_ok=True)

    label = "Imputation" if imp else "Prediction"
    if not isinstance(true, np.ndarray):
        true = true.numpy()
    if preds is not None and not isinstance(preds, np.ndarray):
        preds = preds.numpy()

    plt.figure()
    plt.plot(true, label="Ground Truth", linestyle="--", linewidth=2)
    if preds is not None:
        plt.plot(preds, label=label, linewidth=2)
    plt.legend()
    plt.grid(linestyle=":", color="lightgray")
    plt.savefig(name, bbox_inches="tight")
    plt.close()


def forward_fill(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    filled = x.clone()
    running = filled[:, :1, :]
    for index in range(filled.shape[1]):
        current_mask = mask[:, index:index + 1, :]
        running = torch.where(current_mask, running, filled[:, index:index + 1, :])
        filled[:, index:index + 1, :] = torch.where(current_mask, running, filled[:, index:index + 1, :])
    return filled, mask
