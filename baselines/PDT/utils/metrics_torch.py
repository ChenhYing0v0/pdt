from __future__ import annotations

import torch


class MetricCollector:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self.sum_abs_error = torch.tensor(0.0, device=self.device)
        self.sum_squared_error = torch.tensor(0.0, device=self.device)
        self.sum_abs_per_error = torch.tensor(0.0, device=self.device)
        self.sum_sq_per_error = torch.tensor(0.0, device=self.device)
        self.total = torch.tensor(0.0, device=self.device)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.to(self.device)
        target = target.to(self.device)
        denom = torch.clamp(target.abs(), min=1e-6)
        self.sum_abs_error += torch.sum(torch.abs(preds - target))
        self.sum_squared_error += torch.sum((preds - target) ** 2)
        self.sum_abs_per_error += torch.sum(torch.abs((preds - target) / denom))
        self.sum_sq_per_error += torch.sum(((preds - target) / denom) ** 2)
        self.total += target.numel()

    def compute(self) -> dict[str, float]:
        mse = (self.sum_squared_error / self.total).item()
        return {
            "mae": (self.sum_abs_error / self.total).item(),
            "mse": mse,
            "rmse": mse ** 0.5,
            "mape": (self.sum_abs_per_error / self.total).item(),
            "mspe": (self.sum_sq_per_error / self.total).item(),
        }


def create_metric_collector(device: str = "cpu") -> MetricCollector:
    return MetricCollector(device=device)
