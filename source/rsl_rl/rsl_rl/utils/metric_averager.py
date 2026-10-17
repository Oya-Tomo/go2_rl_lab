"""Utilities for averaging training metrics over multiple logging steps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch


class MetricAverager:
    """Accumulate numeric metrics using a per-key mean or maximum reduction."""

    def __init__(self) -> None:
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}
        self._reductions: dict[str, Literal["mean", "max"]] = {}
        self._num_updates = 0

    @property
    def num_updates(self) -> int:
        """Number of metric batches accumulated in the current window."""
        return self._num_updates

    def update(
        self,
        metrics: Mapping[str, float | int | torch.Tensor],
        reductions: Mapping[str, Literal["mean", "max"]] | None = None,
    ) -> None:
        """Add one batch of metrics to the current averaging window."""
        self._num_updates += 1
        for key, value in metrics.items():
            tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
            reduction = reductions.get(key, "mean") if reductions is not None else "mean"
            if reduction not in ("mean", "max"):
                raise ValueError(f"Unsupported reduction {reduction!r} for metric '{key}'.")
            if key in self._sums:
                if self._sums[key].shape != tensor.shape:
                    raise ValueError(
                        f"Metric '{key}' changed shape from {tuple(self._sums[key].shape)} to {tuple(tensor.shape)}."
                    )
                if self._reductions[key] != reduction:
                    raise ValueError(
                        f"Metric '{key}' changed reduction from {self._reductions[key]!r} to {reduction!r}."
                    )
                if reduction == "max":
                    self._sums[key] = torch.maximum(self._sums[key], tensor)
                else:
                    self._sums[key].add_(tensor)
                self._counts[key] += 1
            else:
                self._sums[key] = tensor.clone()
                self._counts[key] = 1
                self._reductions[key] = reduction

    def mean_and_reset(self) -> dict[str, torch.Tensor]:
        """Return the reduced metrics and clear the accumulation window."""
        if self._num_updates == 0:
            return {}
        means = {
            key: value if self._reductions[key] == "max" else value / self._counts[key]
            for key, value in self._sums.items()
        }
        self.reset()
        return means

    def reset(self) -> None:
        """Clear all accumulated metrics."""
        self._sums.clear()
        self._counts.clear()
        self._reductions.clear()
        self._num_updates = 0
