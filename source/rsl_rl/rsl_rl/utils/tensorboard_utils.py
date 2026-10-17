# Copyright (c) 2026 Tianyang Wu
#
# SPDX-License-Identifier: MIT

"""TensorBoard writer extensions shared by all logging backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class TensorboardSummaryWriter(SummaryWriter):
    """Summary writer with support for metrics that use a custom step field."""

    def add_custom_scalars(
        self,
        tag_scalar_dict: Mapping[str, Any],
        *,
        step_metric: str,
    ) -> None:
        """Log several scalars against a step stored in ``tag_scalar_dict``.

        TensorBoard stores the custom step as each scalar's ``global_step``. External
        logger subclasses can override this method to preserve the step field as a
        separate metric while still writing the local TensorBoard event file.
        """
        if step_metric not in tag_scalar_dict:
            raise KeyError(f"Custom scalar data does not contain step metric {step_metric!r}.")

        global_step = int(tag_scalar_dict[step_metric])
        for tag, scalar_value in tag_scalar_dict.items():
            if tag == step_metric:
                continue
            SummaryWriter.add_scalar(self, tag, scalar_value, global_step=global_step)


__all__ = ["TensorboardSummaryWriter"]
