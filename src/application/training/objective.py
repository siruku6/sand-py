"""
マルチタスクの損失関数を定義するモジュール
"""

from typing import Mapping, Optional

import torch
from torch import nn


class WeightedTaskLoss:
    """
    タスクごとの交差エントロピーを重み付きで合計する損失。

    タスク間で損失のスケールや重要度が異なる場合に、重みで釣り合いを取ります。
    重みを与えなければ全タスク等価に扱います。
    """

    def __init__(self, weights: Optional[Mapping[str, float]] = None) -> None:
        """
        Parameters
        ----------
        weights : Mapping[str, float] | None
            タスク名をキー、重みを値とする対応。指定のないタスクの重みは 1.0 です。
        """
        self._weights = dict(weights or {})
        self._per_task = nn.CrossEntropyLoss()

    def __call__(
        self,
        predictions: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        全タスクの損失を合計したスカラーを返します。

        Parameters
        ----------
        predictions : Mapping[str, torch.Tensor]
            タスク名をキー、形状 (B, num_classes) のロジットを値とする対応。
        targets : Mapping[str, torch.Tensor]
            タスク名をキー、形状 (B,) の正解クラス番号を値とする対応。

        Returns
        -------
        torch.Tensor
            スカラーの損失値。
        """
        device = next(iter(predictions.values())).device
        total = torch.zeros((), device=device)
        for name, labels in targets.items():
            weight = self._weights.get(name, 1.0)
            total = total + weight * self._per_task(predictions[name], labels)
        return total
