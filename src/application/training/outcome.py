"""
1 巡分の実行結果を集約するモジュール

学習と検証は「バッチごとの損失と予測を積み上げ、最後にまとめる」という同じ形を
しているため、その積み上げ方をここに一本化する。
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class PhaseOutcome:
    """データ全体を 1 巡した結果。"""

    # サンプルあたりの平均損失
    mean_loss: float

    # タスク名 -> 正解クラス番号の配列 (N,)
    truth: Mapping[str, np.ndarray]

    # タスク名 -> 予測クラス番号の配列 (N,)
    guess: Mapping[str, np.ndarray]

    @property
    def num_samples(self) -> int:
        """評価に使ったサンプル数。"""
        return len(next(iter(self.truth.values())))


@dataclass
class OutcomeCollector:
    """バッチ単位の結果を受け取り、1 巡分の PhaseOutcome にまとめるクラス。"""

    _loss_sum: float = 0.0
    _seen: int = 0
    _truth: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _guess: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def observe(
        self,
        loss: float,
        predictions: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> None:
        """
        1 バッチ分の結果を積み上げます。

        Parameters
        ----------
        loss : float
            そのバッチの平均損失。
        predictions : Mapping[str, torch.Tensor]
            タスク名をキー、形状 (B, num_classes) のロジットを値とする対応。
        targets : Mapping[str, torch.Tensor]
            タスク名をキー、形状 (B,) の正解クラス番号を値とする対応。
        """
        batch_size = len(next(iter(targets.values())))
        self._loss_sum += loss * batch_size
        self._seen += batch_size

        for name, labels in targets.items():
            self._truth[name].append(labels.cpu().numpy())
            self._guess[name].append(predictions[name].argmax(dim=1).cpu().numpy())

    def seal(self) -> PhaseOutcome:
        """積み上げた結果を PhaseOutcome にまとめます。"""
        return PhaseOutcome(
            mean_loss=self._loss_sum / self._seen,
            truth={name: np.concatenate(parts) for name, parts in self._truth.items()},
            guess={name: np.concatenate(parts) for name, parts in self._guess.items()},
        )
