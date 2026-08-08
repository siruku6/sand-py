"""
データを 1 巡させる処理を、学習用と検証用に分けて定義するモジュール

両者の違いは「勾配を流してパラメータを更新するか」だけなので、
学習ループからは同じ Phase として扱えるようにしている。
"""

from typing import Callable, Mapping, Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from application.training.outcome import OutcomeCollector, PhaseOutcome

# ロジットと正解ラベルからスカラーの損失を返すもの
Objective = Callable[
    [Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]], torch.Tensor
]


class Phase(Protocol):
    """データを 1 巡させ、その結果を返す処理。"""

    @property
    def name(self) -> str:
        """記録に使うフェーズ名。"""
        ...

    def sweep(self, network: nn.Module) -> PhaseOutcome:
        """データ全体を 1 巡し、結果を返します。"""
        ...


class TrunkPolicy(Protocol):
    """学習時にネットワークをどのモードへ置くかを決める方針。"""

    def engage(self, network: nn.Module) -> None:
        """ネットワークを学習可能な状態にします。"""
        ...


class TrainWholeNetwork:
    """ネットワーク全体を学習モードにする方針。"""

    def engage(self, network: nn.Module) -> None:
        network.train()


class KeepTrunkFrozen:
    """
    出力層だけを学習モードにし、特徴抽出部は評価モードに保つ方針。

    特徴抽出部を固定して学習する場合、パラメータを更新対象から外すだけでは
    dropout と batch normalization の統計量が動いてしまうため、
    こちらも合わせて止めます。
    """

    def engage(self, network: nn.Module) -> None:
        network.train()
        network.trunk.eval()


class OptimisationPhase:
    """勾配を流してパラメータを更新する巡回。"""

    def __init__(
        self,
        loader: DataLoader,
        objective: Objective,
        optimiser: torch.optim.Optimizer,
        device: str,
        trunk_policy: TrunkPolicy,
    ) -> None:
        """
        Parameters
        ----------
        loader : DataLoader
            学習データの DataLoader。
        objective : Callable
            ロジットと正解ラベルから損失を計算するもの。
        optimiser : torch.optim.Optimizer
            パラメータ更新に使う optimizer。
        device : str
            "cuda" または "cpu"。
        trunk_policy : TrunkPolicy
            ネットワークを学習可能な状態にする方針。
        """
        self._loader = loader
        self._objective = objective
        self._optimiser = optimiser
        self._device = device
        self._trunk_policy = trunk_policy

    @property
    def name(self) -> str:
        return "train"

    def sweep(self, network: nn.Module) -> PhaseOutcome:
        """学習データを 1 巡し、パラメータを更新しながら結果を集めます。"""
        self._trunk_policy.engage(network)

        collector = OutcomeCollector()
        for frames, targets in tqdm(self._loader, desc="training"):
            frames, targets = _to_device(frames, targets, self._device)

            predictions = network(frames)
            loss = self._objective(predictions, targets)

            self._optimiser.zero_grad()
            loss.backward()
            self._optimiser.step()

            collector.observe(loss.item(), predictions, targets)

        return collector.seal()


class EvaluationPhase:
    """パラメータを更新せずに性能だけを測る巡回。"""

    def __init__(self, loader: DataLoader, objective: Objective, device: str) -> None:
        """
        Parameters
        ----------
        loader : DataLoader
            検証データの DataLoader。
        objective : Callable
            ロジットと正解ラベルから損失を計算するもの。
        device : str
            "cuda" または "cpu"。
        """
        self._loader = loader
        self._objective = objective
        self._device = device

    @property
    def name(self) -> str:
        return "val"

    def sweep(self, network: nn.Module) -> PhaseOutcome:
        """検証データを 1 巡し、結果を集めます。"""
        network.eval()

        collector = OutcomeCollector()
        with torch.no_grad():
            for frames, targets in tqdm(self._loader, desc="validating"):
                frames, targets = _to_device(frames, targets, self._device)

                predictions = network(frames)
                loss = self._objective(predictions, targets)

                collector.observe(loss.item(), predictions, targets)

        return collector.seal()


def _to_device(
    frames: torch.Tensor, targets: Mapping[str, torch.Tensor], device: str
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """1 バッチ分の入力と正解ラベルを、まとめて計算デバイスへ移します。"""
    return (
        frames.to(device),
        {name: labels.to(device) for name, labels in targets.items()},
    )
