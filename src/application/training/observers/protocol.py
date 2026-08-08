"""
学習ループと観測者のあいだで受け渡す報告と、観測者の契約を定めるモジュール
"""

from dataclasses import dataclass
from typing import Mapping, Protocol

from torch import nn

from application.training.outcome import PhaseOutcome
from application.training.stopping import Verdict


@dataclass(frozen=True)
class EpochReport:
    """1 エポックが終わった時点の状況。"""

    epoch: int

    # 学習中のネットワーク。この時点の重みを保存したい観測者が使う
    network: nn.Module

    # フェーズ名 -> そのフェーズの結果
    outcomes: Mapping[str, PhaseOutcome]

    verdict: Verdict


@dataclass(frozen=True)
class RunReport:
    """学習が終わった時点の要約。"""

    # 実際に実行したエポック数。打ち切られた場合は計画より少なくなる
    epochs_completed: int

    network: nn.Module

    # 最終エポックのフェーズ別結果
    final_outcomes: Mapping[str, PhaseOutcome]


class RunObserver(Protocol):
    """学習ループの進行を受け取り、記録や出力を行うもの。"""

    def epoch_completed(self, report: EpochReport) -> None:
        """1 エポックが終わるたびに呼ばれます。"""
        ...

    def run_completed(self, report: RunReport) -> None:
        """学習全体が終わったときに呼ばれます。"""
        ...
