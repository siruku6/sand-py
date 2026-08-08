"""
学習フェーズと検証フェーズを繰り返す進行役を定義するモジュール

ここでは CSV も画像もモデルも書き出さない。起きた出来事を観測者に伝えるだけで、
何を残すかは観測者側が決める。出力を増やしたいときは観測者を足すだけで済む。
"""

from dataclasses import dataclass
from typing import Optional, Sequence

from torch import nn
from tqdm.auto import tqdm

from application.training.observers.protocol import (
    EpochReport,
    RunObserver,
    RunReport,
)
from application.training.outcome import PhaseOutcome
from application.training.phases import Phase
from application.training.stopping import StopPolicy, Verdict


@dataclass(frozen=True)
class LoopPlan:
    """学習ループの進め方。"""

    max_epochs: int

    # 打ち切り判定に使うフェーズ名。ここでは検証フェーズを想定する
    monitored_phase: str = "val"


class TrainingLoop:
    """
    フェーズの繰り返しと打ち切り判定だけを担う進行役。

    フェーズ・打ち切り方針・観測者はすべて外から渡します。
    このクラス自身は具体的な学習手法も出力形式も知りません。
    """

    def __init__(
        self,
        phases: Sequence[Phase],
        stop_policy: StopPolicy,
        observers: Sequence[RunObserver],
    ) -> None:
        """
        Parameters
        ----------
        phases : Sequence[Phase]
            1 エポックで順に実行するフェーズ。通常は学習・検証の 2 つ。
        stop_policy : StopPolicy
            打ち切りとベスト更新を判定する方針。
        observers : Sequence[RunObserver]
            エポック終了時と実行終了時に通知を受け取る観測者。
        """
        self._phases = tuple(phases)
        self._stop_policy = stop_policy
        self._observers = tuple(observers)

    def execute(self, network: nn.Module, plan: LoopPlan) -> RunReport:
        """
        計画に従って学習を進めます。

        Parameters
        ----------
        network : nn.Module
            学習対象のネットワーク。パラメータはその場で更新されます。
        plan : LoopPlan
            エポック数と監視対象フェーズ。

        Returns
        -------
        RunReport
            実行全体の要約。
        """
        latest: Optional[EpochReport] = None

        for epoch in tqdm(range(1, plan.max_epochs + 1), desc="Epochs"):
            outcomes: dict[str, PhaseOutcome] = {
                phase.name: phase.sweep(network) for phase in self._phases
            }
            verdict: Verdict = self._stop_policy.judge(
                outcomes[plan.monitored_phase].mean_loss
            )

            latest = EpochReport(
                epoch=epoch, network=network, outcomes=outcomes, verdict=verdict
            )
            self._announce_epoch(latest)

            if verdict.exhausted:
                break

        if latest is None:
            raise ValueError(f"max_epochs must be 1 or greater: {plan.max_epochs}")

        report = RunReport(
            epochs_completed=latest.epoch,
            network=network,
            final_outcomes=latest.outcomes,
        )
        self._announce_run(report)
        return report

    def _announce_epoch(self, report: EpochReport) -> None:
        for observer in self._observers:
            observer.epoch_completed(report)

    def _announce_run(self, report: RunReport) -> None:
        for observer in self._observers:
            observer.run_completed(report)
