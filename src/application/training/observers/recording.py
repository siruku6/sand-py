"""
学習の経過を、ログと CSV に書き残す観測者を定義するモジュール

何を記録するかはここで決め、実際の書き出しは infrastructure/outputs に任せます。
"""

from application.evaluation.metrics import score
from application.training.observers.protocol import EpochReport, RunReport
from custom_utils.logger import init_logger
from infrastructure.outputs import table
from infrastructure.outputs.workspace import RunWorkspace

logger = init_logger()


class ProgressLog:
    """エポックごとの損失をログに残す観測者。"""

    def __init__(self, total_epochs: int) -> None:
        self._total_epochs = total_epochs

    def epoch_completed(self, report: EpochReport) -> None:
        for phase_name, outcome in report.outcomes.items():
            logger.info(
                f"Epoch [{report.epoch}/{self._total_epochs}] "
                f"{phase_name} loss: {outcome.mean_loss:.4f}"
            )

    def run_completed(self, report: RunReport) -> None:
        logger.info(f"Finished after {report.epochs_completed} epochs.")


class LossCurve:
    """
    エポックごとの損失を CSV に書き出す観測者。

    学習が収束しているか、過学習していないかを後から確認するための記録です。
    """

    FILE_NAME = "loss.csv"

    def __init__(self, workspace: RunWorkspace) -> None:
        self._workspace = workspace
        self._rows: list[dict] = []

    def epoch_completed(self, report: EpochReport) -> None:
        self._rows.append(
            {
                "epoch": report.epoch,
                **{
                    f"{phase_name}_loss": outcome.mean_loss
                    for phase_name, outcome in report.outcomes.items()
                },
            }
        )

    def run_completed(self, report: RunReport) -> None:
        """
        Generates:
          - <workspace>/loss.csv
        """
        # loss は後から差分を比較したいので、丸めずにそのまま残す
        table.save_rows(self._rows, self._workspace.file(self.FILE_NAME))


class ScoreSheet:
    """
    エポック・フェーズ・タスクごとの評価指標を CSV に書き出す観測者。

    1 行 = 1 エポックの 1 タスク分という縦持ち形式にすることで、
    後から任意の軸で集計・比較できるようにしています。
    """

    FILE_NAME = "metrics.csv"
    DECIMALS = 3

    def __init__(self, workspace: RunWorkspace) -> None:
        self._workspace = workspace
        self._rows: list[dict] = []

    def epoch_completed(self, report: EpochReport) -> None:
        for phase_name, outcome in report.outcomes.items():
            for task_name, truth in outcome.truth.items():
                self._rows.append(
                    {
                        "epoch": report.epoch,
                        "phase": phase_name,
                        "task": task_name,
                        "loss": outcome.mean_loss,
                        **score(truth, outcome.guess[task_name]).as_dict(),
                        "num_samples": len(truth),
                    }
                )

    def run_completed(self, report: RunReport) -> None:
        """
        Generates:
          - <workspace>/metrics.csv
        """
        table.save_rows(
            self._rows, self._workspace.file(self.FILE_NAME), decimals=self.DECIMALS
        )
