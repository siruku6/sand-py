"""
学習の成果物 (重み・混同行列・判断根拠の可視化) を残す観測者を定義するモジュール

何を残すかはここで決め、実際の書き出しは infrastructure/outputs に任せます。
"""

from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from application.evaluation.grad_cam import GradCamExplainer, overlay
from application.preprocessing.frames import FrameNormalisation
from application.training.observers.protocol import EpochReport, RunReport
from application.training.outcome import PhaseOutcome
from custom_utils.logger import init_logger
from domain.tasks import TaskSuite
from infrastructure.outputs import model as model_output
from infrastructure.outputs.figure import canvas
from infrastructure.outputs.workspace import RunWorkspace

logger = init_logger()


class Checkpoints:
    """
    ベスト更新時と学習終了時に重みを保存する観測者。

    途中で打ち切られた場合でも、もっとも検証成績の良かった時点の重みが残ります。
    """

    BEST_FILE = "model_best.pth"
    LAST_FILE = "model_last.pth"

    def __init__(self, workspace: RunWorkspace) -> None:
        self._workspace = workspace

    def epoch_completed(self, report: EpochReport) -> None:
        """
        Generates:
          - <workspace>/model_best.pth
        """
        if report.verdict.improved:
            model_output.save(report.network, self._workspace.file(self.BEST_FILE))

    def run_completed(self, report: RunReport) -> None:
        """
        Generates:
          - <workspace>/model_last.pth
        """
        model_output.save(report.network, self._workspace.file(self.LAST_FILE))


class ConfusionMatrices:
    """
    最終エポックの検証結果から、タスクごとの混同行列を描く観測者。

    どのクラスをどのクラスと取り違えているかは正解率だけでは分かりません。
    評価指標が低いときに原因を切り分けるための図です。
    """

    FOLDER = "confusion_matrix"

    def __init__(self, workspace: RunWorkspace, suite: TaskSuite, phase: str) -> None:
        """
        Parameters
        ----------
        workspace : RunWorkspace
            出力先。
        suite : TaskSuite
            クラス名を引くためのタスク定義。
        phase : str
            どのフェーズの結果を描くか。通常は検証フェーズ。
        """
        self._workspace = workspace
        self._suite = suite
        self._phase = phase

    def epoch_completed(self, report: EpochReport) -> None:
        pass

    def run_completed(self, report: RunReport) -> None:
        """
        Generates:
          - <workspace>/confusion_matrix/<タスク名>.png
        """
        outcome: PhaseOutcome = report.final_outcomes[self._phase]
        folder = self._workspace.folder(self.FOLDER)

        for task in self._suite:
            matrix = confusion_matrix(
                outcome.truth[task.name],
                outcome.guess[task.name],
                labels=range(task.num_classes),
            )
            self._draw(matrix, task.name, list(task.class_names), folder)

        logger.info(f"Confusion matrices saved to: {folder}")

    def _draw(
        self, matrix: np.ndarray, task_name: str, class_names: list[str], folder: Path
    ) -> None:
        """1 タスク分の混同行列をヒートマップとして保存します。"""
        side = len(class_names)
        with canvas(figsize=(1.4 * side + 2.5, 1.2 * side + 2)) as (figure, axes):
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes,
            )
            axes.set_title(f"Confusion Matrix: {task_name}")
            axes.set_xlabel("Predicted")
            axes.set_ylabel("True")
            figure.tight_layout()
            figure.savefig(folder / f"{task_name}.png", dpi=150)


class AttributionGallery:
    """
    学習終了後に Grad-CAM を適用し、判断根拠を可視化する観測者。

    検証データの先頭から数枚を取り出し、タスクごとに「元画像」と
    「予測クラスに対する根拠を重ねた画像」を並べた図を書き出します。
    """

    FOLDER = "grad_cam"

    def __init__(
        self,
        workspace: RunWorkspace,
        loader: DataLoader,
        normalisation: FrameNormalisation,
        suite: TaskSuite,
        device: str,
        num_samples: int,
    ) -> None:
        """
        Parameters
        ----------
        workspace : RunWorkspace
            出力先。
        loader : DataLoader
            可視化する画像の取得元。通常は検証データ。
        normalisation : FrameNormalisation
            正規化を打ち消して元の見た目に戻すために使います。
        suite : TaskSuite
            クラス名を引くためのタスク定義。
        device : str
            推論を行うデバイス。
        num_samples : int
            可視化する画像の枚数。0 なら何もしません。
        """
        self._workspace = workspace
        self._loader = loader
        self._normalisation = normalisation
        self._suite = suite
        self._device = device
        self._num_samples = num_samples

    def epoch_completed(self, report: EpochReport) -> None:
        pass

    def run_completed(self, report: RunReport) -> None:
        """
        Generates:
          - <workspace>/grad_cam/<タスク名>.png
        """
        if self._num_samples < 1:
            return

        frames, truth = self._collect_samples()
        if len(frames) == 0:
            return

        network = report.network
        network.eval()
        explainer = GradCamExplainer(network)
        pictures = self._normalisation.decode(frames)

        on_device = frames.to(self._device)
        folder = self._workspace.folder(self.FOLDER)
        for task in self._suite:
            with torch.no_grad():
                predicted = network(on_device)[task.name].argmax(dim=1).cpu().numpy()

            heatmaps = explainer.heatmaps(on_device, task.name, predicted.tolist())
            self._draw(
                pictures=pictures,
                heatmaps=heatmaps,
                captions=[
                    f"true: {task.class_names[actual]}\npred: {task.class_names[guess]}"
                    for actual, guess in zip(truth[task.name].tolist(), predicted)
                ],
                task_name=task.name,
                destination=folder / f"{task.name}.png",
            )

        logger.info(f"Grad-CAM visualisations saved to: {folder}")

    def _collect_samples(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """DataLoader の先頭から、必要な枚数だけ画像と正解ラベルを集めます。"""
        frames: list[torch.Tensor] = []
        labels: dict[str, list[torch.Tensor]] = {name: [] for name in self._suite.names}

        collected = 0
        for batch_frames, batch_targets in self._loader:
            frames.append(batch_frames)
            for name, values in batch_targets.items():
                labels[name].append(values)

            collected += len(batch_frames)
            if collected >= self._num_samples:
                break

        if not frames:
            return torch.empty(0), {}

        keep = self._num_samples
        return (
            torch.cat(frames)[:keep],
            {name: torch.cat(parts)[:keep] for name, parts in labels.items()},
        )

    def _draw(
        self,
        pictures: np.ndarray,
        heatmaps: np.ndarray,
        captions: list[str],
        task_name: str,
        destination: Path,
    ) -> None:
        """元画像と重ね合わせ画像を 2 段に並べた図を保存します。"""
        columns = len(pictures)
        with canvas(
            nrows=2, ncols=columns, figsize=(2.2 * columns, 5.4), squeeze=False
        ) as (figure, axes):
            for column in range(columns):
                axes[0][column].imshow(pictures[column])
                axes[0][column].set_title(captions[column], fontsize=8)
                axes[1][column].imshow(overlay(pictures[column], heatmaps[column]))

                for row in (0, 1):
                    axes[row][column].axis("off")

            figure.suptitle(f"Grad-CAM: {task_name}")
            figure.tight_layout()
            figure.savefig(destination, dpi=150)
