"""
CelebA でマルチタスク分類モデルを学習するエントリポイント

学習環境 (GPU・依存パッケージ・データ配置) が正しく構築できているかを
確認するための動作確認用スクリプトです。手順は docs/tutorial/CelebA.md を参照してください。

このファイルは組み立て役に徹します。どの部品を使うかをここだけで決め、
部品どうしは互いの実体を知らないまま組み合わせられるようにしています。
"""

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from application.models import multi_task_classifier
from application.models.multi_task_classifier import SharedTrunkClassifier
from application.preprocessing.frames import FrameNormalisation
from application.training.loop import LoopPlan, TrainingLoop
from application.training.objective import WeightedTaskLoss
from application.training.observers.artifacts import (
    AttributionGallery,
    Checkpoints,
    ConfusionMatrices,
)
from application.training.observers.protocol import RunObserver
from application.training.observers.recording import LossCurve, ProgressLog, ScoreSheet
from application.training.phases import (
    EvaluationPhase,
    KeepTrunkFrozen,
    OptimisationPhase,
    TrainWholeNetwork,
    TrunkPolicy,
)
from application.training.settings import TrainingSettings
from application.training.stopping import policy_for
from custom_utils.logger import init_logger
from custom_utils.paths import CONFIG_DIR, DATA_PATH, OUTPUT_PATH
from custom_utils.seed import set_seed
from domain.celeba import labels
from domain.tasks import TaskSuite
from infrastructure import config_loader
from infrastructure.construct_dataset.celeba_catalog import CelebaArchive, Split
from infrastructure.construct_dataset.celeba_dataset import build_loader
from infrastructure.outputs.workspace import RunWorkspace

logger = init_logger()

DEFAULT_SETTINGS_PATH: Path = CONFIG_DIR / "celeba.yml"
CELEBA_DIR: Path = DATA_PATH / "celeba"
CELEBA_OUTPUT_DIR: Path = OUTPUT_PATH / "celeba"

# 打ち切り判定と、混同行列・Grad-CAM の元にするフェーズ
MONITORED_PHASE = "val"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_SETTINGS_PATH),
        help=f"設定ファイルのパス (既定: {DEFAULT_SETTINGS_PATH})",
    )
    return parser.parse_args()


def _pick_device() -> str:
    """利用可能なら GPU を、なければ CPU を選びます。"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _open_loaders(
    settings: TrainingSettings, rules: tuple, normalisation: FrameNormalisation
) -> tuple[DataLoader, DataLoader]:
    """学習用と検証用の DataLoader を用意します。"""
    archive = CelebaArchive(CELEBA_DIR)
    to_tensor = normalisation.encoder()

    train_loader = build_loader(
        archive=archive,
        split=Split.TRAIN,
        rules=rules,
        to_tensor=to_tensor,
        batch_size=settings.batch_size,
        limit=settings.num_train_samples,
        shuffle=True,
    )
    val_loader = build_loader(
        archive=archive,
        split=Split.VALIDATION,
        rules=rules,
        to_tensor=to_tensor,
        batch_size=settings.batch_size,
        limit=settings.num_val_samples,
        shuffle=False,
    )

    print("Train size:", len(train_loader.dataset))
    print("Val size:", len(val_loader.dataset))
    return train_loader, val_loader


def _prepare_trunk(
    classifier: SharedTrunkClassifier, settings: TrainingSettings
) -> tuple[TrunkPolicy, optim.Optimizer]:
    """
    特徴抽出部を固定するかどうかに応じて、学習モード方針と optimizer を決めます。

    固定する場合は学習済み特徴量をそのまま使い、タスクごとの出力層だけを学習させます。
    """
    if settings.freeze_backbone:
        print("Freezing backbone layers...")
        classifier.freeze_trunk()
        return KeepTrunkFrozen(), optim.Adam(
            classifier.output_parameters(), lr=settings.learning_rate
        )

    print("Training all layers...")
    return TrainWholeNetwork(), optim.Adam(
        classifier.parameters(), lr=settings.learning_rate
    )


def _assemble_observers(
    workspace: RunWorkspace,
    suite: TaskSuite,
    settings: TrainingSettings,
    val_loader: DataLoader,
    normalisation: FrameNormalisation,
    device: str,
) -> list[RunObserver]:
    """
    残したい出力の分だけ観測者を並べます。

    出力を増やしたいときは、ここにひとつ足すだけで済みます。
    """
    return [
        ProgressLog(total_epochs=settings.num_epochs),
        LossCurve(workspace),
        ScoreSheet(workspace),
        Checkpoints(workspace),
        ConfusionMatrices(workspace, suite, phase=MONITORED_PHASE),
        AttributionGallery(
            workspace=workspace,
            loader=val_loader,
            normalisation=normalisation,
            suite=suite,
            device=device,
            num_samples=settings.grad_cam_samples,
        ),
    ]


def run(settings: TrainingSettings) -> nn.Module:
    """
    設定に従って学習を実行し、学習済みネットワークを返します。

    Parameters
    ----------
    settings : TrainingSettings
        学習の実行条件。

    Returns
    -------
    nn.Module
        学習済みネットワーク。

    Generates:
      - outputs/celeba/<日時>_<experiment_name>/ 配下の学習結果一式
    """
    set_seed(settings.seed)

    device: str = _pick_device()
    logger.info(f"Using device: {device}")
    print(f"Using device: {device}")

    rules = labels.default_rules()
    suite = labels.suite_of(rules)
    normalisation = FrameNormalisation()

    print("Constructing dataset...")
    train_loader, val_loader = _open_loaders(settings, rules, normalisation)

    classifier = multi_task_classifier.assemble(settings.model_name, suite).to(device)
    trunk_policy, optimiser = _prepare_trunk(classifier, settings)

    objective = WeightedTaskLoss()
    workspace = RunWorkspace.create_for(CELEBA_OUTPUT_DIR, settings.experiment_name)

    loop = TrainingLoop(
        phases=[
            OptimisationPhase(
                loader=train_loader,
                objective=objective,
                optimiser=optimiser,
                device=device,
                trunk_policy=trunk_policy,
            ),
            EvaluationPhase(loader=val_loader, objective=objective, device=device),
        ],
        stop_policy=policy_for(settings.patience),
        observers=_assemble_observers(
            workspace, suite, settings, val_loader, normalisation, device
        ),
    )

    print("Training started")
    loop.execute(
        classifier,
        LoopPlan(max_epochs=settings.num_epochs, monitored_phase=MONITORED_PHASE),
    )

    print("Training completed.")
    print(f"Results saved to: {workspace}")
    return classifier


if __name__ == "__main__":
    run(config_loader.load(Path(_parse_args().config), TrainingSettings))
