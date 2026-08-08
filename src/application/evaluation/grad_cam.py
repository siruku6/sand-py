"""
学習済みネットワークが画像のどこを見て判断したかを可視化するモジュール

Grad-CAM は「あるクラスのスコアが、特徴マップのどの位置の活性に強く依存するか」を
勾配から求める手法。評価時にモデルの根拠を確認するために使う。
"""

from contextlib import contextmanager
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

from application.models.multi_task_classifier import SharedTrunkClassifier


class _SingleTaskView(nn.Module):
    """
    マルチタスクのネットワークを、1 タスク分の分類器に見せかけるアダプタ。

    Grad-CAM は「モデルの出力はロジットのテンソル」という前提で作られているため、
    タスク名をキーにした辞書を返す本体をそのまま渡せません。
    """

    def __init__(self, network: SharedTrunkClassifier, task_name: str) -> None:
        super().__init__()
        self._network = network
        self._task_name = task_name

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self._network(frames)[self._task_name]


class GradCamExplainer:
    """
    ネットワークの判断根拠をヒートマップとして取り出すクラス。

    着目する層はネットワーク自身に答えさせるため、モデル構造を変えても
    このクラスに手を入れる必要はありません。
    """

    def __init__(self, network: SharedTrunkClassifier, chunk_size: int = 2) -> None:
        """
        Parameters
        ----------
        network : SharedTrunkClassifier
            説明の対象とする学習済みネットワーク。
        chunk_size : int
            一度に処理する画像枚数。Grad-CAM は逆伝播を伴うため学習時と同程度の
            メモリを使います。枚数をまとめて渡された場合もここで小分けにします。
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be 1 or greater: {chunk_size}")

        self._network = network
        self._chunk_size = chunk_size

        # 着目する層は入力サイズによって変わりうるため、最初の呼び出し時に実物で決める
        self._target_layer: Optional[nn.Module] = None

    def heatmaps(
        self, frames: torch.Tensor, task_name: str, class_indices: Sequence[int]
    ) -> np.ndarray:
        """
        指定タスクの指定クラスに対するヒートマップを計算します。

        Parameters
        ----------
        frames : torch.Tensor
            形状 (B, 3, H, W) の正規化済み画像バッチ。
        task_name : str
            根拠を見たいタスク名。
        class_indices : Sequence[int]
            画像ごとに、どのクラスのスコアに対する根拠を見るか。

        Returns
        -------
        np.ndarray
            形状 (B, H, W)、値域 0.0-1.0 のヒートマップ。
        """
        view = _SingleTaskView(self._network, task_name)
        target_layer = self._resolve_target_layer(frames)

        with _gradients_enabled(self._network):
            with GradCAM(model=view, target_layers=[target_layer]) as explainer:
                pieces = [
                    explainer(
                        input_tensor=frames[start : start + self._chunk_size],
                        targets=[
                            ClassifierOutputTarget(int(index))
                            for index in class_indices[start : start + self._chunk_size]
                        ],
                    )
                    for start in range(0, len(frames), self._chunk_size)
                ]

        return np.concatenate(pieces)

    def _resolve_target_layer(self, frames: torch.Tensor) -> nn.Module:
        """着目する層を、実際の入力サイズで一度だけ決めて覚えておきます。"""
        if self._target_layer is None:
            self._target_layer = self._network.attribution_layer(frames[:1])
        return self._target_layer


def overlay(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    元画像にヒートマップを重ねた画像を作ります。

    Parameters
    ----------
    image : np.ndarray
        形状 (H, W, 3)、値域 0.0-1.0 の元画像。
    heatmap : np.ndarray
        形状 (H, W)、値域 0.0-1.0 のヒートマップ。

    Returns
    -------
    np.ndarray
        形状 (H, W, 3)、値域 0-255 の重ね合わせ画像。
    """
    return show_cam_on_image(image, heatmap, use_rgb=True)


@contextmanager
def _gradients_enabled(network: nn.Module) -> Iterator[None]:
    """
    一時的に全パラメータの勾配計算を有効にします。

    特徴抽出部を固定して学習した場合、そのままでは活性が計算グラフに乗らず
    Grad-CAM が勾配を取れません。可視化のあいだだけ元に戻し、終了後に復元します。
    """
    original = [parameter.requires_grad for parameter in network.parameters()]
    for parameter in network.parameters():
        parameter.requires_grad_(True)
    try:
        yield
    finally:
        for parameter, was_enabled in zip(network.parameters(), original):
            parameter.requires_grad_(was_enabled)
