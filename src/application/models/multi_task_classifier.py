"""
特徴抽出部を共有し、タスクごとに出力層を分けるネットワークを定義するモジュール
"""

from typing import Iterator

import timm
import torch
from torch import nn

from domain.tasks import TaskSuite


class SharedTrunkClassifier(nn.Module):
    """
    1 つの特徴抽出部 (trunk) を共有し、タスクごとに線形の出力層を持つ分類器。

    trunk は外から受け取ります。timm でも自作でも、特徴ベクトルを返すモジュール
    でありさえすれば差し替えられます。
    """

    def __init__(self, trunk: nn.Module, feature_dim: int, suite: TaskSuite) -> None:
        """
        Parameters
        ----------
        trunk : nn.Module
            画像から特徴ベクトルを取り出すモジュール。分類層を含まないもの。
        feature_dim : int
            trunk が出力する特徴ベクトルの次元数。
        suite : TaskSuite
            解くタスクの並び。この順序で出力層を作ります。
        """
        super().__init__()
        self.trunk = trunk
        self.outputs = nn.ModuleDict(
            {task.name: nn.Linear(feature_dim, task.num_classes) for task in suite}
        )

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        画像バッチからタスクごとのロジットを計算します。

        Parameters
        ----------
        frames : torch.Tensor
            形状 (B, 3, H, W) の画像バッチ。

        Returns
        -------
        dict[str, torch.Tensor]
            タスク名をキー、形状 (B, num_classes) のロジットを値とする辞書。
        """
        features = self.trunk(frames)
        return {name: layer(features) for name, layer in self.outputs.items()}

    def output_parameters(self) -> Iterator[nn.Parameter]:
        """出力層だけのパラメータを返します。trunk を固定して学習する場合に使います。"""
        return self.outputs.parameters()

    def freeze_trunk(self) -> None:
        """特徴抽出部のパラメータを更新対象から外します。"""
        for parameter in self.trunk.parameters():
            parameter.requires_grad = False

    def attribution_layer(self, probe: torch.Tensor) -> nn.Module:
        """
        判断根拠の可視化で着目すべき層を返します。

        Grad-CAM は「特徴マップのどの位置が効いたか」を見る手法なので、
        着目する層には縦横の解像度が残っている必要があります。
        単純に最後の畳み込み層を選ぶと、大域プーリング後に置かれた 1x1 の層を
        拾ってしまい、位置情報が失われた一様なヒートマップになります。

        そこで実際に 1 枚流して各層の出力形状を調べ、解像度が残っている層のうち
        もっとも深いものを選びます。この方法ならモデル構造ごとの決め打ちが要りません。

        Parameters
        ----------
        probe : torch.Tensor
            形状を調べるために流す画像。形状 (1, 3, H, W) を想定します。

        Returns
        -------
        nn.Module
            着目対象の畳み込み層。

        Raises
        ------
        ValueError
            解像度が残る畳み込み層が 1 つも見つからない場合。
        """
        spatial_layers: list[nn.Module] = []
        handles = []

        def remember(
            module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            if output.ndim == 4 and output.shape[-2] * output.shape[-1] > 1:
                spatial_layers.append(module)

        for module in self.trunk.modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(remember))

        try:
            with torch.no_grad():
                self.trunk(probe)
        finally:
            for handle in handles:
                handle.remove()

        if not spatial_layers:
            raise ValueError("trunk has no spatially resolved Conv2d layer")
        return spatial_layers[-1]


def assemble(model_name: str, suite: TaskSuite) -> SharedTrunkClassifier:
    """
    timm の学習済みモデルを特徴抽出部として、マルチタスク分類器を組み立てます。

    Parameters
    ----------
    model_name : str
        特徴抽出部に使う timm のモデル名。
    suite : TaskSuite
        解くタスクの並び。

    Returns
    -------
    SharedTrunkClassifier
        組み立てた分類器。
    """
    trunk = timm.create_model(model_name, pretrained=True)
    feature_dim: int = trunk.get_classifier().in_features

    # 分類層を取り除き、特徴抽出器として使う
    trunk.reset_classifier(0)

    return SharedTrunkClassifier(trunk=trunk, feature_dim=feature_dim, suite=suite)
