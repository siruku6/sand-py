"""
画像をモデル入力に揃える前処理と、その逆変換を扱うモジュール

Grad-CAM の重ね描きでは正規化前の見た目に戻す必要があるため、
正規化のパラメータと逆変換を 1 か所に持たせて食い違いを防いでいる。
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


@dataclass(frozen=True)
class FrameNormalisation:
    """モデル入力サイズと正規化パラメータの組。"""

    size: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def encoder(self) -> Callable[[Image.Image], torch.Tensor]:
        """
        PIL 画像を正規化済みテンソルに変換する関数を返します。

        データ拡張は行いません。学習時と検証時で同じ前処理を使います。
        """
        return transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(self.mean), std=list(self.std)),
            ]
        )

    def decode(self, frames: torch.Tensor) -> np.ndarray:
        """
        正規化を打ち消し、そのまま描画できる画像配列に戻します。

        Parameters
        ----------
        frames : torch.Tensor
            形状 (B, 3, H, W) の正規化済みテンソル。

        Returns
        -------
        np.ndarray
            形状 (B, H, W, 3)、値域 0.0-1.0 の配列。
        """
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        restored = frames.detach().cpu() * std + mean
        return restored.clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()
