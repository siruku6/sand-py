"""
CelebA の索引と画像ファイルを PyTorch の DataLoader に橋渡しするモジュール
"""

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from domain.celeba.labels import LabelRule
from infrastructure.construct_dataset.celeba_catalog import (
    CelebaArchive,
    LabelledImages,
    Split,
)


class FaceAttributeDataset(Dataset):
    """
    顔画像 1 枚と、そのタスク別正解ラベルを返す Dataset。

    索引 (どの画像がどのラベルか) は構築済みのものを受け取り、
    このクラスは画像の読み込みとテンソル化だけを担います。

    Reads:
      - <image_dir>/<画像名>
    """

    def __init__(
        self,
        image_dir: Path,
        index: LabelledImages,
        to_tensor: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        """
        Parameters
        ----------
        image_dir : Path
            顔画像が並ぶディレクトリ。
        index : LabelledImages
            画像名とタスク別ラベルの索引。
        to_tensor : Callable
            PIL 画像をモデル入力テンソルに変換する関数。
        """
        self._image_dir = image_dir
        self._index = index
        self._to_tensor = to_tensor
        self._labels = {
            name: torch.from_numpy(values) for name, values in index.labels.items()
        }

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(
        self, position: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = Image.open(self._image_dir / self._index.image_names[position])
        targets = {name: values[position] for name, values in self._labels.items()}
        return self._to_tensor(image.convert("RGB")), targets


def build_loader(
    archive: CelebaArchive,
    split: Split,
    rules: tuple[LabelRule, ...],
    to_tensor: Callable[[Image.Image], torch.Tensor],
    batch_size: int,
    limit: int,
    shuffle: bool,
) -> DataLoader:
    """
    指定 split の DataLoader を組み立てます。

    CelebA は全体で 20 万枚を超えるため、動作確認では `limit` 枚だけを使います。

    Parameters
    ----------
    archive : CelebaArchive
        読み込み元の CelebA ディレクトリ。
    split : Split
        使用するデータ分割。
    rules : tuple[LabelRule, ...]
        適用するラベル導出規則。
    to_tensor : Callable
        PIL 画像をモデル入力テンソルに変換する関数。
    batch_size : int
        バッチサイズ。
    limit : int
        先頭から何枚を使うか。
    shuffle : bool
        エポックごとに並びを入れ替えるかどうか。

    Returns
    -------
    DataLoader
        組み立てた DataLoader。
    """
    dataset = FaceAttributeDataset(
        image_dir=archive.image_dir,
        index=archive.index(split, rules).take(limit),
        to_tensor=to_tensor,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
