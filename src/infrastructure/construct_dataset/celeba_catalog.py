"""
CelebA の配布ファイルから、画像名と正解ラベルの一覧を組み立てるモジュール

画像そのものは 20 万枚を超えるため、ここでは「どの画像がどのラベルを持つか」の
索引だけを作る。画像の読み込みは Dataset 側の責務とする。
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from domain.celeba.labels import LabelRule


class Split(Enum):
    """CelebA が公式に定めるデータ分割。値は list_eval_partition.txt の分割番号。"""

    TRAIN = 0
    VALIDATION = 1
    TEST = 2


@dataclass(frozen=True)
class LabelledImages:
    """1 つの split に属する画像名と、そのタスク別正解ラベル。"""

    image_names: tuple[str, ...]

    # タスク名 -> クラス番号の配列 (N,)。並びは image_names と対応する
    labels: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.image_names)

    def take(self, count: int) -> "LabelledImages":
        """先頭 `count` 件だけを取り出した索引を返します。"""
        return LabelledImages(
            image_names=self.image_names[:count],
            labels={name: values[:count] for name, values in self.labels.items()},
        )


class CelebaArchive:
    """
    展開済みの CelebA ディレクトリを読むアダプタ。

    ファイル名や属性値の符号 (-1/1) といった CelebA 固有の事情を吸収し、
    外にはクラス番号の配列だけを渡します。

    Reads:
      - <root>/list_eval_partition.txt
      - <root>/list_attr_celeba.txt
    """

    PARTITION_FILE = "list_eval_partition.txt"
    ATTRIBUTE_FILE = "list_attr_celeba.txt"
    IMAGE_FOLDER = "img_align_celeba"

    def __init__(self, root: Path) -> None:
        """
        Parameters
        ----------
        root : Path
            CelebA の各ファイルを配置したディレクトリ。
        """
        self._root = root

    @property
    def image_dir(self) -> Path:
        """顔画像が並ぶディレクトリ。"""
        return self._root / self.IMAGE_FOLDER

    def index(self, split: Split, rules: tuple[LabelRule, ...]) -> LabelledImages:
        """
        指定 split に属する画像名と、規則から導いた正解ラベルを返します。

        Parameters
        ----------
        split : Split
            取り出すデータ分割。
        rules : tuple[LabelRule, ...]
            適用するラベル導出規則の並び。

        Returns
        -------
        LabelledImages
            画像名とタスク別ラベルの索引。
        """
        wanted = tuple(
            dict.fromkeys(name for rule in rules for name in rule.required_attributes)
        )
        attributes = self._attributes_of(split, wanted)
        return LabelledImages(
            image_names=tuple(attributes.index),
            labels={rule.task.name: rule.derive(attributes) for rule in rules},
        )

    def _attributes_of(self, split: Split, wanted: tuple[str, ...]) -> pd.DataFrame:
        """指定 split の必要な属性列だけを、0/1 に正規化して読み込みます。"""
        partitions = pd.read_csv(
            self._root / self.PARTITION_FILE,
            sep=r"\s+",
            header=None,
            names=["image", "split"],
            index_col="image",
            dtype={"split": np.int8},
        )
        attributes = pd.read_csv(
            self._root / self.ATTRIBUTE_FILE,
            sep=r"\s+",
            header=1,  # 1 行目は画像枚数なので、2 行目を属性名のヘッダとして扱う
            index_col=0,  # 画像名を index にする
            # 40 列すべてを読むとメモリを無駄に使うため、規則が申告した列だけを読む。
            # 値は -1 / 1 しか取らないので最小の整数型で受ける
            usecols=list(wanted),
            dtype={name: np.int8 for name in wanted},
        )

        belongs_to_split = partitions.index[partitions["split"] == split.value]

        # 属性値は -1 / 1 で格納されているため 0 / 1 に変換する
        return (attributes.loc[belongs_to_split] + 1) // 2
