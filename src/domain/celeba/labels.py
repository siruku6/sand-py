"""
CelebA の 40 種類の属性から、各タスクの正解ラベルを導くルールを定義するモジュール

「属性をそのまま使う」タスクと「複数の属性を優先順位で 1 つに畳み込む」タスクが
混在するため、導出方法をルールオブジェクトとして差し替え可能にしている。
タスクを増やすときは、このモジュールにルールを 1 つ足すだけで済む。
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from domain.tasks import ClassificationTask, TaskSuite


class LabelRule(Protocol):
    """属性テーブルから 1 タスク分の正解ラベルを導く規則。"""

    @property
    def task(self) -> ClassificationTask:
        """この規則が担当するタスク。"""
        ...

    @property
    def required_attributes(self) -> tuple[str, ...]:
        """
        この規則が参照する属性名。

        CelebA の属性表は 40 列 20 万行あり、全列を保持すると無視できない量の
        メモリを使います。規則自身に必要な列を申告させ、読み込み側が
        それだけを読めるようにしています。
        """
        ...

    def derive(self, attributes: pd.DataFrame) -> np.ndarray:
        """
        属性テーブルからクラス番号の配列を導きます。

        Parameters
        ----------
        attributes : pd.DataFrame
            0/1 に正規化済みの属性テーブル。行が画像、列が属性名。

        Returns
        -------
        np.ndarray
            クラス番号の配列 (N,)。
        """
        ...


@dataclass(frozen=True)
class PassThroughFlag:
    """0/1 の属性を、そのまま 2 クラスのラベルとして使う規則。"""

    task: ClassificationTask
    attribute: str

    @property
    def required_attributes(self) -> tuple[str, ...]:
        return (self.attribute,)

    def derive(self, attributes: pd.DataFrame) -> np.ndarray:
        return attributes[self.attribute].to_numpy(dtype=np.int64)


@dataclass(frozen=True)
class FirstMatchingFlag:
    """
    複数の 0/1 属性を調べ、最初に立っていた属性のクラス番号を採用する規則。

    「黒髪でも茶髪でもある」ような重複を、宣言した順序で決着させます。
    どの属性も立っていない場合は `fallback` を採用します。
    """

    task: ClassificationTask

    # 調べる順序 = 優先順位。属性名とクラス番号の組を順に並べる
    candidates: tuple[tuple[str, int], ...]

    # どの候補にも該当しなかった場合のクラス番号
    fallback: int

    @property
    def required_attributes(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.candidates)

    def derive(self, attributes: pd.DataFrame) -> np.ndarray:
        # np.select は最初に成立した条件を採用するため、candidates の並び順が優先順位になる
        conditions = [attributes[name] == 1 for name, _ in self.candidates]
        class_numbers = [number for _, number in self.candidates]
        return np.select(conditions, class_numbers, default=self.fallback).astype(
            np.int64
        )


GENDER = ClassificationTask(name="gender", class_names=("female", "male"))
SMILE = ClassificationTask(name="smile", class_names=("not_smiling", "smiling"))
HAIR = ClassificationTask(name="hair", class_names=("black", "blond", "brown", "other"))


def default_rules() -> tuple[LabelRule, ...]:
    """チュートリアルで解く 3 タスク分のラベル導出規則を返します。"""
    return (
        PassThroughFlag(task=GENDER, attribute="Male"),
        PassThroughFlag(task=SMILE, attribute="Smiling"),
        FirstMatchingFlag(
            task=HAIR,
            candidates=(("Black_Hair", 0), ("Blond_Hair", 1), ("Brown_Hair", 2)),
            fallback=3,
        ),
    )


def suite_of(rules: tuple[LabelRule, ...]) -> TaskSuite:
    """ラベル導出規則の並びから、対応するタスクの並びを組み立てます。"""
    return TaskSuite([rule.task for rule in rules])
