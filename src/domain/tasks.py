"""
「同時に解く分類タスクの集まり」を表現するモジュール

何タスクを何クラスで解くかは、モデルの出力層・損失関数・評価・混同行列の
すべてが必要とする情報である。各所が個別に定義すると必ず食い違うため、
このモジュールの値オブジェクトを唯一の情報源として受け渡す。
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationTask:
    """1 つの多クラス分類タスク。"""

    name: str

    # クラス番号の昇順に並べたクラス名。混同行列の軸ラベルにもそのまま使う
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        """このタスクのクラス数。"""
        return len(self.class_names)


class TaskSuite:
    """
    同時に解くタスクの並び。

    並び順はモデルの出力ヘッドを作る順序と一致するため、順序を保持する型として扱う。
    """

    def __init__(self, tasks: Sequence[ClassificationTask]) -> None:
        if not tasks:
            raise ValueError("TaskSuite requires at least one task")

        if len(tasks) != len({task.name for task in tasks}):
            raise ValueError("Task names must be unique")

        self._tasks: tuple[ClassificationTask, ...] = tuple(tasks)

    def __iter__(self) -> Iterator[ClassificationTask]:
        return iter(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, name: str) -> ClassificationTask:
        for task in self._tasks:
            if task.name == name:
                return task
        raise KeyError(name)

    @property
    def names(self) -> tuple[str, ...]:
        """タスク名の並び。"""
        return tuple(task.name for task in self._tasks)
