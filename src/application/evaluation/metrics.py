"""
分類結果から評価指標を算出するモジュール
"""

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# クラスごとの成績を均等に扱う平均方法。
# CelebA はクラス不均衡が大きいため、頻度の高いクラスだけ当てても高得点にならないようにする
_AVERAGE = "macro"


@dataclass(frozen=True)
class TaskScore:
    """1 タスク分の評価指標。"""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float

    def as_dict(self) -> dict[str, float]:
        """CSV の列に展開できる辞書に変換します。"""
        return asdict(self)


def score(truth: np.ndarray, guess: np.ndarray) -> TaskScore:
    """
    1 タスク分の正解と予測から評価指標を計算します。

    Parameters
    ----------
    truth : np.ndarray
        正解クラス番号の配列 (N,)。
    guess : np.ndarray
        予測クラス番号の配列 (N,)。

    Returns
    -------
    TaskScore
        accuracy / precision / recall / F1 をまとめた値オブジェクト。
    """
    return TaskScore(
        accuracy=accuracy_score(truth, guess),
        precision_macro=precision_score(
            truth, guess, average=_AVERAGE, zero_division=0
        ),
        recall_macro=recall_score(truth, guess, average=_AVERAGE, zero_division=0),
        f1_macro=f1_score(truth, guess, average=_AVERAGE, zero_division=0),
    )
