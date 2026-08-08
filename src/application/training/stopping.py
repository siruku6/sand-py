"""
検証スコアの推移から、ベスト更新と打ち切りを判定するモジュール

「打ち切るかどうか」は実験のたびに方針が変わるため、判定を差し替え可能にしている。
判定側はモデルの保存を行わない。何をするかは判定を受け取った側が決める。
"""

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class Verdict:
    """1 エポック分のスコアに対する判定。"""

    # 過去の最良スコアを更新したか
    improved: bool

    # ここで学習を打ち切るべきか
    exhausted: bool


class StopPolicy(Protocol):
    """エポックごとのスコアを受け取り、続行するかを判定する方針。"""

    def judge(self, score: float) -> Verdict:
        """
        スコアを 1 つ受け取り、判定を返します。

        Parameters
        ----------
        score : float
            監視対象のスコア。小さいほど良い値を想定します。

        Returns
        -------
        Verdict
            ベスト更新と打ち切りの判定。
        """
        ...


class _BestScore:
    """これまでの最良スコアを覚え、更新されたかを答える内部ヘルパ。"""

    def __init__(self, minimum_gain: float) -> None:
        self._best: Optional[float] = None
        self._minimum_gain = minimum_gain

    def offer(self, score: float) -> bool:
        """スコアを提示し、最良を更新したかを返します。"""
        if self._best is not None and score >= self._best - self._minimum_gain:
            return False

        self._best = score
        return True


class RunEveryEpoch:
    """打ち切りを行わず、ベスト更新の判定だけを返す方針。"""

    def __init__(self, minimum_gain: float = 0.0) -> None:
        """
        Parameters
        ----------
        minimum_gain : float
            更新とみなすための最小改善幅。わずかな揺らぎを更新と数えたくない場合に使います。
        """
        self._best = _BestScore(minimum_gain)

    def judge(self, score: float) -> Verdict:
        return Verdict(improved=self._best.offer(score), exhausted=False)


class StopWhenStale:
    """改善しないまま指定エポック数が経過したら打ち切る方針。"""

    def __init__(self, patience: int, minimum_gain: float = 0.0) -> None:
        """
        Parameters
        ----------
        patience : int
            何エポック連続で更新が無ければ打ち切るか。
        minimum_gain : float
            更新とみなすための最小改善幅。
        """
        if patience < 1:
            raise ValueError(f"patience must be 1 or greater: {patience}")

        self._patience = patience
        self._best = _BestScore(minimum_gain)
        self._stale_epochs = 0

    def judge(self, score: float) -> Verdict:
        improved = self._best.offer(score)
        self._stale_epochs = 0 if improved else self._stale_epochs + 1
        return Verdict(
            improved=improved, exhausted=self._stale_epochs >= self._patience
        )


def policy_for(patience: Optional[int]) -> StopPolicy:
    """
    設定値から打ち切り方針を選びます。

    Parameters
    ----------
    patience : int | None
        None なら打ち切らない方針、整数ならその回数で打ち切る方針を返します。
    """
    return RunEveryEpoch() if patience is None else StopWhenStale(patience)
