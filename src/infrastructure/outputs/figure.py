"""
図の描画まわりの共通設定を行うモジュール

図を書き出すモジュールは matplotlib を直接 import せず、ここを経由する。
画面のない実行環境 (docker コンテナ) で描画が失敗しないよう、
バックエンドの選択をこの 1 か所に集約している。
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["plt", "canvas"]


class canvas:
    """
    図を確実に閉じるためのコンテキストマネージャ。

    学習 1 回で何枚も描くため、閉じ忘れるとメモリを食い潰します。

    Usage
    -----
    >>> with canvas(figsize=(6, 4)) as (figure, axes):
    ...     axes.plot([1, 2, 3])
    ...     figure.savefig("example.png")
    """

    def __init__(self, **subplots_kwargs) -> None:
        self._kwargs = subplots_kwargs
        self._figure = None

    def __enter__(self):
        self._figure, axes = plt.subplots(**self._kwargs)
        return self._figure, axes

    def __exit__(self, *exception_info) -> None:
        plt.close(self._figure)
