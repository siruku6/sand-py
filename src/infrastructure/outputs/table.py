"""
表形式の記録を CSV に書き出すモジュール
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from custom_utils.logger import init_logger

logger = init_logger()


def save_rows(
    rows: list[dict], destination: Path, decimals: Optional[int] = None
) -> Path:
    """
    辞書の並びを CSV として保存します。

    Parameters
    ----------
    rows : list[dict]
        1 要素が 1 行になる辞書の並び。すべて同じキーを持つ前提です。
    destination : Path
        書き出し先のファイルパス。
    decimals : int | None
        小数を丸める桁数。None なら丸めずにそのまま書き出します。
        loss のように値をそのまま比較したい記録では None を指定します。

    Returns
    -------
    Path
        書き出したファイルのパス。

    Generates:
      - <destination>
    """
    frame = pd.DataFrame(rows)
    if decimals is not None:
        frame = frame.round(decimals)

    frame.to_csv(destination, index=False)
    logger.info(f"Table saved to: {destination}")
    return destination
