"""
学習済みモデルの重みをファイルに保存するモジュール
"""

from pathlib import Path

import torch
from torch import nn

from custom_utils.logger import init_logger

logger = init_logger()


def save(network: nn.Module, destination: Path) -> Path:
    """
    ネットワークの重みを保存します。

    モデル構造ではなく `state_dict` のみを保存するため、読み込む際は
    学習時と同じネットワークを組み立ててから `load_state_dict` してください。

    Parameters
    ----------
    network : nn.Module
        保存対象のネットワーク。
    destination : Path
        書き出し先のファイルパス。

    Returns
    -------
    Path
        書き出したファイルのパス。

    Generates:
      - <destination>
    """
    torch.save(network.state_dict(), destination)
    logger.info(f"Model saved to: {destination}")
    return destination
