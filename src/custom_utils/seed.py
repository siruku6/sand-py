import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    全てのランダムシードを固定して再現性を確保する

    Parameters
    ----------
    seed : int
        固定するシード値

    Notes
    -----
    - Python標準のrandom、NumPy、PyTorchのシードを固定
    - CUDAを使用する場合も含めて全てのGPUシードを固定
    - 完全な再現性のためにはcudnn.deterministicとbenchmarkの設定も必要だが、
      パフォーマンス低下の可能性があるためコメントアウト
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 完全な再現性のための設定（パフォーマンス低下の可能性あり）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
