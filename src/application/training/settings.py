"""
1 回の学習実行を決めるパラメータをまとめるモジュール

読み込み手段 (YAML かコード直書きか) には依存しません。
設定ファイルからの生成は infrastructure/config_loader.py が担います。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrainingSettings:
    """
    学習の実行条件。

    設定ファイル (configs/celeba.yml) の各項目とフィールド名が 1 対 1 で対応します。
    項目を増やす場合は、このクラスと設定ファイルの両方に追加してください。
    """

    # 出力ディレクトリ名の末尾に付く実験名
    experiment_name: str = "tutorial_celeba"

    # 特徴抽出に使う timm のモデル名
    model_name: str = "mobilenetv4_conv_small.e2400_r224_in1k"

    learning_rate: float = 1e-5
    num_epochs: int = 10
    batch_size: int = 16

    # 動作確認用に、データセット全体から先頭何枚を切り出すか
    num_train_samples: int = 500
    num_val_samples: int = 100

    # True の場合、特徴抽出部を固定してタスクごとの出力層だけを学習する
    freeze_backbone: bool = False

    # 検証 loss が何エポック連続で改善しなければ打ち切るか。None なら打ち切らない
    patience: Optional[int] = None

    # 学習後に Grad-CAM を適用する検証画像の枚数。0 なら適用しない
    grad_cam_samples: int = 8

    seed: int = 42
