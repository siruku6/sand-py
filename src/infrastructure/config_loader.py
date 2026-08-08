"""
YAML の設定ファイルを設定オブジェクトに読み込むモジュール

読み込み先のクラスは呼び出し側が指定します。この層は「ファイルを読んで詰め替える」
手段だけを提供し、どんな設定項目があるかは知りません。
"""

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

SettingsT = TypeVar("SettingsT")


def load(settings_path: Path, settings_class: type[SettingsT]) -> SettingsT:
    """
    YAML の設定ファイルを読み込み、指定した dataclass に変換します。

    設定ファイル側の書き間違いを実行前に気付けるよう、対応するフィールドが
    存在しないキーはエラーとして扱います。

    Parameters
    ----------
    settings_path : Path
        読み込む YAML ファイルのパス。
    settings_class : type
        変換先の dataclass。

    Returns
    -------
    SettingsT
        設定ファイルの内容を反映したインスタンス。
        記載のない項目には dataclass の既定値が使われます。

    Raises
    ------
    ValueError
        `settings_class` が dataclass でない場合、または設定ファイルに
        未知のキーが含まれている場合。

    Reads:
      - <settings_path>
    """
    if not is_dataclass(settings_class):
        raise ValueError(f"settings_class must be a dataclass: {settings_class}")

    with open(settings_path, "r", encoding="utf-8") as settings_file:
        entries: dict[str, Any] = yaml.safe_load(settings_file) or {}

    accepted = {field.name for field in fields(settings_class)}
    rejected = sorted(set(entries) - accepted)
    if rejected:
        raise ValueError(
            f"Unknown keys in {settings_path}: {rejected}. "
            f"Available keys: {sorted(accepted)}"
        )

    return settings_class(**entries)
