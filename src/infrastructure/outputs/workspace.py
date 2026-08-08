"""
1 回の実行に対応する出力先ディレクトリを扱うモジュール

出力を行うクラスが個別に `mkdir` やパス連結を書くと、置き場所を変えたいときに
全箇所を直すことになる。ディレクトリの決め方をここに閉じ込め、各クラスには
「どこに置くか」ではなく「何という名前で置くか」だけを決めさせる。
"""

from pathlib import Path

from custom_utils.datetime_module import now_str


class RunWorkspace:
    """1 回の実行が生成するファイルの置き場。"""

    def __init__(self, root: Path) -> None:
        """
        Parameters
        ----------
        root : Path
            この実行の出力先ディレクトリ。存在しない場合は作成します。
        """
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create_for(cls, base_dir: Path, experiment_name: str) -> "RunWorkspace":
        """
        実行日時を含む一意なディレクトリを作り、その Workspace を返します。

        同じ実験名で何度実行しても過去の結果が上書きされないよう、
        ディレクトリ名の先頭に実行日時を付けます。
        """
        return cls(base_dir / f"{now_str()}_{experiment_name}")

    @property
    def root(self) -> Path:
        """この実行の出力先ディレクトリ。"""
        return self._root

    def file(self, name: str) -> Path:
        """直下のファイルパスを返します。"""
        return self._root / name

    def folder(self, name: str) -> Path:
        """直下のサブディレクトリを作成し、そのパスを返します。"""
        path = self._root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __str__(self) -> str:
        return str(self._root)
