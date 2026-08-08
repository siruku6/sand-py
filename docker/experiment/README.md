# Docker での環境構築

流れを以下に記載する

## 1. コードの動作に必要なファイルの準備

- コンテナ内では `uv run` でコードを実行します。 `uv` は起動したディレクトリの `pyproject.toml` を参照するため、リポジトリのルートディレクトリに配置してください

    ```bash
    # sand-py のルートディレクトリで実行
    $ cp docker/experiment/pyproject.toml ./pyproject.toml
    ```

- `src/` ディレクトリや `data/` ディレクトリと同階層に `pyproject.toml` が配置されていれば問題ありません。

## 2. Python 環境の準備

- docker コンテナの build (CPU のみの環境)

    ```bash
    $ cd docker/experiment/
    $ cp .env.example .env

    $ docker compose build
    ```

- GPU 環境の場合

    ```bash
    $ cd docker/experiment/
    $ cp .env.example .env

    # GPU 用の override ファイルを用意する
    $ cp docker-compose.override.sample.yml docker-compose.override.yml

    $ docker compose build
    ```

## 3. Docker 環境へのアクセス

- 以下のコマンドで、 docker コンテナ内に入ることが可能です

    ```bash
    $ docker compose up -d
    $ docker compose exec uv_env bash
    ```

## 4. jupyter へのアクセス

- ローカルマシンで `docker compose up` した場合は、 `localhost:{JUPYTER_PORT}` から jupyter へアクセス可能です
- JUPYTER_PORT は `.env` ファイルに設定されているポート番号です
- password 欄には `demo` と入力すれば利用可能です

## 5. 割り当てメモリの確認

`docker compose up` は mlflow・tensorboard・jupyter の3つを常駐させるため、それだけで 1GB 強を消費します。
そのうえで学習を回すので、Docker に割り当てるメモリが小さいと学習プロセスが OOM で強制終了します。

- 目安として、**4GB 以上**を割り当ててください
- 学習時のメモリ使用量は「約 400MB + 13.5MB × バッチサイズ」程度です (MobileNetV4 / 224x224 の場合)
- Docker Desktop は設定画面から、 colima は起動オプションから変更します

    ```bash
    # colima の場合の例
    $ colima stop
    $ colima start --cpu 4 --memory 8
    ```

## 6. 動作確認

- 動作確認用のデータセットとして、 [celebA を利用するチュートリアル](/docs/tutorial/CelebA.md) を記載しています
- この手順により、環境構築に成功しているかの動作確認が可能です
