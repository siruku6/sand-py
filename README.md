# sand-py

## 概要

- 機械学習まわりの検証・試作を行うためのサンドボックスリポジトリ
- 実験は Docker コンテナ上で行い、ホスト環境に依存しないようにしています

## 環境

* ソフトウェア
    * Python 3.12
    * パッケージ等は、[`docker/experiment/pyproject.toml`](/docker/experiment/pyproject.toml) を参照
* Docker に割り当てるメモリは 4GB 以上を推奨します
    * 常駐する mlflow・tensorboard・jupyter だけで 1GB 強を消費するため、割り当てが小さいと学習プロセスが OOM で強制終了します

## ディレクトリ構成

```text
.
├── configs/               <- 設定ファイルの格納先
├── data/                  <- 学習や推論に利用するデータセットの格納先
├── docker/                <- Dockerfileなど、環境設定ファイルの格納先
├── docs/                  <- 各種ドキュメントの格納先
├── logs/                  <- 学習ログや推論ログの格納先
├── notebooks/             <- Jupyter Notebookの格納先
├── outputs/               <- モデルや推論結果など、出力ファイルの格納先
├── src/                   <- ソースコードの格納先
└── README.md              <- 開発者向けの最上位のREADME
```

ソースコードは `domain` / `application` / `infrastructure` の 3 層に分けています。
層の役割と依存の向きは [src/README.md](/src/README.md) を参照してください。

## 実行方法

以下のステップで作業を行うとスムーズです。

1. [学習用環境構築](/docker/experiment/README.md)
1. [チュートリアル (CelebA データセットで学習)](/docs/tutorial/CelebA.md)
