# outputs

- 学習済みモデル・推論結果・評価指標など、コードが生成したファイルの置き場です
- 中身は `.gitignore` の対象で、この README だけが git 管理されています

## ディレクトリの分け方

タスクごとにディレクトリを分け、その下に実行ごとのディレクトリを作ります。
実行ごとのディレクトリ名は `<実行日時>_<実験名>` の形式です。

```text
outputs/
 └─ celeba/
    └─ 20260808_112308_tutorial_celeba/
       ├── loss.csv                  <- エポックごとの学習/検証 loss
       ├── metrics.csv               <- エポック・タスクごとの accuracy / F1 など
       ├── model_best.pth            <- 検証 loss が最小だった時点の重み
       ├── model_last.pth            <- 最終エポック時点の重み
       ├── confusion_matrix/
       │   └── <タスク名>.png        <- 最終エポックの検証結果の混同行列
       └── grad_cam/
           └── <タスク名>.png        <- 判断根拠の可視化
```

これらのファイルは、学習ループを監視する観測者がそれぞれ書き出しています。
出力を増減させたい場合は [src/application/training/observers/](/src/application/training/observers/) の観測者を
足し引きしてください。組み合わせを決めているのは [src/main_celeba.py](/src/main_celeba.py) です。
