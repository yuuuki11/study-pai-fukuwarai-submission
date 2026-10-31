# 再現手順と再現可能な範囲

## 実験時の環境

- macOS、Python3.11、PyTorch2.6.0/MPSで学習・推論。
- Docker名 `pai-advanced1`、イメージ `airobotbook/ros2-desktop-ai-robot-book-humble:latest`、Ubuntu/ROS2 Humble/Ignition Gazebo Fortressで実行。latestタグの同一性は将来保証されません。環境の自動構築Dockerfileは同梱していません。
- コンテナでCRANE+のdescription/control、ros_gz_sim、ros_gz_bridge、ros_gz_image、controller_manager、robot_state_publisherが導入・ビルドされている必要があります。講義の環境構築を前提とします。
- このRepoをコンテナの `/workspace` にマウントし、ROSとCRANE+パッケージのsetup.bashを読み込んで実行します。

## シーンの表示

```sh
# コンテナ内。既に起動している場合は実行しない。
bash /workspace/scripts/start_fukuwarai.sh
# 必要に応じて、先端を離した状態で初期配置をリセット
python3 /workspace/scripts/reset_mouth.py --seed 42
```

起動にはGUIセッションが必要です。任意の手動操作は `scripts/teleop.sh`。自動評価・教師収集との同時起動は禁止です。モデルの物理座標は採点だけに使用します。

## 固定した講義実装

```sh
git clone https://github.com/matsuolab/pai_training.git /tmp/fukuwarai-pai-training-source
cd /tmp/fukuwarai-pai-training-source
git checkout b4d6b5459066a80433475a83567504988875248a
git submodule update --init --recursive
# lerobotは768e36660d1408c71118a2760f831c037fbfa17dに固定
```

ホストで本Repoへ戻り、Python3.11の仮想環境に `config/requirements-training.txt` を導入します。ROS側にはNumPy/Pillow/OpenCV/SciPy等、各scriptのimportを満たす必要があります。silhouette比較候補には別途 `config/requirements-silhouette-cues.txt` を使用します。

## データと学習

生画像・rosbag・重みは非同梱です。変換には `collect_visual_demo.py` の同期済み記録とresultが必要です。`convert_fukuwarai_to_lerobot.py --help` で入力と出力を指定し、`--legacy-position-criteria` は使用しません。復帰区間は元軌道と品質証拠を保持した別区間として扱います。

最終候補は25全軌道+2連続復帰区間、26/1の分割、連続性マスク、終端20%抽出重み、継続学習元チェックポイントに依存します。これらがない新規データでの学習は、報告したモデルの完全再現にはなりません。ハイパーパラメータはレポートと `evidence/training-config.json` を参照してください。

## 保存済みモデルでの評価（別途重みが必要）

```sh
.venv-training/bin/python scripts/evaluate_fukuwarai_batch.py \
  --checkpoint data/training/act_v3_terminal20_6000_20261027/checkpoint \
  --output data/evaluation/reproduction_new_run \
  --n-action-steps 8 --shared-joint-path --shared-path-projection \
  --endpoint-tracking-gate --max-sim-s 600 --seeds 2026102308 2026102305
```

実験は提出準備のため停止済みで、この手順は説明です。今回のパッケージ作成中には実行していません。旧評価seedを新規学習・調整へ混ぜないでください。新規出力を使い、コンテナ内の `/tmp/fukuwarai-expert.lock` と稼働Publisherの排他を守ります。

## ローカル検証

Python構文、shell構文、選択した単体テストを確認しています。シミュレータの新規環境構築からの再実行は未検証です。`python -m unittest discover -s tests` の全件にはROS・講義ソース等の環境依存があります。

追加100試行のrun_fukuwarai_placement100.pyは、ローカルの事前登録JSONと重み・実行環境を必要とします。collect_expert_demo.pyは評価のExpert/kinematics基底クラスとして同梱しています。旧教師を新評価の代わりに使う意味ではありません。
