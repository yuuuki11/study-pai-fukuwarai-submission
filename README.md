# 押して作るロボット福笑い

CRANE+ V2が俯瞰画像を見て口パーツを押し、顔の目標位置・向きに配置する模倣学習の課題です。ROS 2 Humble / Ignition Gazebo Fortressで実験しました。教師データ収集の運用にGPT-6 Lunaを使い、ACTで関節角の列を学習・予測します。

**最終状態で成功12/50件、成功率24%でした。** 成功12件は最終画像を目視でも確認しました。実機や未知の摩擦・形状での性能は未検証です。

## 資料

- [提出用PDF（8ページ）](report.pdf) / [同じ内容のMarkdown](docs/report.md)
- [理解用ガイド](docs/guide.md)
- [再現手順と限界](docs/reproduction.md)
- [出典・自作部分・ライセンス](THIRD_PARTY.md)
- [全50件の最終判定](evidence/placement50-final-outcomes.csv) / [最終状態の保持時間](evidence/placement50-terminal-state-review.json)

## 成功条件と評価方法

中心誤差10mm以内、向き20度以内、傾き5度以内、接地高さ誤差2mm以内、先端が盤面から40mm以上離れた状態で、終了時まで全条件を1simulation秒以上保持した場合を成功とします。輪郭一致は判定対象外です。全50件を分母とし、記録品質不足のモデル失敗6件も失敗に含めます。

モデル・物理・実行条件を固定し、過去の教師・評価記録にないseedで最大600simulation秒ずつ評価しました。当初100件を予定し、途中結果確認後に元seed順の先頭50件で終了しました。過去の別モデル・別条件の成否は合算していません。全動画の体系的な目視監査は未実施です。

## 実装

| 場所 | 内容 |
|---|---|
| worlds / models / launch | 福笑い盤面、口の形状、GazeboとCRANE+の起動 |
| config | 寸法・関節形状・依存バージョン・評価条件 |
| scripts/collect_visual_demo.py | 画像ベースの教師操作と記録 |
| scripts/synced_recorder.py | 画像・関節・操作指令の時刻同期 |
| scripts/convert_fukuwarai_to_lerobot.py | 品質・区間境界・マスクを保つ変換 |
| scripts/train_fukuwarai_act.py | 講義ACTの学習・保存・検証 |
| scripts/serve_fukuwarai_act.py | RGB・関節・因果的特徴による推論 |
| scripts/evaluate_fukuwarai_batch.py | 固定seedの実行評価 |
| scripts/fukuwarai_joint_path.py / fukuwarai_tracking_gate.py | 共通軌道と追従確認 |
| scripts/score_fukuwarai_placement.py | 保存観測から配置を採点 |
| tests | 数値処理・データ変換・採点・実行制約の検査 |

## 同梱範囲と再現上の制約

使用コード・シーン・設定・テスト・レポート・結果抜粋を収録しました。元開発RepoのGit履歴は継承していません。重み・元画像・データセット・生ログ・動画は同梱していないため、Repoだけで報告済みモデルの同一推論や独立再採点を完了することはできません。原記録は開発環境に保持しています。

実験履歴の詳細は[詳細レポート](docs/report-detailed.md)、元100件の事前登録と50件への変更は[プロトコル](docs/placement100-protocol-2026-10-31.md)およびevidenceを参照してください。古い比較結果や事後評価は当時の記録であり、今回の24%とは異なる評価群です。
