# 押して作るロボット福笑い

CRANE+ V2が上方カメラの画像を見て、口パーツを押して所定の位置・向きに合わせ、先端を離す模倣学習の課題です。ROS 2 Humble / Ignition Gazebo Fortressで実験しました。

**最終モデルは厳格v2で0/2成功でした。** 旧モデルの探索用評価1/5成功、同じ旧モデルの未使用5seed評価0/5と区別します。実機での成功、未知条件への汎化は実証していません。

## まず読むもの

1. [理解用ガイド](docs/guide.md)：目的、仕組み、成功と失敗の読み方。
2. [提出用レポートPDF（事後評価補足を含む7ページ）](report.pdf) / [同じ内容のMarkdown](docs/report.md)。
3. [詳細な実験レポート](docs/report-detailed.md)：設定、比較実験、全テスト結果。
4. [再現手順と限界](docs/reproduction.md)、[出典と自作部分](THIRD_PARTY.md)。

## 実装範囲

| 場所 | 内容 |
|---|---|
| worlds / models / launch | 福笑い盤面、口の形状、GazeboとCRANE+の起動 |
| config | 静的寸法・関節形状・依存バージョン・厳格v2採点規則 |
| scripts/collect_visual_demo.py | 画像から口を推定する教師の操作と記録 |
| scripts/synced_recorder.py | 画像・関節・操作指令の時刻同期 |
| scripts/convert_fukuwarai_to_lerobot.py | 品質・episode境界・マスクを保持した変換 |
| scripts/train_fukuwarai_act.py | 講義ACTの学習・保存・検証 |
| scripts/serve_fukuwarai_act.py | RGBと関節に基づく推論 |
| scripts/evaluate_fukuwarai_batch.py | 固定seedの実動作評価 |
| scripts/fukuwarai_joint_path.py / fukuwarai_tracking_gate.py | 共通軌道と実測追従確認 |
| scripts/fukuwarai_success.py / rescore_fukuwarai.py | 輪郭・向き・離脱・保持の採点 |
| tests | 数値処理、マスク、採点、実行制約の検査 |

比較実験の特徴抽出・損失なども、学習スクリプトから参照されるため含めています。旧座標制御バッチ、夜間自動実験、開発日誌、使用しない物体は含めません。

## 公開準備の状態

このRepoは公開用に切り出したPrivateの作業版です。元RepoのGit履歴は継承していません。学習済み重み・元画像・データセット・生ログ・動画は同梱しません。したがって、コードだけで報告済みモデルの同一推論を再現することはできません。結果の抜粋とハッシュを[evidence/results.json](evidence/results.json)に示します。独立再採点にはローカル保管の元記録が必要です。

提出動画のURLとOmniCampus用の資料名は、提出時に設定します。GitHubの所有者名はURLに残るため、公開前に提出先の匿名化要件と照合してください。

## 追加の事後評価

[大まかな配置の緩和基準](docs/posthoc-coarse-evaluation.md)では同じ記録が1/2成功。中心15mm・yaw30度へ緩め、輪郭判定を外した別基準です。厳格v2の0/2は変わりません。新規試行や性能改善ではありません。
