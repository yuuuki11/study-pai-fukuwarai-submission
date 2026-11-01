# 出典と自作部分

## 講義・外部依存（ソースを同梱しない）

- [matsuolab/pai_training](https://github.com/matsuolab/pai_training)：講義の学習基盤。固定コミット `b4d6b5459066a80433475a83567504988875248a`。
- 上記のLeRobotサブモジュール：ACT本体とデータローダー。固定コミット `768e36660d1408c71118a2760f831c037fbfa17d`。ACTネットワーク自体を独自考案したものではありません。
- ROS2 Humble / Ignition Gazebo Fortress / CRANE+ descriptionとcontrol：シミュレーション、ロボット形状、制御基盤。外部環境に導入されたものを利用。
- [CRANE+教材ソース](https://github.com/AI-Robot-Book-Humble/crane_plus)：コミット `19c1d6069126cd27488bf1bbefd410704bfc515d`。ロボットdescription/controlと静的関節形状の出典。
- [airobotbook](https://github.com/AI-Robot-Book)：実験コンテナとCRANE+を含む教材環境。
- [matsuolab/physicalai_teleop](https://github.com/matsuolab/physicalai_teleop)：初期の講義と同じキーボード操作で利用。ソース改変なし、同梱なし。

外部依存の取得・利用にはそれぞれのライセンスが適用されます。このRepoで外部コードに新しいライセンスを付け直していません。CRANE+由来の静的関節形状値をconfigに保持しています。

## 本課題で実装・変更した部分

福笑い盤面と口のモデル、画像ベースの教師制御、同期記録、品質ゲート、LeRobot変換、maskと分割、因果的特徴、ACT学習の接続と抽出重み、推論HTTP接続、共通軌道、安全制約、追従ゲート、厳格v2採点、比較評価とレポートを作成しました。講義ACTのネットワークは外部実装を利用し、学習・推論周辺を実装しました。

AIコーディング支援を実装・検証・文章整理に使用しました。実験結果は保存記録に基づき、失敗も報告しています。

動画のOP/EDは講座提供の任意素材です。このコードRepoには配布素材・完成動画を含めていません。

## PDF用日本語フォント

Noto Sans JP（Google Fonts / SIL Open Font License 1.1）をPDFに埋め込む。tools/fonts/NotoSansJP-Regular.ttf は公式可変フォントのweight=400を静的化したもの。ライセンスはtools/fonts/OFL.txt。取得元: https://github.com/google/fonts/tree/main/ofl/notosansjp 。
