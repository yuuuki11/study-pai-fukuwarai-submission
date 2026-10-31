# パッケージ検証（2026-10-31）

- Python構文コンパイル、全shellのbash構文検査：合格。
- unittest discover：97件、失敗0、skip12（ROS等の実行環境依存）。新たな学習・評価・収集は実行しない。
- Markdownの相対リンク：存在確認。
- テキストファイルの個人ホームパス、開発者名、既知の認証token形式：検出なし。
- 選択した元コードとSHA256をevidence/source-manifest.jsonへ記録。モデル入力・採点・物理のコードは切り出し時に変更しない。
- PDF全7ページを画像化して確認。日本語フォント埋め込み、個人氏名・所属・メールアドレスを記載しない。
- コードの完全な新規環境構築・オンライン再実行は未検証。必要なデータと重みは非同梱。

PDFの再生成はtools/render_report.py（Python/reportlab/Pillow）。docs/report.mdが本文。既定フォントはmacOSのArial Unicode.ttf。別OSでは使用可能な日本語TrueTypeフォントを指定するようスクリプトを調整する。

Git履歴は新規作成し、Author/Committerは汎用のSubmission Author。GitHub所有者名やpushしたアカウントの情報は別であり、公開前の匿名化確認が必要。
