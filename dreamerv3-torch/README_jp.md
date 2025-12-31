# 概要
- DreamerV3 の PyTorch 実装。DMC/Atari などのビジョン・プロプリオセプション環境を共通ハイパーパラメータで学習できる。
- 本リポジトリでは新たに**音声モーダルを MLP でエンコード／デコードして World Model に統合**できるようにしている（`audio` キーの観測を想定）。
- 参考実装: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1)、JAX/TensorFlow 版 DreamerV3、DrQv2 など。

# セットアップと実行
- 依存関係: `pip install -r requirements.txt`（Python 3.11 推奨）。
- DMC Vision 例: `python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk`
- ログ確認: `tensorboard --logdir ./logdir`
- Docker 利用時は `Dockerfile` に手順入り。
- 追加した音声モーダルを使う場合、環境の観測に `audio`（shape 例: `(audio_dim,)`）を含めるだけで自動的に MLP が動く。`audio_dim>0` を設定すると環境側に音声がなくてもダミー音声（ゼロ＋任意ノイズ）が自動注入されるのでパイプラインを即テスト可能。詳細は下記「音声モーダルの取り込み」を参照。

# ディレクトリと主要ファイル
- `dreamer.py` : 学習エントリポイント。環境生成、データ収集、学習ループ、チェックポイント保存を統括。
- `models.py` : World Model（RSSM + マルチモーダルエンコーダ／デコーダ + 報酬/継続率ヘッド）と、想像上のロールアウトに基づく Actor-Critic（ImagBehavior）。
- `networks.py` : RSSM、Conv/MLP ベースのエンコーダ／デコーダ、Actor/Value/Reward/Cont の各ネットワーク。今回、音声用 MLP パスを追加。
- `exploration.py` : Random / Plan2Explore による探索戦略（不一致モデルの分散で内的報酬を生成）。
- `tools.py` : ロガー、データ収集・サンプリング、確率分布ユーティリティ、学習率管理などの補助関数群。
- `envs/` : DMC/Atari/MemoryMaze/Minecraft などの環境ラッパーとアクション正規化・OneHot 化・TimeLimit 等のユーティリティ。
- `configs.yaml` : デフォルトとベンチマークごとのハイパーパラメータ定義。`encoder/decoder` に音声系の設定を追加済み。
- `parallel.py` : プロセス／スレッド越しに環境を並列化するシンプルなインターフェース。

# 学習フローの概要
1. **設定読込** (`configs.yaml`) と引数マージ後、ログ出力先やステップ数を環境ステップに合わせてスケール。
2. **環境初期化** (`make_env`)。OneHotAction や NormalizeActions、TimeLimit などのラッパーを付与。
3. **経験データのプリフィル**: ランダムポリシーで `prefill` ステップ分のエピソードを収集し、`.npz` で保存。
4. **データローダ生成**: `tools.sample_episodes` で長さ `batch_length` のトラジェクトリをストリームし、`batch_size` でスタック。
5. **学習ループ** (`Dreamer.__call__`):
   - World Model の更新 (`WorldModel._train`): 観測を `MultiEncoder` で埋め込み、RSSM でポスターリオ／プライヤを推論。デコーダ／報酬／継続率ヘッドの負の対数尤度 + KL を最小化。
   - 想像上のロールアウト (`ImagBehavior._train`): Latent 上でポリシーを展開し、価値推定とアクター更新を行う（必要に応じて Plan2Explore の内的報酬）。
   - 評価・動画生成: 周期的に `video_pred_log` で再構成＋オープンループ予測を TensorBoard に保存。
6. **チェックポイント**: `logdir/latest.pt` にモデルと Optimizer 状態を保存。

# モデル構成のポイント
- **RSSM (`networks.RSSM`)**: 離散／連続に対応する確率的遷移器。`obs_step` で観測埋め込みと前時刻の状態・行動からポスターリオを、`img_step` でポリシーが使うプライヤを計算。
- **マルチモーダルエンコーダ (`networks.MultiEncoder`)**:
  - 画像系は ConvEncoder（ストライド 2 のスタック）、ベクトル系は MLP。
  - `audio_keys` でマッチした観測は専用の MLP（`audio_layers/audio_units`）で埋め込み、他の MLP 入力と区別して結合。
- **マルチデコーダ (`networks.MultiDecoder`)**:
  - 画像は ConvTranspose 系で復元（`image_dist` が `mse` の場合 MSE 損失を採用）。
  - ベクトル観測は MLP。`audio_keys` に一致した項目は AudioDecoder MLP を通す。
- **頭部ネットワーク**: 報酬（`symlog_disc`）、継続率（Bernoulli）を RSSM の特徴から予測。`grad_heads` に含まれるヘッドは勾配を World Model に伝播。
- **挙動学習 (`ImagBehavior`)**: 想像上の軌跡に対し λ-return でターゲットを作り、アクターは REINFORCE/ダイナミクス勾配/ハイブリッドを選択可能。Critic はスロースタートターゲットで安定化。
- **探索 (`Plan2Explore`)**: RSSM 特徴を予測する不一致モデル群で分散を算出し、内的報酬として利用。

# 音声モーダルの取り込み
- 新規設定（`configs.yaml`）:
  - `encoder.audio_keys`: 観測辞書から音声を拾うキーの正規表現（デフォルト `audio`）。
  - `encoder.audio_layers` / `encoder.audio_units`: 音声 MLP の層数・ユニット数（未指定なら通常の MLP 設定を流用）。
  - `decoder.audio_keys`, `decoder.audio_layers`, `decoder.audio_units`: 復元側の設定。
  - `audio_dim` / `audio_noise_std`: 0 より大きく設定すると環境観測に `audio` が無くても `(audio_dim,)` のベクトルを自動挿入（ゼロ＋ガウスノイズ）し、学習パイプラインをすぐ動かせる。
- 使い方:
  1. 環境の観測に `audio`（例: `np.float32` の `(audio_dim,)`、もしくは低次元スペクトログラム `(freq_bins,)`）を追加。
  2. 追加の前処理は不要。`MultiEncoder` が `audio_keys` で一致したベクトルをシンログ正規化付き MLP に通し、RSSM への埋め込みに加える。
  3. `MultiDecoder` が同キーを復元し、他の観測と一緒に再構成損失を計算。これにより音声も潜在状態に反映される。
- 既存タスク（画像のみ）では音声キーが無いので動作・性能に影響はない。

# データとログの扱い
- 収集データは `traindir` / `evaldir` に `.npz`（各キーに配列）として保存。`is_first`, `is_terminal`, `discount`, `action`, `reward`, `image`, `audio` などが格納され、`sample_episodes` が長さ `batch_length` の切り出しを行う。
- `tools.Logger` が `metrics.jsonl` と TensorBoard（scalar / image / video）を出力。`video_pred_log` 有効時は再構成＋オープンループ予測を可視化。

# 参考リソース
- 音声を扱う類似研究例: https://github.com/Azuma413/sound_wm_turtlebot
- DreamerV3 (JAX): https://github.com/danijar/dreamerv3
- DreamerV2 (PyTorch): https://github.com/jsikyoon/dreamer-torch, https://github.com/RajGhugare19/dreamerv2
