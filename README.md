# Alpamayo-CARLA Integration

NVIDIA [Alpamayo-R1](https://huggingface.co/nvidia/Alpamayo-R1-10B) および [Alpamayo-1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B) VLA (Vision-Language-Action) モデルを [CARLA](https://carla.org/) シミュレータ上で動作させるためのクライアント実装です。

2つの動作モードを備えています：

| モード | スクリプト | 概要 |
|--------|-----------|------|
| **Agent（閉ループ）** | `run_agent_nxt.py` | Alpamayoの推論結果で車両を実際に制御する |
| **Observer（開ループ）** | `run_observer_nxt.py` | CARLAオートパイロットが走行し、Alpamayoは観察・推論のみ |

どちらもPygameダッシュボードで4カメラ映像・BEV軌跡・CoT推論テキスト・HUD情報をリアルタイム表示し、MP4録画にも対応しています。

---

## 動作環境

| 項目 | バージョン / 要件 |
|------|------------------|
| OS | Ubuntu 22.04+ (テスト済み) |
| Python | 3.12 |
| GPU | CUDA対応GPU（VRAM 24GB以上推奨、モデル≈22GB） |
| CARLA | 0.9.15 または 0.9.16（Docker推奨） |
| Alpamayo | `nvidia/Alpamayo-R1-10B` または `nvidia/Alpamayo-1.5-10B`（HuggingFace、初回自動ダウンロード） |

---

## 環境構築

### 1. CARLA サーバーの起動

Docker で CARLA を起動します（GPU パススルー必須）：

```bash
# CARLA 0.9.15 の例
docker run --privileged --gpus all --net=host \
  -e DISPLAY=$DISPLAY \
  carlasim/carla:0.9.15 \
  /bin/bash ./CarlaUE4.sh -RenderOffScreen -nosound
```

> **注意**: クライアント（Python carla パッケージ）とサーバーのバージョンを合わせてください。バージョンが異なると警告やSegfaultが発生します。

### 2. Python 仮想環境のセットアップ

Alpamayo-R1 の依存関係を含む venv を使用します：

```bash
# Alpamayo リポジトリをクローン
git clone https://github.com/NVIDIA/alpamayo.git /path/to/alpamayo

# venv 作成 & 有効化
python3.12 -m venv /path/to/alpamayo/ar1_venv
source /path/to/alpamayo/ar1_venv/bin/activate

# Alpamayo-R1 パッケージのインストール（editable mode）
pip install -e /path/to/alpamayo

# CARLA Python クライアント
pip install carla==0.9.15  # サーバーに合わせる（Python 3.12 は 0.9.16 のみ対応の場合あり）

# その他依存パッケージ
pip install pygame numpy scipy torch torchvision transformers
```

> **Python バージョンに関する注意**: `carla==0.9.15` は Python 3.10 までの wheel しか提供していません。Python 3.12 を使用する場合は `carla==0.9.16` + CARLA 0.9.16 サーバーの組み合わせが必要です。

### 3. ディレクトリ構成

```
alpamayo-carla/
├── README.md                 # このファイル
├── examples/
│   ├── run_agent_nxt.py      # Agent モード起動スクリプト
│   └── run_observer_nxt.py   # Observer モード起動スクリプト
└── src/
    ├── __init__.py
    ├── alpamayo_wrapper_nxt.py         # Alpamayo R1/1.5 モデルのロード・推論ラッパー
    ├── carla_alpamayo_agent_nxt.py     # Agent モード本体（閉ループ制御）
    ├── carla_observer_nxt.py           # Observer モード本体（開ループ観察）
    ├── display_nxt.py                  # Pygame ダッシュボード表示
    ├── nav_planner_nxt.py              # ナビゲーション指示生成（Alpamayo 1.5用）
    ├── sensor_manager_nxt.py           # CARLA カメラ・センサー管理
    └── trajectory_optimizer_nxt.py     # 軌跡後処理（平滑化・快適性最適化）
```

---

## 使い方

### 共通：venv の有効化とディレクトリ移動

```bash
source /path/to/alpamayo/ar1_venv/bin/activate
cd alpamayo-carla/examples
```

---

### Agent モード（閉ループ自動運転）

Alpamayo の推論結果を使って車両を実際に制御します。毎ティック（0.1秒）ごとに推論を実行し、Pure-Pursuit + 速度制御で車両を操縦します。

`--model` で指定したモデル名からバージョン（R1 / 1.5）を自動検出します。

```bash
# === Alpamayo R1 ===
source /path/to/alpamayo/ar1_venv/bin/activate
cd alpamayo-carla/examples

python run_agent_nxt.py --frames 500 --map Town01
python run_agent_nxt.py --frames 500 --map Town01 --record output.mp4
python run_agent_nxt.py --dummy --frames 200

# === Alpamayo 1.5（ナビゲーション付き） ===
source /path/to/alpamayo1.5/a1_5_venv/bin/activate
cd alpamayo-carla/examples

# 基本実行（ナビゲーション自動有効）
python run_agent_nxt.py --model nvidia/Alpamayo-1.5-10B --frames 500 --map Town03

# CFG ナビゲーション（ガイダンス強化）
python run_agent_nxt.py --model nvidia/Alpamayo-1.5-10B --cfg-nav --frames 500

# ナビゲーション無効（R1 相当の動作）
python run_agent_nxt.py --model nvidia/Alpamayo-1.5-10B --no-nav --frames 500
```

#### Agent オプション一覧

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--host` | `localhost` | CARLA サーバーホスト |
| `--port` | `2000` | CARLA サーバーポート |
| `--map` | *(現在のマップ)* | 使用マップ（Town01〜Town15, Town10HD 等） |
| `--weather` | `ClearNoon` | 天候プリセット |
| `--frames` | `500` | 最大推論フレーム数 |
| `--dummy` | `false` | ダミーモデル使用（GPU不要テスト用） |
| `--spawn` | `-1`(ランダム) | スポーンポイントインデックス |
| `--vehicle` | `vehicle.mercedes.coupe_2020` | 車両ブループリント |
| `--model` | `nvidia/Alpamayo-1.5-10B` | HuggingFace モデルID（`nvidia/Alpamayo-R1-10B` で R1 使用） |
| `--no-nav` | `false` | ナビゲーション指示を無効化（1.5 のみ。R1 では無視） |
| `--cfg-nav` | `false` | ナビゲーションに CFG（分類器自由ガイダンス）を使用（1.5 のみ） |
| `--cfg-nav-weight` | *(モデルデフォルト)* | CFG ガイダンス重み α（1.5 のみ） |
| `--nav-dest` | `-1`(ランダム) | ルート目的地のスポーンポイントインデックス |
| `--max-speed` | `30.0` | 最大速度 (km/h) |
| `--min-speed` | `0.0` | 最低巡航速度 (km/h)。推論が低速で不安定な場合に設定 |
| `--steer-gain` | `1.0` | ステアリングゲイン倍率（>1 でカーブを鋭く） |
| `--max-gen-len` | `256` | VLM 最大生成トークン数（64 は高速だが推論が弱くなる） |
| `--num-traj-samples` | `6` | 軌跡候補サンプル数（多い=安定だが遅い） |
| `--diffusion-steps` | `5` | フローマッチング拡散ステップ数 |
| `--cam-res` | `full` | カメラ解像度: `full`(1900×1080), `half`(960×540), `low`(640×360), `WxH` |
| `--temperature` | `0.6` | VLM テキスト生成温度（低い=決定論的） |
| `--top-p` | `0.98` | Nucleus sampling 閾値 |
| `--sim-fps` | `10.0` | シミュレーション FPS（10Hz = 0.1秒/ティック、訓練時と同一） |
| `--inference-interval` | `1` | N ティックごとに推論実行 |
| `--no-display` | `false` | Pygame ダッシュボード無効化 |
| `--record PATH` | *(なし)* | MP4 録画出力先 |
| `--crf` | `23` | H.264 CRF（0=ロスレス, 23=標準, 51=最低品質） |
| `--traj-opt` | `false` | 軌跡オプティマイザ有効化 |
| `--traj-opt-smooth` | `1.0` | 平滑性コスト重み |
| `--traj-opt-deviation` | `0.1` | 元軌跡からの逸脱コスト重み |
| `--traj-opt-comfort` | `2.0` | 快適性ペナルティ重み |
| `--traj-opt-iter` | `50` | 最適化最大反復回数 |
| `--no-retime` | `false` | Frenet retiming 無効化 |
| `--retime-alpha` | `0.25` | Retiming 強度 [0..1] |

---

### Observer モード（開ループ評価）

CARLA の TrafficManager オートパイロットが車両を運転し、Alpamayo はカメラ映像を受け取って推論のみを行います。推論結果は制御に一切使用されません。

推論はバックグラウンドスレッドで非同期に実行され、シミュレーション時間は推論中も継続して進行します。BEV 表示ではAlpamayo の予測軌跡と実際の走行軌跡を重ねて比較できます。

```bash
# 基本実行（Town03マップ、3000ティック≈5分）
python run_observer_nxt.py --ticks 3000 --map Town03

# MP4 録画付き
python run_observer_nxt.py --ticks 3000 --map Town01 --record obs01.mp4

# NPC 交通を追加（車両20台、歩行者10人）
python run_observer_nxt.py --ticks 3000 --map Town03 --npc-vehicles 20 --npc-walkers 10

# GPU なしテスト
python run_observer_nxt.py --dummy --ticks 500

# オートパイロット速度を制限速度より20%速く（デフォルト）
python run_observer_nxt.py --ticks 3000 --autopilot-speed -20
```

#### Observer オプション一覧

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--host` | `localhost` | CARLA サーバーホスト |
| `--port` | `2000` | CARLA サーバーポート |
| `--map` | *(現在のマップ)* | 使用マップ |
| `--weather` | `ClearNoon` | 天候プリセット |
| `--ticks` | `3000` | 最大シミュレーションティック数 |
| `--dummy` | `false` | ダミーモデル使用 |
| `--spawn` | `-1`(ランダム) | スポーンポイントインデックス |
| `--vehicle` | `vehicle.mercedes.coupe_2020` | 車両ブループリント |
| `--model` | `nvidia/Alpamayo-R1-10B` | HuggingFace モデルID |
| `--autopilot-speed` | `-20.0` | オートパイロット速度 % オフセット（負=速い, 正=遅い） |
| `--npc-vehicles N` | `0` | スポーンするNPC車両数 |
| `--npc-walkers N` | `0` | スポーンするNPC歩行者数 |
| `--max-gen-len` | `256` | VLM 最大生成トークン数（64 は高速だが推論が弱くなる） |
| `--num-traj-samples` | `6` | 軌跡候補サンプル数 |
| `--diffusion-steps` | `5` | フローマッチング拡散ステップ数 |
| `--cam-res` | `full` | カメラ解像度 |
| `--temperature` | `0.6` | VLM テキスト生成温度 |
| `--top-p` | `0.98` | Nucleus sampling 閾値 |
| `--sim-fps` | `10.0` | シミュレーション FPS |
| `--no-display` | `false` | Pygame ダッシュボード無効化 |
| `--record PATH` | *(なし)* | MP4 録画出力先 |
| `--crf` | `23` | H.264 CRF |

---

## ダッシュボード表示

Pygame ウィンドウに以下が表示されます：

```
┌──────────────────────────────────────────────────────┐
│  camera_cross_left  │  camera_front_wide  │  camera_  │
│     _120fov         │    _120fov          │  cross_   │
│                     │                     │  right_   │
│                     │                     │  120fov   │
├─────────────────────┴─────────────────────┴──────────┤
│ camera_front_  │         │  Chain-of-Thought          │
│   tele_30fov   │  BEV    │  推論テキスト              │
│                │  鳥瞰図  │                           │
│  HUD情報       │         │                           │
└────────────────┴─────────┴───────────────────────────┘
```

### BEV（鳥瞰図）の凡例

- **Agent モード**:
  - 緑線: 選択された軌跡（medoid）
  - 灰色線: 候補軌跡
  - 青線: モデル生の軌跡
  - オレンジ線: 最適化後の軌跡（`--traj-opt` 使用時）
- **Observer モード**:
  - 緑線 (alpamayo): Alpamayo の予測軌跡
  - パステル赤線 (actual): 実際の車両走行軌跡（推論入力時刻〜推論完了時刻まで）
  - 灰色線: 候補軌跡

### HUD 情報

- 速度 (km/h)
- 推論回数
- 推論時間
- Observer モードでは: 遅延ティック数、オートパイロット制御値 (throttle/brake/steer)

### 操作

- **ESC** またはウィンドウ閉じ: 終了
- **Ctrl+C**: ターミナルから終了

---

## アーキテクチャ概要

### カメラ構成

Alpamayo-R1 の学習時のカメラリグ（NVIDIA Hyperion）を再現しています：

| カメラ名 | FOV | 向き | 用途 |
|---------|-----|------|------|
| `camera_cross_left_120fov` | 120° | 左55° | 左方交差確認 |
| `camera_front_wide_120fov` | 120° | 正面 | 前方広角 |
| `camera_cross_right_120fov` | 120° | 右55° | 右方交差確認 |
| `camera_front_tele_30fov` | 30° | 正面 | 前方望遠 |

各カメラは 4 フレーム分のテンポラルバッファを保持し、合計 **4カメラ × 4フレーム = 16画像** をモデルに入力します。

### Ego History

車両の過去の位置・姿勢を **16ステップ × 0.1秒 = 1.6秒分** 保持し、モデルに入力します。座標系はリグフレーム（後輪軸原点、X前方、Y左、Z上）です。

### 推論パイプライン

1. カメラフレーム収集（4カメラ × 4テンポラル）
2. Ego history 構築（16ステップの位置・回転）
3. Alpamayo-R1 推論
   - VLM による Chain-of-Thought テキスト生成
   - フローマッチング拡散モデルによる軌跡生成（複数サンプル）
4. Medoid 選択（サンプル間のL2距離が最小のものを選択）
5. （Agent のみ）Pure-Pursuit ステアリング + FF+P 速度制御で車両操縦

### Agent モード制御の特徴

- **Pure-Pursuit ステアリング**: 速度に応じたアダプティブルックアヘッド
- **FF+P 速度制御**: Feed-Forward + Proportional 制御（定常偏差を排除）
- **非対称EMA平滑化**: 加速時は緩やか、減速時は即座に反応
- **曲率ベース速度制限**: Menger曲率から安全速度を計算し、カーブで自動減速
- **多数決Go/Stop判定**: 全サンプルの軌跡長で発進/停止を投票

### Observer モードの特徴

- **非同期推論**: バックグラウンドスレッドで推論し、メインループは止めない
- **フローズンスナップショット**: 推論完了時に「推論入力時刻からの実走行軌跡」を計算して保持。次の推論完了まで表示を維持
- **遅延表示**: 推論結果の入力時刻から現在までのティック遅延をHUDに表示
- **NPC交通**: ランダムな車両・歩行者をスポーン可能

---

## パフォーマンスに関する注記

- 推論は **RTX 4090** で約 **1.8〜2.0 秒/推論**（`num-traj-samples=6`, `diffusion-steps=5`）
- サンプル数を増やすと線形に時間が増加（6→12 で約2倍）
- `--cam-res half` や `--cam-res low` でレンダリング負荷を軽減できますが、推論速度への影響は軽微です（モデル内部でリサイズされるため）
- `--temperature` を下げる（例: `0.1`）と CoT テキストが安定し、軌跡のばらつきも減少します

---

## 既知の制限事項

- **低速時の推論不安定**: 停車状態から発進する際、モデルが非常に短い軌跡を出力することがあり、発進に時間がかかることがあります。`--min-speed` で緩和可能です。
- **マップ依存性**: Town10HD など路肩が広いマップでは、路肩をレーンと誤認識してそちらに乗り上げることがあります。Town01, Town03 等のシンプルなマップで安定します。
- **90度カーブ**: 急カーブでは曲がりきれずに膨らむことがあります。`--steer-gain` を上げることで改善できる場合があります。
- **CARLA サーバーとの接続**: 長時間実行時や NPC 大量スポーン時にサーバーが応答しなくなることがあります。エラーが発生した場合はグレースフルに終了します。

---

## ライセンス

Alpamayo-R1 モデルは [NVIDIA のライセンス](https://huggingface.co/nvidia/Alpamayo-R1-10B) に従います。CARLA は MIT ライセンスです。本リポジトリのコードのライセンスは別途定義してください。

