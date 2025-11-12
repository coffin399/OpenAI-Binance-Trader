# Binance Auto Trader

![Python Version](https://img.shields.io/badge/python-3.11-blue?logo=python)
![Status](https://img.shields.io/badge/status-experimental-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Binance 向けの高度な AI 自動売買プラットフォームです。AI による数量決定、保有資産活用、積極的トレード戦略に対応し、リアルタイム監視、バックテスト、Discord 通知、Web ダッシュボードを提供します。

> ⚠️ **実験的なベース実装です** — 本ソフトウェアは研究・検証用途を想定しています。収益性や可用性は保証されません。必ず自己責任で利用し、テストネットやドライランで十分に検証した上で本番運用に移行してください。

## 🚀 主な特徴

### 🔥 **最新 AI 機能**
- **🧠 AI 数量決定**: `quantity: 0.0` 設定で AI が市場状況に基づき最適な取引数量を自動計算
- **📈 積極的トレード**: 動的信頼度閾値、市場機会検出、ボラティリティ活用で積極的なエントリーを実現
- **💰 保有資産活用**: 保有中の資産を自動売却して新規ポジション開始、資産効率を最大化

### 🔧 **高度な取引機能**
- **🔄 ポジション復元**: ボット再起動時にオープンポジションを自動検知・復元
- **🎯 少額対応**: 4000円からの少額スタートに最適化、リスク管理機能強化
- **🛡️ LOT_SIZE対応**: Binance の取引制約に自動対応、エラーを防止

### 📊 **可視化・管理**
- **🌐 モダン Web ダッシュボード**: FastAPI + Chart.js でリアルタイム監視、バックテスト結果表示
- **🔔 Discord 通知**: 取引結果を ANSI コード付きで整形、レート制限対応
- **📱 リアルタイム監視**: 複数通貨ペアの同時監視、戦略別パフォーマンス表示

## ディレクトリ構成

```
binance_auto_trader/
├── ai/
├── config/
├── exchange/
├── models/
├── services/
├── strategies/
├── utils/
└── web/
```

## 🛠️ 必要環境

- Python 3.10 以上
- Binance API キー（テストネット／本番）
- 任意：Discord Bot トークンと通知先チャンネル ID
- 任意：OpenAI / OpenRouter などの AI API キー

## 🚀 インストールと初期セットアップ

### 1. **ワンクリックインストール（Windows）**
```cmd
run_bot.bat --install-only
```
仮想環境 `venv/` が自動作成され、依存関係がインストールされます。

### 2. **設定ファイルの初期化**
- 初回起動で `config.default.yaml` が `config.yaml` にコピー
- `config.yaml` を編集して API キーや取引条件を設定

### 3. **最新設定例**

```yaml
binance:
  api_key: "YOUR_API_KEY"
  api_secret: "YOUR_API_SECRET"
  testnet: false  # 本番取引

trading:
  symbols: ["BTC/JPY", "ETH/JPY", "DOGE/JPY"]
  timeframe: 1m
  quantity: 0.0  # AI 数量決定モード
  max_open_trades: "6Trades"
  max_investment_per_trade: "0JPY"  # AI が投資額も決定

# 初期資金設定（4000円少額スタート対応）
initial_capital:
  jpy_amount: 4000
  allow_start_from_cash: true

# 保有資産活用設定
asset_management:
  use_held_assets: true  # 保有資産を売却して新規ポジション
  max_asset_utilization: 0.8  # 80% 利用率
  prefer_profitable_assets: true  # 利益資産を優先売却

ai:
  providers:
    - name: openrouter
      base_url: https://openrouter.ai/api/v1
      model: "deepseek/deepseek-chat-v3.1:free"
      temperature: 0.6
      api_keys: ["YOUR_AI_API_KEY"]

strategies:
  active:
    - ai_hybrid
  ai_hybrid:
    min_confidence: 0.30        # 低めの信頼度で積極的に取引
    aggressive_mode: true        # 積極モード
    use_technical_signals: true  # 技術指標を活用
    dynamic_threshold: true      # 動的閾値
    opportunity_detection: true # 機会検出

runtime:
  dry_run: false  # 実際の取引を実行
```

### 環境変数による上書き例

| 環境変数 | 内容 |
| --- | --- |
| `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `BINANCE_TESTNET` | Binance 認証情報やテストネット設定 |
| `TRADING_SYMBOLS` | `BTC/JPY,DOGE/JPY` のように区切りで複数指定 |
| `OPENAI_API_KEYS`, `ANTHROPIC_API_KEYS` 等 | プロバイダ名（大文字）+ `_API_KEYS` で複数キー登録 |
| `DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_IDS` | Discord 通知設定 |
| `TRADING_DRY_RUN` | `true` / `false` でドライラン切り替え |

## 起動方法

### ワンクリック起動

```cmd
run_bot.bat
```

仮想環境の有無をチェックし、依存関係のインストールから bot 本体・ダッシュボード起動までを自動的に実行します。

### 手動起動

```bash
python -m binance_auto_trader.main
```

- 起動直後にバックテスト（有効時）が実行され、結果がダッシュボードに反映されます。
- 続いてライブ取引ループが開始され、設定した通貨ペアとストラテジーを順番に処理します。

## Web ダッシュボード

- 既定 URL：`http://127.0.0.1:8000`
- 上部カードに運用モード、トレード数、稼働ストラテジー数、平均・最大利益率、Sortino レシオを表示。
- 価格チャートは銘柄ごとに色分けされ、建玉と直近クローズ済みトレードを同時に確認できます。
- バックテスト結果のテーブルも表示され、`web.refresh_interval_seconds` で更新間隔を調整可能。

## Discord 通知

- `discord.enabled` が true なら、ログが Discord にもストリーミングされます。
- 売買時に `TRADE OPEN ...` / `TRADE CLOSE ...` の形式で銘柄・数量・価格を投稿。ANSI コード付きで読みやすく整形。
- `rate_limit.max_messages`・`rate_limit.per_seconds` で通知頻度を制御できます。

## 🤖 AI Hybrid 戦略の強化機能

### 1. **AI 数量決定**
市場状況に基づき最適な取引数量を AI が自動計算：
- **BTC**: 0.00005〜0.0005（50円〜500円相当）
- **ETH**: 0.0005〜0.005（15円〜150円相当）
- **DOGE**: 10〜200（10円〜200円相当）

### 2. **積極的トレード戦略**
```yaml
aggressive_mode: true
dynamic_threshold: true
opportunity_detection: true
```
- **動的信頼度**: RSI、モメンタム、ボリュームで閾値を動的調整
- **機会検出**: 強いトレンドやボリューム急増を自動検知
- **HOLD 回避**: 積極モードでは HOLD を最小化

### 3. **技術指標による補強**
- **RSI**: 30/70 閾値で売買シグナル
- **モメンタム**: SMA クロスオーバーでトレンド判定
- **ボリューム**: 1.5倍以上で信頼度アップ
- **ボラティリティ**: 高ボラ時にリスク管理

## 💰 保有資産活用機能

### 自動資産売却
```
JPY残高不足 → 保有資産価値を計算 → 必要額を売却 → 新規エントリー
```

### 売却戦略
- **価値優先**: JPY 価値の大きい資産から売却
- **利益優先**: 利益が出ている資産を優先（設定可能）
- **利用率制御**: 80% 利用率で全売却リスクを回避

### 復元機能
ボット再起動時に保有ポジションを自動検知：
```
Restoring open positions from exchange...
Restored position: ETH/JPY qty=0.001 price=533471.0000
Position restoration completed. Restored 1 positions.
```

## バックテスト

- `backtesting.enabled` を true にすると、設定済み通貨ペアとストラテジーでヒストリカルデータを解析します。
- 平均リターン、最大リターン、Sortino レシオ、トレード数などがダッシュボードの「Backtest Performance」に表示されます。

## 🛡️ 安全運用ガイド

### 1. **段階的運用**
1. `dry_run: true` でテスト
2. `testnet: true` でテストネット運用
3. 少額から本番運用開始

### 2. **リスク管理**
- **少額スタート**: 4000円など少額から開始
- **数量制限**: AI 数量決定で過大取引を防止
- **資産活用**: 80% 利用率でリスクを管理

### 3. **監視体制**
- **Web ダッシュボード**: 常時監視
- **ログ確認**: `trading_bot.log` で詳細確認
- **Discord 通知**: リアルタイムで取引状況を把握

### 4. **推奨設定（少額スタート）**
```yaml
initial_capital:
  jpy_amount: 4000  # 4000円から開始

asset_management:
  max_asset_utilization: 0.8  # 80% 利用率

strategies:
  ai_hybrid:
    min_confidence: 0.30  # 積極的な信頼度
```

## 📝 ログ例

### AI 数量決定
```
AI calculated quantity: 0.0001 for BTC/JPY
✅ USING AI QUANTITY: 0.0001 -> 0.0001
Determined quantity: 0.0001 for BTC/JPY
```

### 保有資産活用
```
Using held assets. Available: 2328 JPY (cash: 363 + assets: 1965)
Selling assets to secure 137 JPY
Sold ETH 0.001 for 426 JPY
Successfully sold assets to secure 137 JPY
```

### ポジション復元
```
Restoring open positions from exchange...
Restored position: ETH/JPY qty=0.001 price=533471.0000
Position restoration completed. Restored 1 positions.
```

## 🔧 環境変数

| 環境変数 | 内容 |
| --- | --- |
| `BINANCE_API_KEY`, `BINANCE_API_SECRET` | Binance 認証情報 |
| `TRADING_SYMBOLS` | `BTC/JPY,ETH/JPY` のようにカンマ区切り |
| `OPENROUTER_API_KEYS` | AI API キー（カンマ区切りで複数） |
| `DISCORD_BOT_TOKEN` | Discord Bot トークン |
| `TRADING_DRY_RUN` | `true` / `false` でドライラン切り替え |

## 🤝 貢献

- バグ報告や機能要望は Issue で受け付けています
- プルリクエストも歓迎します

## 📄 ライセンス

MIT License - 詳細は LICENSE ファイルを参照してください。

---

> 💡 **ヒント**: まずは `dry_run: true` で動作を確認し、慣れてきたら少額での本番運用を開始することをお勧めします。
