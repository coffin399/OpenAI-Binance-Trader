# Binance Auto Trader

Binance 向けのマルチストラテジー自動売買プラットフォームです。リアルタイム監視、バックテスト、Discord 通知、複数 AI との連携に対応し、設計の一部は [Harvard Algorithmic Trading with AI](https://github.com/moondevonyt/Harvard-Algorithmic-Trading-with-AI) を参考にしています。

> ⚠️ **実験的なベース実装です** — 本ソフトウェアは研究・検証用途を想定しています。収益性や可用性は保証されません。必ず自己責任で利用し、テストネットやドライランで十分に検証した上で本番運用に移行してください。

## 主な特徴

- 🔁 **コンフィグ駆動**：初回起動時に `config.default.yaml` が `config.yaml` にコピーされます。以降は `config.yaml` のみ編集します。
- 🧠 **AI 主導のトレード判断（デフォルト）**：OpenAI 互換 API など複数のプロバイダを登録し、失敗した API キーは自動的に次のキーへ切り替えます。
- 💴 **日本円ペア重視**：`BTC/JPY`、`ETH/JPY`、`XRP/JPY`、`SOL/JPY`、`BNB/JPY`、`DOGE/JPY` をテンプレートに収録。
- 📊 **モダンな Web ダッシュボード**：FastAPI + Chart.js でドライラン/実運用のステータス、トレード数、平均/最大利益率、Sortino レシオ、価格推移を可視化。
- 🔔 **Discord 通知**：売買した通貨・数量・価格を ANSI コード付きで投稿し、チャンネルごとにレート制限を設定可能。
- 🧪 **バックテスト内蔵**：起動時に設定済みペア / ストラテジーで過去データを評価し、結果をダッシュボードへ反映。
- 🛡️ **ドライランが標準**：`runtime.dry_run` が true の限り、実注文は Binance へ送信されません。

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

## 必要環境

- Python 3.10 以上
- Binance API キー（テストネット／本番）
- 任意：Discord Bot トークンと通知先チャンネル ID
- 任意：OpenAI / Anthropic などの API キー

## インストールと初期セットアップ

1. **仮想環境の作成と依存関係インストール（Windows）**
   ```cmd
   run_bot.bat --install-only
   ```
   `venv/` が自動作成され、`pip` 更新および `requirements` のインストールが完了します。

2. **設定ファイルの初期化**
   - 初回起動で `config.default.yaml` が `config.yaml` にコピーされます。
   - `config.yaml` を編集して API キー、取引条件、通知設定を入力します。

3. **`config.yaml` の主な設定例**

   ```yaml
   binance:
     api_key: "YOUR_API_KEY"
     api_secret: "YOUR_API_SECRET"
     testnet: true

   trading:
     symbols: ["BTC/JPY", "ETH/JPY", "DOGE/JPY"]
     timeframe: 1h
     quantity: 0.0
     max_open_trades: "3Trades"
     max_investment_per_trade: "5000JPY"

   ai:
     providers:
       - name: openai
         base_url: https://api.openai.com/v1
         model: gpt-4o-mini
         temperature: 0.2
         api_keys: ["key1", "key2"]

   discord:
     enabled: true
     bot_token: "..."
     channel_ids: ["1234567890", "2345678901"]

   runtime:
     dry_run: true
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

## AI ストラテジー

- デフォルトで `ai_hybrid` が有効。最新ローソク足データをもとに AI へ BUY / SELL / HOLD の判断を問い合わせます。
- プロバイダごとに複数 API キーを登録でき、エラーが発生したキーは自動でスキップされます。
- 独自ストラテジーを追加する際は `strategies/` にクラスを実装し、`STRATEGY_REGISTRY` に登録してください。

## バックテスト

- `backtesting.enabled` を true にすると、設定済み通貨ペアとストラテジーでヒストリカルデータを解析します。
- 平均リターン、最大リターン、Sortino レシオ、トレード数などがダッシュボードの「Backtest Performance」に表示されます。

## 安全運用のヒント

1. **必ずドライラン／テストネットで検証する**：`runtime.dry_run` を false にする前に挙動を十分確認してください。
2. **投入金額と同時建玉の制御**：`max_investment_per_trade` と `max_open_trades` を設定し、許容リスクを明確にしましょう（0 で無制限）。
3. **API キー管理**：複数キーのローテーションと安全な保管を徹底してください。
4. **ログとダッシュボードの監視**：`trading_bot.log`、Discord 通知、ダッシュボードを常時観測し、異常を検知したらすぐに停止できる体制を整えましょう。

## ライセンス

MIT License
