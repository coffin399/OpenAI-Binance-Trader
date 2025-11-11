from __future__ import annotations

import logging
import threading
import webbrowser
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from binance_auto_trader.services.trade_tracker import TradeTracker

logger = logging.getLogger(__name__)

DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Binance Auto Trader Dashboard</title>
    <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />
    <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />
    <link
      href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap\"
      rel=\"stylesheet\"
    />
    <script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js\"></script>
    <style>
      :root {
        color-scheme: dark light;
      }
      body {
        margin: 0;
        font-family: 'Inter', sans-serif;
        background: radial-gradient(circle at top, #0f172a, #020617);
        color: #e2e8f0;
        min-height: 100vh;
      }
      .page {
        max-width: 1200px;
        margin: 0 auto;
        padding: 32px 24px 64px;
      }
      h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 12px;
      }
      .subtitle {
        color: #94a3b8;
        margin-bottom: 32px;
      }
      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 18px;
        margin-bottom: 32px;
      }
      .card {
        backdrop-filter: blur(12px);
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.75));
        border: 1px solid rgba(148, 163, 184, 0.08);
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.25);
      }
      .card h2 {
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        margin: 0 0 12px;
      }
      .metric {
        font-size: 2rem;
        font-weight: 600;
        margin: 4px 0;
      }
      .metric.small {
        font-size: 1.25rem;
      }
      .chart-card {
        display: flex;
        flex-direction: column;
        gap: 16px;
      }
      canvas {
        background: rgba(15, 23, 42, 0.55);
        border-radius: 16px;
        padding: 16px;
      }
      .list {
        display: grid;
        gap: 10px;
        margin-top: 16px;
      }
      .list-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 16px;
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.08);
      }
      .pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.12);
        color: #e2e8f0;
        font-size: 0.85rem;
      }
      .pill .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 16px;
      }
      th,
      td {
        padding: 12px 14px;
        text-align: left;
      }
      th {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
      }
      tr {
        border-bottom: 1px solid rgba(148, 163, 184, 0.08);
      }
      .two-column {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 24px;
      }
      .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 8px;
      }
      .timestamp {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-bottom: 24px;
      }
      @media (max-width: 768px) {
        .page {
          padding: 24px 16px 56px;
        }
        .metric {
          font-size: 1.5rem;
        }
      }
    </style>
  </head>
  <body>
    <main class=\"page\">
      <header>
        <h1>Binance Auto Trader</h1>
        <p class=\"subtitle\">Live trading telemetry. Mode: <span id=\"mode-pill\" class=\"pill\"></span></p>
      </header>

      <section class=\"grid\">
        <div class=\"card\">
          <h2>Total Trades</h2>
          <div class=\"metric\" id=\"metric-trades\">0</div>
          <div class=\"timestamp\" id=\"metric-updated-at\"></div>
        </div>
        <div class=\"card\">
          <h2>Active Strategies</h2>
          <div class=\"metric\" id=\"metric-strategies\">0</div>
        </div>
        <div class=\"card\">
          <h2>Average PnL %</h2>
          <div class=\"metric\" id=\"metric-average\">0%</div>
        </div>
        <div class=\"card\">
          <h2>Best Trade %</h2>
          <div class=\"metric\" id=\"metric-max\">0%</div>
        </div>
        <div class=\"card\">
          <h2>Sortino Ratio</h2>
          <div class=\"metric\" id=\"metric-sortino\">0</div>
        </div>
      </section>

      <section class=\"card chart-card\">
        <div class=\"section-title\">Price Action</div>
        <canvas id=\"price-chart\" height=\"320\"></canvas>
      </section>

      <section class=\"two-column\" style=\"margin-top: 32px;\">
        <div class=\"card\">
          <div class=\"section-title\">Open Positions</div>
          <div class=\"list\" id=\"open-trades\"></div>
        </div>
        <div class=\"card\">
          <div class=\"section-title\">Recent Closed Trades</div>
          <div class=\"list\" id=\"recent-trades\"></div>
        </div>
      </section>

      <section class=\"card\" style=\"margin-top: 32px;\">
        <div class=\"section-title\">Backtest Performance</div>
        <div class=\"grid\" style=\"margin-top: 12px; margin-bottom: 18px;\">
          <div>
            <div class=\"metric small\" id=\"bt-trades\">0</div>
            <div class=\"subtitle\">Trades</div>
          </div>
          <div>
            <div class=\"metric small\" id=\"bt-strategies\">0</div>
            <div class=\"subtitle\">Strategies</div>
          </div>
          <div>
            <div class=\"metric small\" id=\"bt-average\">0%</div>
            <div class=\"subtitle\">Average Return %</div>
          </div>
          <div>
            <div class=\"metric small\" id=\"bt-max\">0%</div>
            <div class=\"subtitle\">Max Return %</div>
          </div>
          <div>
            <div class=\"metric small\" id=\"bt-sortino\">0</div>
            <div class=\"subtitle\">Sortino Ratio</div>
          </div>
        </div>
        <table id=\"bt-table\">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Symbol</th>
              <th>Trades</th>
              <th>Avg %</th>
              <th>Max %</th>
              <th>Sortino</th>
              <th>Generated</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </section>
    </main>

    <script>
      const REFRESH_INTERVAL = {refresh_interval} * 1000;
      let priceChart;

      function formatNumber(value, options = {{ suffix: '', precision: 2 }}) {{
        if (value === null || value === undefined) return '—';
        const suffix = options.suffix || '';
        const precision = options.precision ?? 2;
        if (typeof value === 'string') return `${{value}}${{suffix}}`;
        return `${{Number(value).toFixed(precision)}}${{suffix}}`;
      }}

      function renderPriceChart(series) {{
        const ctx = document.getElementById('price-chart');
        const labels = series.length ? series[0].timestamps : [];
        const datasets = series.map((entry) => ({
          label: entry.symbol,
          data: entry.prices,
          borderColor: entry.color,
          borderWidth: 2,
          tension: 0.3,
          fill: false,
        }));

        if (priceChart) {{
          priceChart.data.labels = labels;
          priceChart.data.datasets = datasets;
          priceChart.update();
        }} else {{
          priceChart = new Chart(ctx, {{
            type: 'line',
            data: {{ labels, datasets }},
            options: {{
              plugins: {{
                legend: {{ labels: {{ color: '#cbd5f5' }} }},
              }},
              scales: {{
                x: {{ ticks: {{ color: '#94a3b8' }} }},
                y: {{ ticks: {{ color: '#94a3b8' }} }},
              }},
            }},
          }});
        }}
      }}

      function renderTradeLists(data) {{
        const openEl = document.getElementById('open-trades');
        const recentEl = document.getElementById('recent-trades');
        openEl.innerHTML = '';
        recentEl.innerHTML = '';

        data.open_trades.forEach((trade) => {{
          const el = document.createElement('div');
          el.className = 'list-item';
          el.innerHTML = `
            <div>
              <div class="pill"><span class="dot" style="background:${{trade.color}}"></span>${{trade.symbol}}</div>
              <div style="color:#94a3b8; font-size:0.85rem; margin-top:6px;">${{trade.strategy}} • ${{trade.entry_price.toFixed(4)}}</div>
            </div>
            <div style="text-align:right;">
              <div>${{trade.action}}</div>
              <div style="color:#94a3b8; font-size:0.8rem;">Since ${{trade.opened_at ? new Date(trade.opened_at).toLocaleString() : '—'}}</div>
            </div>`;
          openEl.appendChild(el);
        }});

        data.recent_closed_trades.forEach((trade) => {{
          const el = document.createElement('div');
          el.className = 'list-item';
          const pnlClass = trade.pnl_percent >= 0 ? '#10b981' : '#ef4444';
          el.innerHTML = `
            <div>
              <div class="pill"><span class="dot" style="background:${{trade.color}}"></span>${{trade.symbol}}</div>
              <div style="color:#94a3b8; font-size:0.85rem; margin-top:6px;">${{trade.strategy}}</div>
            </div>
            <div style="text-align:right;">
              <div style="color:${{pnlClass}}">${{trade.pnl_percent.toFixed(2)}}%</div>
              <div style="color:#94a3b8; font-size:0.8rem;">Closed ${{trade.closed_at ? new Date(trade.closed_at).toLocaleString() : '—'}}</div>
            </div>`;
          recentEl.appendChild(el);
        }});
      }}

      function renderBacktests(payload) {{
        const summary = payload.summary;
        document.getElementById('bt-trades').textContent = summary.trade_count;
        document.getElementById('bt-strategies').textContent = summary.strategy_count;
        document.getElementById('bt-average').textContent = formatNumber(summary.average_profit, {{ suffix: '%'}});
        document.getElementById('bt-max').textContent = formatNumber(summary.max_profit, {{ suffix: '%'}});
        document.getElementById('bt-sortino').textContent = formatNumber(summary.sortino_ratio, {{ precision: 2 }});

        const tbody = document.querySelector('#bt-table tbody');
        tbody.innerHTML = '';
        payload.results.forEach((row) => {{
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${{row.strategy}}</td>
            <td>${{row.symbol}}</td>
            <td>${{row.trade_count}}</td>
            <td>${{formatNumber(row.average_return, {{ suffix: '%'}})}}</td>
            <td>${{formatNumber(row.max_return, {{ suffix: '%'}})}}</td>
            <td>${{formatNumber(row.sortino_ratio, {{ precision: 2 }})}}</td>
            <td>${{row.generated_at ? new Date(row.generated_at).toLocaleString() : '—'}}</td>`;
          tbody.appendChild(tr);
        }});
      }}

      async function refreshDashboard() {{
        try {{
          const [summaryRes, priceRes, btRes] = await Promise.all([
            fetch('/api/summary'),
            fetch('/api/price-history'),
            fetch('/api/backtest'),
          ]);
          const summary = await summaryRes.json();
          const prices = await priceRes.json();
          const backtest = await btRes.json();

          const modeEl = document.getElementById('mode-pill');
          modeEl.textContent = summary.mode;
          modeEl.style.background = summary.mode === 'LIVE' ? 'rgba(34,197,94,0.15)' : 'rgba(59,130,246,0.15)';
          modeEl.style.border = '1px solid rgba(148, 163, 184, 0.12)';

          document.getElementById('metric-trades').textContent = summary.trade_count;
          document.getElementById('metric-strategies').textContent = summary.strategy_count;
          document.getElementById('metric-average').textContent = formatNumber(summary.average_profit, {{ suffix: '%'}});
          document.getElementById('metric-max').textContent = formatNumber(summary.max_profit, {{ suffix: '%'}});
          document.getElementById('metric-sortino').textContent = formatNumber(summary.sortino_ratio, {{ precision: 2 }});
          document.getElementById('metric-updated-at').textContent = `Updated ${new Date().toLocaleTimeString()}`;

          renderPriceChart(prices.series || []);
          renderTradeLists(summary);
          renderBacktests(backtest);
        }} catch (error) {{
          console.error('Dashboard refresh failed', error);
        }}
      }}

      refreshDashboard();
      setInterval(refreshDashboard, REFRESH_INTERVAL);
    </script>
  </body>
</html>
"""


def _build_app(trade_tracker: TradeTracker, refresh_interval: int) -> FastAPI:
    app = FastAPI(title="Binance Auto Trader Dashboard", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    dashboard_html = DASHBOARD_TEMPLATE.format(refresh_interval=refresh_interval)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:  # noqa: D401 - FastAPI endpoint
        return dashboard_html

    @app.get("/api/summary", response_class=JSONResponse)
    async def summary() -> Dict[str, Any]:
        return trade_tracker.get_summary_payload()

    @app.get("/api/price-history", response_class=JSONResponse)
    async def price_history() -> Dict[str, Any]:
        return trade_tracker.get_price_history_payload()

    @app.get("/api/backtest", response_class=JSONResponse)
    async def backtest() -> Dict[str, Any]:
        return trade_tracker.get_backtest_payload()

    return app


_dashboard_thread: threading.Thread | None = None


def start_dashboard(trade_tracker: TradeTracker, config) -> None:
    web_config = getattr(config, "web", None)
    if not web_config or not getattr(web_config, "enabled", False):
        logger.info("Web dashboard disabled in configuration")
        return

    global _dashboard_thread
    if _dashboard_thread and _dashboard_thread.is_alive():
        logger.info("Web dashboard already running")
        return

    host = getattr(web_config, "host", "127.0.0.1")
    port = int(getattr(web_config, "port", 8000))
    refresh_interval = int(getattr(web_config, "refresh_interval_seconds", 5))

    app = _build_app(trade_tracker, refresh_interval)

    def _run_server() -> None:
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        logger.info("Starting web dashboard at http://%s:%s", host, port)
        server.run()

    _dashboard_thread = threading.Thread(target=_run_server, daemon=True)
    _dashboard_thread.start()
    try:
        webbrowser.open_new_tab(f"http://{host}:{port}")
    except Exception as err:  # noqa: BLE001
        logger.warning("Failed to open browser automatically: %s", err)
