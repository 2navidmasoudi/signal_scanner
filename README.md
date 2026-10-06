# Crypto Signal Scanner

An alert-only scanner for the 53 requested USDT markets. It uses CCXT public OHLCV data, ranks the available markets once per scan, and can send the best qualified setup to Telegram. It does not place orders and does not need exchange API credentials.

Candidate ranking combines higher-timeframe and entry-timeframe EMA trends, MACD momentum, RSI, 20-candle breakouts, relative volume, ADX trend strength, low-ADX Bollinger Band reversals, and fixed-range volume-profile context. The technical layer also calculates SMA20/50, Ichimoku 9/26/52, MFI14, UTC-session VWAP, 10-bar OBV direction, KDJ 9/3/3, DMI (+DI/-DI with ADX), and regular MACD divergence from confirmed swing pivots. RSI, EMA, MACD, and ADX were already present. New indicators are grouped as trend, volume-flow, and momentum confirmations; their combined score bonus is capped at 8 points to limit double-counting of correlated signals. Triangle breakout and TD Setup 9 checks are advisory proxies, not standalone trade gates. The TD proxy only detects a fresh 9-bar Setup completion; it does not implement the Countdown or perfected-setup rules. The triangle proxy fits three confirmed swing highs and lows and requires a converging boundary breakout with volume confirmation. A candidate must also pass volume, RSI extension, and live-price drift filters before it can be alerted. An extended RSI can pass only with a directional breakout and stronger volume confirmation. Entry, stop, and target references use the current in-progress entry candle; malformed, stale, or missing current candle data is rejected instead of being replaced with an old close. Stops and targets are calculated from ATR. Scores are unvalidated rule-based rankings, not probabilities or profit guarantees. The displayed score is capped at 100; raw scores break ties between capped candidates. Confirmed patterns opposing the signal direction are displayed separately as counter-signals. If no candidate reaches the threshold or passes its quality filters, the bot reports `NO TRADE`.

ICT and RTM-inspired detections are advisory confluence and never block a candidate by themselves. The ICT proxy checks a confirmed swing-level liquidity sweep, a displacement close through the opposing confirmed swing, a three-candle fair-value gap, and a first retest of that gap. The RTM-inspired proxy checks a compact base, a strong departure, and a first return with rejection. Additional price-action checks mark fresh three-candle FVGs, recent close-throughs of confirmed swings as BOS or CHoCH, and the last opposing candle before such a break as an order-block zone. FVG and order-block retests count as confirmations only when the latest candle touches the fresh zone and closes with a directional rejection. BOS/CHoCH labels use a simple higher-high/higher-low or lower-high/lower-low trend proxy. The FRVP uses a trailing fixed range of completed entry candles, reports POC and the configured value area (70% by default), and is advisory; candles' volume is allocated to rows in proportion to price-range overlap, so the profile is an OHLCV approximation rather than trade-level volume-at-price. Qualified setups are ranked by heuristic score first; matching confirmed patterns break score ties before relative volume and ADX. These signals do not bypass the score, volume, RSI, drift, or data-quality checks. Missing or conflicting detections are shown for context and are not hard blocks.

## Windows setup

1. Install Python 3.11 or newer.
2. Open PowerShell in this folder and install the dependencies:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   Copy-Item .env.example .env
   ```

3. Edit `.env`. Telegram is optional. To enable it, set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`; keep `.env` private. Public market data does not require exchange keys.
4. Start the scanner with `py -3 main.py` or double-click `run.bat`.

The first scan begins immediately. Further scans start every 30 seconds by default. The default entry timeframe is 5m, with a 4h trend filter. Shorter entry candles can react sooner and produce more noise. The terminal uses colored tables and status panels; Telegram messages use HTML formatting, color-signaling emoji, and a compact setup summary. Telegram sends only the top-ranked READY setup per scan. It suppresses repeat alerts for the same symbol and direction while that setup remains the best READY setup, including across scans and restarts; a different best setup or a transition through NO TRADE rearms notifications. When no READY setups remain, it sends one silent NO TRADE update on entering that state; repeated unchanged scans are suppressed. Scanning continues every 30 seconds. The scanner does not calculate historical expectancy or model fees, spread, slippage, funding, liquidation, open positions, or exits.

## Historical backtest

`backtest.py` reuses the scanner's signal calculation and ranking, loads public historical OHLCV through CCXT, and evaluates the highest-ranked READY setup at each entry-candle open. It uses only already-closed entry and trend candles to create each signal. Entry fills are modeled at the next candle open with adverse slippage; the default full-position exit is TP1, with stop-first handling when a candle touches both stop and target. Choose `--exit-target tp2` to evaluate TP2 instead. Trades from later candles may overlap, because a later alert does not prove an earlier position has closed.

Example PowerShell commands:

```powershell
py -3 backtest.py --start 2026-06-01 --end 2026-09-01
py -3 backtest.py --start 2026-06-01 --end 2026-09-01 --score-sweep 55,60,65,70,75 --split-date 2026-08-01
```

Dates are UTC and `--end` is exclusive. The script warms up indicators before `--start` and fetches up to the configured holding horizon after `--end` so late-window trades can finish. `--score-sweep` compares thresholds; by default the report uses a chronological 70/30 in-sample/out-of-sample split. Use the out-of-sample numbers to check whether an apparent improvement persists, and keep a later untouched period for final evaluation. Each run is cached under `backtest_cache` and writes per-trade details to a date-stamped CSV; `--output` changes the CSV path.

The default estimated cost is 5 bps taker fee and 3 bps slippage per side. Set `--fee-bps` and `--slippage-bps` to realistic values for the account and market. The default maximum holding period is 288 bars (24 hours with 5m entry candles), configurable with `--max-hold-bars`. Reports include win rate, expectancy, profit factor, total R, realized trade-level drawdown, censored trades, and maximum overlapping positions. Overlapping trades are treated as independent one-unit-risk outcomes, so this is not an account-level portfolio simulation. Funding, spread, liquidation, partial fills, latency, and exchange fee tiers are not modeled. Historical performance, including positive out-of-sample results, does not guarantee future returns.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `EXCHANGE_ID` | `binanceusdm` | Any exchange id supported by CCXT |
| `MARKET_TYPE` | `swap` | Market category: `swap`, `future`, or `spot` |
| `ENTRY_TIMEFRAME` | `5m` | Entry signal candle size |
| `TREND_TIMEFRAME` | `4h` | Higher timeframe filter |
| `OHLCV_LIMIT` | `200` | Candles fetched per market and timeframe |
| `MAX_CONCURRENCY` | `8` | Maximum in-flight candle requests |
| `FRVP_LOOKBACK_BARS` | `96` | Completed entry candles in the rolling fixed profile range (8 hours on 5m candles) |
| `FRVP_ROWS` | `48` | Price bins used to estimate the profile |
| `FRVP_VALUE_AREA_PERCENT` | `0.70` | Volume percentage used to expand the value area from POC |
| `SCAN_INTERVAL_SECONDS` | `30` | Delay between scan starts |
| `MIN_SIGNAL_SCORE` | `65` | Minimum score required for a qualified signal |
| `MIN_VOLUME_RATIO` | `1.0` | Minimum volume relative to the prior 20 entry candles |
| `MAX_LONG_RSI` | `70` | Long RSI ceiling unless breakout confirmation is strong |
| `MIN_SHORT_RSI` | `30` | Short RSI floor unless breakdown confirmation is strong |
| `BREAKOUT_VOLUME_RATIO` | `1.5` | Volume required to override the RSI extension filter |
| `MAX_ENTRY_DRIFT_ATR` | `0.25` | Maximum current-price drift from the signal close, in ATR units |
| `TELEGRAM_BOT_TOKEN` | empty | Telegram bot token |
| `TELEGRAM_CHAT_ID` | empty | Destination chat id |

Markets that are not listed, active, or available in the selected market type are reported and skipped. Exchange listings and regional access can vary. No technical strategy can guarantee the most profitable trade; review alerts and risk before acting.
