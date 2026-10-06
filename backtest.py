"""Historical replay for the alert-only scanner.

Signals use only candles known at the signal time. A signal is evaluated at the
next entry-candle open, and the simulated entry uses adverse slippage.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ccxt.async_support as ccxt
import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from main import (
    SYMBOLS,
    TIMEFRAME_MS,
    Config,
    Signal,
    closed_candles,
    compute_signal,
    resolve_markets,
)


console = Console()
CACHE_DIR = Path(__file__).with_name("backtest_cache")


@dataclass(frozen=True)
class MarketHistory:
    requested_symbol: str
    market_symbol: str
    entry: np.ndarray
    trend: np.ndarray
    entry_times: np.ndarray
    trend_times: np.ndarray


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: str
    score: int
    signal_ms: int
    entry_ms: int
    exit_ms: int | None
    status: str
    gross_r: float | None
    net_r: float | None
    mae_r: float | None
    mfe_r: float | None


def parse_utc(value: str) -> int:
    """Parse a UTC date or ISO timestamp and return milliseconds."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date/time: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1_000)


def cache_path(
    cache_dir: Path,
    exchange_id: str,
    market_type: str,
    requested: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> Path:
    return cache_dir / f"{exchange_id}_{market_type}_{requested}_{timeframe}_{since_ms}_{until_ms}.csv"


def read_cache(path: Path) -> list[list[float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [[float(value) for value in row] for row in csv.reader(handle) if len(row) == 6]


def write_cache(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


async def fetch_history(
    exchange: Any,
    exchange_id: str,
    market_type: str,
    market_symbol: str,
    requested_symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    cache_dir: Path,
) -> list[list[float]]:
    path = cache_path(cache_dir, exchange_id, market_type, requested_symbol, timeframe, since_ms, until_ms)
    if path.exists():
        rows = read_cache(path)
        if rows:
            console.print(f"[dim]Cache hit[/dim] {requested_symbol} {timeframe}: {len(rows):,} candles")
            return rows

    duration = TIMEFRAME_MS[timeframe]
    limit = 1_000
    cursor = since_ms
    by_timestamp: dict[int, list[float]] = {}
    while cursor < until_ms:
        batch = await exchange.fetch_ohlcv(
            market_symbol,
            timeframe=timeframe,
            since=cursor,
            limit=limit,
        )
        if not batch:
            break
        usable = []
        for raw in batch:
            if len(raw) < 6:
                continue
            timestamp = int(raw[0])
            if since_ms <= timestamp < until_ms:
                row = [float(value) for value in raw[:6]]
                by_timestamp[timestamp] = row
                usable.append(timestamp)
        if not usable:
            break
        last_timestamp = max(usable)
        next_cursor = last_timestamp + duration
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < limit:
            break

    rows = [by_timestamp[key] for key in sorted(by_timestamp)]
    if rows:
        write_cache(path, rows)
    return rows


async def load_market_history(
    exchange: Any,
    requested_symbol: str,
    market_symbol: str,
    config: Config,
    start_ms: int,
    fetch_end_ms: int,
    cache_dir: Path,
    semaphore: asyncio.Semaphore,
) -> MarketHistory | None:
    entry_since = start_ms - config.candle_limit * TIMEFRAME_MS[config.entry_timeframe]
    trend_since = start_ms - config.candle_limit * TIMEFRAME_MS[config.trend_timeframe]
    trend_end = min(fetch_end_ms, int(time.time() * 1_000)) + TIMEFRAME_MS[config.trend_timeframe]
    try:
        async with semaphore:
            entry_rows = await fetch_history(
                exchange, config.exchange_id, config.market_type, market_symbol, requested_symbol, config.entry_timeframe,
                entry_since, fetch_end_ms, cache_dir,
            )
        async with semaphore:
            trend_rows = await fetch_history(
                exchange, config.exchange_id, config.market_type, market_symbol, requested_symbol, config.trend_timeframe,
                trend_since, trend_end, cache_dir,
            )
        if not entry_rows or not trend_rows:
            logger_message = f"{requested_symbol}: missing entry or trend candles"
            console.print(f"[yellow]Data skipped:[/yellow] {logger_message}")
            return None
        # Validate every downloaded row while retaining the current candle, whose
        # already-known opening price can be used for an entry-time decision.
        entry = closed_candles(entry_rows, config.entry_timeframe, int(entry_rows[-1][0]) + TIMEFRAME_MS[config.entry_timeframe])
        trend = closed_candles(trend_rows, config.trend_timeframe, int(trend_rows[-1][0]) + TIMEFRAME_MS[config.trend_timeframe])
        return MarketHistory(
            requested_symbol,
            market_symbol,
            entry,
            trend,
            entry[:, 0].astype(np.int64),
            trend[:, 0].astype(np.int64),
        )
    except Exception as exc:
        console.print(f"[yellow]Data skipped:[/yellow] {requested_symbol}: {type(exc).__name__}: {str(exc)[:140]}")
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay the scanner's READY signals against historical CCXT candles."
    )
    parser.add_argument("--start", required=True, type=parse_utc, help="UTC start date/time, inclusive (for example 2026-06-01)")
    parser.add_argument("--end", required=True, type=parse_utc, help="UTC end date/time, exclusive; allow time after it for exits")
    parser.add_argument("--symbols", help="Comma-separated subset; default is all 53 scanner symbols")
    parser.add_argument("--min-score", type=int, help="Override MIN_SIGNAL_SCORE from .env")
    parser.add_argument("--score-sweep", help="Optional comma-separated thresholds to compare, e.g. 55,60,65,70,75")
    parser.add_argument("--split-date", type=parse_utc, help="Optional out-of-sample split date; default is 70%% through the requested window")
    parser.add_argument("--exit-target", choices=("tp1", "tp2"), default="tp1", help="Exit the full simulated position at TP1 or TP2; default: tp1")
    parser.add_argument("--max-hold-bars", type=int, default=288, help="Close at candle close after this many bars; default 288 (24h on 5m)")
    parser.add_argument("--fee-bps", type=float, default=5.0, help="Estimated taker fee per side in basis points; default 5")
    parser.add_argument("--slippage-bps", type=float, default=3.0, help="Adverse execution slippage per side in basis points; default 3")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="Historical candle cache folder")
    parser.add_argument("--output", type=Path, help="Trade-level CSV output; default name includes the date range")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent market-history downloads; default 3")
    return parser


def rank_key(signal: Signal) -> tuple[int, int, float, float]:
    return (signal.raw_score, len(signal.price_action_confirmations), signal.volume_ratio, signal.adx)


def signal_candidates_at(
    timestamp: int,
    histories: list[MarketHistory],
    config: Config,
    entry_duration: int,
    trend_duration: int,
) -> list[tuple[Signal, int, MarketHistory]]:
    candidates: list[tuple[Signal, int, MarketHistory]] = []
    for history in histories:
        entry_times = history.entry_times
        current_index = int(np.searchsorted(entry_times, timestamp, side="left"))
        if current_index >= len(history.entry) or int(entry_times[current_index]) != timestamp or current_index == 0:
            continue

        # At timestamp t, bars before t are closed. The bar opening at t
        # supplies only its open price, which was available at signal time.
        prior_entry = history.entry[max(0, current_index - config.candle_limit):current_index]
        if len(prior_entry) < 60 or timestamp - int(prior_entry[-1, 0]) != entry_duration:
            continue
        trend_times = history.trend_times
        closed_trend_count = int(np.searchsorted(trend_times, timestamp - trend_duration, side="right"))
        prior_trend_all = history.trend[:closed_trend_count]
        prior_trend = prior_trend_all[-config.candle_limit:]
        if len(prior_trend) < 60:
            continue
        if timestamp - (int(prior_trend[-1, 0]) + trend_duration) > trend_duration:
            continue

        signal, _ = compute_signal(
            history.requested_symbol,
            history.market_symbol,
            prior_entry,
            prior_trend,
            config,
            float(history.entry[current_index, 1]),
        )
        if signal is not None:
            candidates.append((signal, current_index, history))
    return candidates


def simulate_trade(
    signal: Signal,
    index: int,
    history: MarketHistory,
    closed_count: int,
    entry_duration: int,
    max_hold_bars: int,
    exit_target: str,
    fee_bps: float,
    slippage_bps: float,
) -> Trade:
    sign = 1.0 if signal.side == "LONG" else -1.0
    slip = slippage_bps / 10_000.0
    entry_price = signal.price * (1.0 + sign * slip)
    target = signal.target_1 if exit_target == "tp1" else signal.target_2
    # The plan levels are anchored to the signal's observed price. Incorporate
    # entry slippage in realized risk and exclude impossible/inverted trades.
    risk = sign * (entry_price - signal.stop)
    if not math.isfinite(risk) or risk <= 0:
        return Trade(
            symbol=signal.requested_symbol,
            side=signal.side,
            score=signal.score,
            signal_ms=signal.candle_timestamp,
            entry_ms=int(history.entry[index, 0]),
            exit_ms=None,
            status="INVALID",
            gross_r=None,
            net_r=None,
            mae_r=None,
            mfe_r=None,
        )

    last_index = min(index + max_hold_bars, closed_count)
    adverse = 0.0
    favorable = 0.0
    for bar_index in range(index, last_index):
        row = history.entry[bar_index]
        bar_open, high, low, close = (float(row[column]) for column in (1, 2, 3, 4))
        if sign > 0:
            adverse = max(adverse, max(0.0, entry_price - low) / risk)
            favorable = max(favorable, max(0.0, high - entry_price) / risk)
            stop_hit = low <= signal.stop
            target_hit = high >= target
            if stop_hit:
                raw_exit = min(bar_open, signal.stop)
                exit_price = raw_exit * (1.0 - slip)
                status = "STOP"
            elif target_hit:
                raw_exit = max(bar_open, target)
                exit_price = raw_exit * (1.0 - slip)
                status = "TP1" if exit_target == "tp1" else "TP2"
            else:
                continue
            gross = exit_price - entry_price
        else:
            adverse = max(adverse, max(0.0, high - entry_price) / risk)
            favorable = max(favorable, max(0.0, entry_price - low) / risk)
            stop_hit = high >= signal.stop
            target_hit = low <= target
            if stop_hit:
                raw_exit = max(bar_open, signal.stop)
                exit_price = raw_exit * (1.0 + slip)
                status = "STOP"
            elif target_hit:
                raw_exit = min(bar_open, target)
                exit_price = raw_exit * (1.0 + slip)
                status = "TP1" if exit_target == "tp1" else "TP2"
            else:
                continue
            gross = entry_price - exit_price
        fees = (entry_price + exit_price) * fee_bps / 10_000.0
        return Trade(
            symbol=signal.requested_symbol,
            side=signal.side,
            score=signal.score,
            signal_ms=signal.candle_timestamp,
            entry_ms=int(history.entry[index, 0]),
            exit_ms=int(row[0]) + entry_duration,
            status=status,
            gross_r=gross / risk,
            net_r=(gross - fees) / risk,
            mae_r=adverse,
            mfe_r=favorable,
        )

    bars_seen = max(0, last_index - index)
    if bars_seen < max_hold_bars:
        return Trade(
            symbol=signal.requested_symbol,
            side=signal.side,
            score=signal.score,
            signal_ms=signal.candle_timestamp,
            entry_ms=int(history.entry[index, 0]),
            exit_ms=None,
            status="CENSORED",
            gross_r=None,
            net_r=None,
            mae_r=adverse,
            mfe_r=favorable,
        )

    row = history.entry[last_index - 1]
    raw_exit = float(row[4])
    exit_price = raw_exit * (1.0 - sign * slip)
    gross = sign * (exit_price - entry_price)
    fees = (entry_price + exit_price) * fee_bps / 10_000.0
    return Trade(
        symbol=signal.requested_symbol,
        side=signal.side,
        score=signal.score,
        signal_ms=signal.candle_timestamp,
        entry_ms=int(history.entry[index, 0]),
        exit_ms=int(row[0]) + entry_duration,
        status="TIME",
        gross_r=gross / risk,
        net_r=(gross - fees) / risk,
        mae_r=adverse,
        mfe_r=favorable,
    )


def max_open_trades(trades: list[Trade]) -> int:
    events: list[tuple[int, int]] = []
    for trade in trades:
        if trade.exit_ms is None or trade.status in {"CENSORED", "INVALID"}:
            continue
        events.append((trade.entry_ms, 1))
        events.append((trade.exit_ms, -1))
    open_count = peak = 0
    for _, change in sorted(events, key=lambda event: (event[0], event[1])):
        open_count += change
        peak = max(peak, open_count)
    return peak


def metrics(trades: list[Trade]) -> dict[str, float | int]:
    valid = [trade for trade in trades if trade.net_r is not None]
    values = [float(trade.net_r) for trade in valid]
    winners = [value for value in values if value > 0]
    losers = [value for value in values if value < 0]
    net_wins = sum(value for value in values if value > 0)
    net_losses = -sum(value for value in values if value < 0)
    profit_factor = net_wins / net_losses if net_losses > 0 else (math.inf if net_wins > 0 else 0.0)

    realized = sorted(valid, key=lambda trade: (trade.exit_ms or 0, trade.entry_ms))
    equity = peak = max_drawdown = 0.0
    for trade in realized:
        equity += float(trade.net_r or 0.0)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return {
        "trades": len(valid),
        "censored": sum(trade.status == "CENSORED" for trade in trades),
        "win_rate": len(winners) / len(valid) * 100.0 if valid else 0.0,
        "expectancy": sum(values) / len(values) if values else 0.0,
        "median_r": float(np.median(values)) if values else 0.0,
        "profit_factor": profit_factor,
        "total_r": sum(values),
        "max_drawdown_r": max_drawdown,
        "stop_rate": sum(trade.status == "STOP" for trade in valid) / len(valid) * 100.0 if valid else 0.0,
        "target_rate": sum(trade.status in {"TP1", "TP2"} for trade in valid) / len(valid) * 100.0 if valid else 0.0,
        "time_rate": sum(trade.status == "TIME" for trade in valid) / len(valid) * 100.0 if valid else 0.0,
        "max_open": max_open_trades(valid),
        "avg_mae_r": float(np.mean([trade.mae_r for trade in valid if trade.mae_r is not None])) if valid else 0.0,
        "avg_mfe_r": float(np.mean([trade.mfe_r for trade in valid if trade.mfe_r is not None])) if valid else 0.0,
    }


def add_metrics_row(table: Table, label: str, subset: list[Trade]) -> None:
    data = metrics(subset)
    pf = "∞" if math.isinf(float(data["profit_factor"])) else f"{float(data['profit_factor']):.2f}"
    table.add_row(
        label,
        str(data["trades"]),
        str(data["censored"]),
        f"{float(data['win_rate']):.1f}%",
        f"{float(data['expectancy']):+.3f}R",
        f"{float(data['median_r']):+.3f}R",
        pf,
        f"{float(data['total_r']):+.2f}R",
        f"{float(data['max_drawdown_r']):.2f}R",
        str(data["max_open"]),
    )


def write_trade_results(path: Path, trades_by_threshold: dict[int, list[Trade]]) -> None:
    fields = (
        "min_score", "symbol", "side", "score", "signal_candle_utc", "entry_utc",
        "exit_utc", "status", "gross_r", "net_r", "mae_r", "mfe_r",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for threshold, trades in sorted(trades_by_threshold.items()):
            for trade in sorted(trades, key=lambda item: (item.entry_ms, item.symbol)):
                writer.writerow({
                    "min_score": threshold,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "score": trade.score,
                    "signal_candle_utc": datetime.fromtimestamp(trade.signal_ms / 1_000, timezone.utc).isoformat(),
                    "entry_utc": datetime.fromtimestamp(trade.entry_ms / 1_000, timezone.utc).isoformat(),
                    "exit_utc": datetime.fromtimestamp(trade.exit_ms / 1_000, timezone.utc).isoformat() if trade.exit_ms else "",
                    "status": trade.status,
                    "gross_r": "" if trade.gross_r is None else f"{trade.gross_r:.6f}",
                    "net_r": "" if trade.net_r is None else f"{trade.net_r:.6f}",
                    "mae_r": "" if trade.mae_r is None else f"{trade.mae_r:.6f}",
                    "mfe_r": "" if trade.mfe_r is None else f"{trade.mfe_r:.6f}",
                })


def print_report(
    trades_by_threshold: dict[int, list[Trade]],
    split_ms: int,
    start_ms: int,
    end_ms: int,
    exit_target: str,
    fee_bps: float,
    slippage_bps: float,
    history_count: int,
    requested_count: int,
) -> None:
    console.print(Panel.fit(
        "[bold cyan]CRYPTO SIGNAL SCANNER BACKTEST[/bold cyan]\n"
        f"Markets loaded: {history_count}/{requested_count}  |  Exit: {exit_target.upper()}  |  "
        f"Fee: {fee_bps:g} bps/side  |  Slippage: {slippage_bps:g} bps/side\n"
        f"Signals: {datetime.fromtimestamp(start_ms / 1000, timezone.utc):%Y-%m-%d %H:%M} UTC to "
        f"{datetime.fromtimestamp(end_ms / 1000, timezone.utc):%Y-%m-%d %H:%M} UTC (end exclusive)\n"
        f"Out-of-sample split: {datetime.fromtimestamp(split_ms / 1000, timezone.utc):%Y-%m-%d %H:%M} UTC",
        border_style="bright_magenta",
    ))
    table = Table(title="Results by minimum READY score", header_style="bold cyan")
    for column in ("Min score", "Trades", "Censored", "Win rate", "Avg net R", "Median R", "PF", "Total net R", "Max DD R", "Max open"):
        table.add_column(column, justify="right")
    for threshold, trades in sorted(trades_by_threshold.items()):
        add_metrics_row(table, str(threshold), trades)
        in_sample = [trade for trade in trades if trade.entry_ms < split_ms]
        out_sample = [trade for trade in trades if trade.entry_ms >= split_ms]
        add_metrics_row(table, f"{threshold} IS", in_sample)
        add_metrics_row(table, f"{threshold} OOS", out_sample)
    console.print(table)
    console.print(
        "[dim]R is measured against the planned stop distance. PF, average R, and total R include estimated fees and slippage. "
        "Same-candle stop/target collisions assume the stop happened first. Overlapping trades are included independently; Max DD R is a realized trade-level curve, not account-level drawdown.[/dim]"
    )
    console.print(
        "[yellow]Historical results do not validate future returns. Funding, spread, liquidation, partial fills, latency, and exchange-specific fees are not modeled. "
        "FRVP remains the scanner's OHLCV approximation.[/yellow]"
    )


async def run_backtest(args: argparse.Namespace) -> None:
    if args.end <= args.start:
        raise ValueError("--end must be after --start")
    if args.end > int(time.time() * 1_000):
        raise ValueError("--end cannot be in the future; choose a completed evaluation window")
    if args.max_hold_bars < 1 or args.max_hold_bars > 10_000:
        raise ValueError("--max-hold-bars must be between 1 and 10000")
    if (
        not math.isfinite(args.fee_bps)
        or not math.isfinite(args.slippage_bps)
        or args.fee_bps < 0
        or args.slippage_bps < 0
    ):
        raise ValueError("Fee and slippage must be non-negative")
    if args.concurrency < 1 or args.concurrency > 10:
        raise ValueError("--concurrency must be between 1 and 10")

    load_dotenv(Path(__file__).with_name(".env"))
    config = Config.from_env()
    if args.min_score is not None:
        if not 1 <= args.min_score <= 100:
            raise ValueError("--min-score must be between 1 and 100")
        config = replace(config, min_signal_score=args.min_score, telegram_token="", telegram_chat_id="")
    else:
        config = replace(config, telegram_token="", telegram_chat_id="")

    requested_symbols = tuple(
        item.strip().upper() for item in args.symbols.split(",") if item.strip()
    ) if args.symbols else SYMBOLS
    if not requested_symbols:
        raise ValueError("No symbols selected")
    unknown = [symbol for symbol in requested_symbols if symbol not in SYMBOLS]
    if unknown:
        raise ValueError(f"Symbols not in the scanner watchlist: {', '.join(unknown)}")

    if args.score_sweep:
        thresholds = sorted({int(part.strip()) for part in args.score_sweep.split(",") if part.strip()})
        if not thresholds or any(score < 1 or score > 100 for score in thresholds):
            raise ValueError("--score-sweep must contain scores from 1 to 100")
    else:
        thresholds = [config.min_signal_score]

    entry_duration = TIMEFRAME_MS[config.entry_timeframe]
    trend_duration = TIMEFRAME_MS[config.trend_timeframe]
    fetch_end_ms = min(args.end + args.max_hold_bars * entry_duration, int(time.time() * 1_000)) + entry_duration
    exchange_class = getattr(ccxt, config.exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unknown CCXT exchange id: {config.exchange_id}")
    exchange = exchange_class({
        "enableRateLimit": True,
        "timeout": 30_000,
        "options": {"defaultType": config.market_type},
    })
    histories: list[MarketHistory] = []
    try:
        markets = await exchange.load_markets()
        resolved, unavailable = resolve_markets(markets, requested_symbols, config.market_type)
        if unavailable:
            console.print(f"[yellow]Unavailable markets:[/yellow] {', '.join(unavailable)}")
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [
            load_market_history(
                exchange, requested, market_symbol, config, args.start, fetch_end_ms,
                args.cache_dir, semaphore,
            )
            for requested, market_symbol in resolved.items()
        ]
        loaded = await asyncio.gather(*tasks)
        histories = [history for history in loaded if history is not None]
    finally:
        await exchange.close()

    if not histories:
        raise RuntimeError("No usable historical market data was loaded")

    now_ms = int(time.time() * 1_000)
    closed_counts = {
        history.requested_symbol: int(np.searchsorted(
            history.entry_times, now_ms - entry_duration, side="right"
        ))
        for history in histories
    }
    signal_times = sorted({
        int(timestamp)
        for history in histories
        for timestamp in history.entry_times
        if args.start <= int(timestamp) < args.end
    })
    trades_by_threshold: dict[int, list[Trade]] = {score: [] for score in thresholds}
    console.print(f"[cyan]Replaying {len(signal_times):,} candle opens across {len(histories)} markets...[/cyan]")
    for timestamp in signal_times:
        candidates = signal_candidates_at(timestamp, histories, config, entry_duration, trend_duration)
        candidates.sort(key=lambda item: rank_key(item[0]), reverse=True)
        for threshold in thresholds:
            best = next((item for item in candidates if item[0].score >= threshold and not item[0].filter_failures), None)
            if best is None:
                continue
            signal, index, history = best
            trades_by_threshold[threshold].append(
                simulate_trade(
                    signal, index, history, closed_counts[history.requested_symbol],
                    entry_duration, args.max_hold_bars, args.exit_target,
                    args.fee_bps, args.slippage_bps,
                )
            )

    if args.split_date is not None:
        split_ms = args.split_date
    else:
        split_ms = args.start + int((args.end - args.start) * 0.70)
    if not args.start < split_ms < args.end:
        raise ValueError("The split date must fall strictly between --start and --end")
    output_path = args.output or Path(__file__).with_name(
        "backtest_results_"
        f"{datetime.fromtimestamp(args.start / 1_000, timezone.utc):%Y%m%d}_"
        f"{datetime.fromtimestamp(args.end / 1_000, timezone.utc):%Y%m%d}.csv"
    )
    write_trade_results(output_path, trades_by_threshold)
    print_report(
        trades_by_threshold, split_ms, args.start, args.end, args.exit_target,
        args.fee_bps, args.slippage_bps, len(histories), len(requested_symbols),
    )
    console.print(f"Trade details written to: [bold]{output_path.resolve()}[/bold]")


if __name__ == "__main__":
    parsed_args = build_arg_parser().parse_args()
    try:
        asyncio.run(run_backtest(parsed_args))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Backtest cancelled.[/bold yellow]")
