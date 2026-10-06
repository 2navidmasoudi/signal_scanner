"""Alert-only, multi-strategy crypto market scanner."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import ccxt.async_support as ccxt
import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


SYMBOLS = (
    "1000PEPEUSDT",
    "1000SHIBUSDT",
    "AAVEUSDT",
    "ADAUSDT",
    "ALGOUSDT",
    "APEUSDT",
    "APTUSDT",
    "ARBUSDT",
    "ASTERUSDT",
    "ATOMUSDT",
    "AVAXUSDT",
    "BANDUSDT",
    "BCHUSDT",
    "BMTUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "BTWUSDT",
    "CAKEUSDT",
    "DASHUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "ENAUSDT",
    "ETCUSDT",
    "ETHUSDT",
    "FETUSDT",
    "FILUSDT",
    "GRAMUSDT",
    "HBARUSDT",
    "HYPEUSDT",
    "ICPUSDT",
    "INJUSDT",
    "KAITOUSDT",
    "KSMUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "NEARUSDT",
    "NOTUSDT",
    "ONDOUSDT",
    "OPUSDT",
    "PUMPUSDT",
    "QNTUSDT",
    "SANDUSDT",
    "SOLUSDT",
    "SUIUSDT",
    "TRUMPUSDT",
    "TRXUSDT",
    "TUTUSDT",
    "UNIUSDT",
    "VETUSDT",
    "WLDUSDT",
    "XLMUSDT",
    "XRPUSDT",
    "ZECUSDT",
)

TIMEFRAME_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("market_scanner")
STATE_FILE = Path(__file__).with_name("scanner_state.json")


def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class Config:
    exchange_id: str
    market_type: str
    entry_timeframe: str
    trend_timeframe: str
    interval_seconds: int
    candle_limit: int
    concurrency: int
    frvp_lookback_bars: int
    frvp_rows: int
    frvp_value_area_percent: float
    min_signal_score: int
    min_volume_ratio: float
    max_long_rsi: float
    min_short_rsi: float
    breakout_volume_ratio: float
    max_entry_drift_atr: float
    telegram_token: str
    telegram_chat_id: str

    @classmethod
    def from_env(cls) -> "Config":
        market_type = os.getenv("MARKET_TYPE", "swap").strip().lower()
        if market_type not in {"swap", "spot", "future"}:
            raise ValueError("MARKET_TYPE must be swap, spot, or future")
        entry_tf = os.getenv("ENTRY_TIMEFRAME", "5m").strip()
        trend_tf = os.getenv("TREND_TIMEFRAME", "4h").strip()
        if entry_tf not in TIMEFRAME_MS or trend_tf not in TIMEFRAME_MS:
            raise ValueError("Supported timeframes: 1m, 5m, 15m, 1h, 4h")
        if TIMEFRAME_MS[trend_tf] <= TIMEFRAME_MS[entry_tf]:
            raise ValueError("TREND_TIMEFRAME must be longer than ENTRY_TIMEFRAME")
        return cls(
            exchange_id=os.getenv("EXCHANGE_ID", "binanceusdm").strip().lower(),
            market_type=market_type,
            entry_timeframe=entry_tf,
            trend_timeframe=trend_tf,
            interval_seconds=env_int("SCAN_INTERVAL_SECONDS", 30, 10, 86_400),
            candle_limit=env_int("OHLCV_LIMIT", 200, 80, 1_000),
            concurrency=env_int("MAX_CONCURRENCY", 8, 1, 30),
            frvp_lookback_bars=env_int("FRVP_LOOKBACK_BARS", 96, 20, 500),
            frvp_rows=env_int("FRVP_ROWS", 48, 12, 128),
            frvp_value_area_percent=env_float("FRVP_VALUE_AREA_PERCENT", 0.70, 0.50, 0.90),
            min_signal_score=env_int("MIN_SIGNAL_SCORE", 65, 1, 100),
            min_volume_ratio=env_float("MIN_VOLUME_RATIO", 1.0, 0.1, 20.0),
            max_long_rsi=env_float("MAX_LONG_RSI", 70.0, 50.0, 100.0),
            min_short_rsi=env_float("MIN_SHORT_RSI", 30.0, 0.0, 50.0),
            breakout_volume_ratio=env_float("BREAKOUT_VOLUME_RATIO", 1.5, 1.0, 20.0),
            max_entry_drift_atr=env_float("MAX_ENTRY_DRIFT_ATR", 0.25, 0.01, 5.0),
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        )


@dataclass(frozen=True)
class FixedRangeVolumeProfile:
    poc: float
    vah: float
    val: float
    profile_high: float
    profile_low: float
    row_width: float
    total_volume: float
    bars: int


@dataclass(frozen=True)
class Signal:
    requested_symbol: str
    market_symbol: str
    candle_timestamp: int
    side: str
    score: int
    price: float
    stop: float
    target_1: float
    target_2: float
    rsi: float
    adx: float
    volume_ratio: float
    atr_percent: float
    entry_drift_atr: float
    frvp_profile: FixedRangeVolumeProfile | None
    frvp_position: str
    price_action_context: tuple[str, ...]
    price_action_confirmations: tuple[str, ...]
    reasons: tuple[str, ...]
    filter_failures: tuple[str, ...]


@dataclass(frozen=True)
class PriceActionPattern:
    side: str
    label: str
    confirmed: bool = True


@dataclass(frozen=True)
class StructureBreak:
    index: int
    side: str
    kind: str
    level: float


@dataclass(frozen=True)
class ScanResult:
    signals: tuple[Signal, ...]
    failed: tuple[tuple[str, str], ...]
    unavailable: tuple[str, ...]
    scanned_count: int


def ema(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    if len(values) < period:
        return result
    result[period - 1] = float(np.mean(values[:period]))
    alpha = 2.0 / (period + 1.0)
    for index in range(period, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def rsi(values: np.ndarray, period: int = 14) -> float:
    if len(values) <= period:
        return float("nan")
    changes = np.diff(values)
    gains = np.maximum(changes, 0.0)
    losses = np.maximum(-changes, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for index in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + float(gains[index])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses[index])) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def atr_and_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> tuple[float, float]:
    if len(close) < period + 2:
        return float("nan"), float("nan")
    previous_close = close[:-1]
    true_range = np.maximum.reduce((high[1:] - low[1:], np.abs(high[1:] - previous_close), np.abs(low[1:] - previous_close)))
    up_move = np.diff(high)
    down_move = -np.diff(low)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    def wilder_series(values: np.ndarray) -> np.ndarray:
        result = np.full(values.shape, np.nan, dtype=float)
        if len(values) < period:
            return result
        result[period - 1] = float(np.sum(values[:period]))
        for index in range(period, len(values)):
            result[index] = result[index - 1] - result[index - 1] / period + values[index]
        return result

    tr_smoothed = wilder_series(true_range)
    plus_smoothed = wilder_series(plus_dm)
    minus_smoothed = wilder_series(minus_dm)
    atr_value = float(tr_smoothed[-1] / period)
    valid = np.isfinite(tr_smoothed) & (tr_smoothed > 0)
    plus_di = np.zeros_like(tr_smoothed)
    minus_di = np.zeros_like(tr_smoothed)
    plus_di[valid] = 100.0 * plus_smoothed[valid] / tr_smoothed[valid]
    minus_di[valid] = 100.0 * minus_smoothed[valid] / tr_smoothed[valid]
    denominator = plus_di + minus_di
    dx = np.full(tr_smoothed.shape, np.nan, dtype=float)
    valid_dx = valid & (denominator > 0)
    dx[valid_dx] = 100.0 * np.abs(plus_di[valid_dx] - minus_di[valid_dx]) / denominator[valid_dx]
    valid_indexes = np.flatnonzero(np.isfinite(dx))
    if len(valid_indexes) < period:
        return atr_value, float("nan")
    first_adx_index = valid_indexes[period - 1]
    adx_series = np.full(dx.shape, np.nan, dtype=float)
    initial_dx_indexes = valid_indexes[:period]
    adx_series[first_adx_index] = float(np.mean(dx[initial_dx_indexes]))
    for index in range(first_adx_index + 1, len(dx)):
        if np.isfinite(dx[index]):
            adx_series[index] = ((adx_series[index - 1] * (period - 1)) + dx[index]) / period
        elif np.isfinite(adx_series[index - 1]):
            adx_series[index] = adx_series[index - 1]
    return atr_value, float(adx_series[-1])


def confirmed_swings(high: np.ndarray, low: np.ndarray, radius: int = 2) -> tuple[list[int], list[int]]:
    swing_highs: list[int] = []
    swing_lows: list[int] = []
    for index in range(radius, len(high) - radius):
        if high[index] > np.max(high[index - radius : index]) and high[index] > np.max(high[index + 1 : index + radius + 1]):
            swing_highs.append(index)
        if low[index] < np.min(low[index - radius : index]) and low[index] < np.min(low[index + 1 : index + radius + 1]):
            swing_lows.append(index)
    return swing_highs, swing_lows


def calculate_fixed_range_volume_profile(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback_bars: int,
    row_count: int,
    value_area_percent: float,
) -> FixedRangeVolumeProfile | None:
    """Approximate a rolling fixed-range profile from OHLCV candles."""
    count = min(len(close), lookback_bars)
    if count < 20 or row_count < 2 or not 0 < value_area_percent <= 1:
        return None
    range_highs = high[-count:]
    range_lows = low[-count:]
    range_closes = close[-count:]
    range_volumes = volume[-count:]
    profile_low = float(np.min(range_lows))
    profile_high = float(np.max(range_highs))
    if not all(math.isfinite(value) for value in (profile_low, profile_high)) or profile_high <= profile_low:
        return None

    edges = np.linspace(profile_low, profile_high, row_count + 1)
    row_volumes = np.zeros(row_count, dtype=float)
    for bar_low, bar_high, bar_close, bar_volume in zip(range_lows, range_highs, range_closes, range_volumes):
        if bar_volume <= 0:
            continue
        overlap = np.maximum(0.0, np.minimum(bar_high, edges[1:]) - np.maximum(bar_low, edges[:-1]))
        covered = float(np.sum(overlap))
        if covered > 0:
            row_volumes += bar_volume * overlap / covered
        else:
            row = int(np.clip(np.searchsorted(edges, bar_close, side="right") - 1, 0, row_count - 1))
            row_volumes[row] += bar_volume

    total_volume = float(np.sum(row_volumes))
    if not math.isfinite(total_volume) or total_volume <= 0:
        return None

    poc_index = int(np.argmax(row_volumes))
    low_index = high_index = poc_index
    included_volume = float(row_volumes[poc_index])
    target_volume = total_volume * value_area_percent
    while included_volume < target_volume:
        below = low_index - 1 if low_index > 0 else None
        above = high_index + 1 if high_index + 1 < row_count else None
        if below is None and above is None:
            break
        below_volume = float(row_volumes[below]) if below is not None else -1.0
        above_volume = float(row_volumes[above]) if above is not None else -1.0
        if above_volume > below_volume:
            candidate, candidate_volume = above, above_volume
        elif below_volume > above_volume:
            candidate, candidate_volume = below, below_volume
        else:
            # Resolve equal adjacent rows toward the POC; choose above if equally distant.
            below_distance = abs(poc_index - below) if below is not None else math.inf
            above_distance = abs(above - poc_index) if above is not None else math.inf
            if above_distance <= below_distance:
                candidate, candidate_volume = above, above_volume
            else:
                candidate, candidate_volume = below, below_volume
        if candidate is None or candidate_volume <= 0:
            break
        if included_volume + candidate_volume > target_volume:
            break
        included_volume += candidate_volume
        if candidate < poc_index:
            low_index = candidate
        else:
            high_index = candidate

    row_width = float(edges[1] - edges[0])
    return FixedRangeVolumeProfile(
        poc=float((edges[poc_index] + edges[poc_index + 1]) / 2.0),
        vah=float(edges[high_index + 1]),
        val=float(edges[low_index]),
        profile_high=profile_high,
        profile_low=profile_low,
        row_width=row_width,
        total_volume=total_volume,
        bars=count,
    )


def classify_frvp_position(
    profile: FixedRangeVolumeProfile | None,
    close_price: float,
    atr_value: float,
    volume_ratio: float,
) -> tuple[str, PriceActionPattern | None]:
    if profile is None:
        return "Unavailable", None
    if close_price > profile.vah:
        return "Above VAH", PriceActionPattern("LONG", "FRVP LONG above VAH", volume_ratio >= 1.2)
    if close_price < profile.val:
        return "Below VAL", PriceActionPattern("SHORT", "FRVP SHORT below VAL", volume_ratio >= 1.2)

    poc_tolerance = max(profile.row_width, 0.10 * atr_value)
    if close_price > profile.poc + poc_tolerance:
        return "Inside VA above POC", PriceActionPattern("LONG", "FRVP LONG above POC", False)
    if close_price < profile.poc - poc_tolerance:
        return "Inside VA below POC", PriceActionPattern("SHORT", "FRVP SHORT below POC", False)
    return "At POC", None


def detect_recent_fvgs(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_value: float,
    lookback: int = 24,
) -> tuple[PriceActionPattern, ...]:
    """Find the newest fresh three-candle imbalance for each direction."""
    count = len(close)
    if count < 4 or atr_value <= 0:
        return ()
    last = count - 1
    patterns: list[PriceActionPattern] = []
    min_gap = 0.05 * atr_value

    for side in ("LONG", "SHORT"):
        first = max(2, count - lookback)
        for index in range(last, first - 1, -1):
            if side == "LONG":
                zone_low = float(high[index - 2])
                zone_high = float(low[index])
            else:
                zone_low = float(high[index])
                zone_high = float(low[index - 2])
            if zone_high - zone_low < min_gap or zone_high <= zone_low:
                continue

            later = range(index + 1, count)
            if side == "LONG":
                fully_filled = any(low[item] <= zone_low for item in later)
            else:
                fully_filled = any(high[item] >= zone_high for item in later)
            if fully_filled:
                continue

            touched_before = any(
                low[item] <= zone_high and high[item] >= zone_low
                for item in range(index + 1, last)
            )
            if touched_before:
                continue
            touched_now = index < last and low[last] <= zone_high and high[last] >= zone_low
            midpoint = (zone_low + zone_high) / 2.0
            rejected = (
                close[last] > close[last - 1] and close[last] >= midpoint
                if side == "LONG"
                else close[last] < close[last - 1] and close[last] <= midpoint
            )
            confirmed = bool(touched_now and rejected)
            status = "first retest" if touched_now and rejected else "retest pending" if touched_now else "fresh"
            patterns.append(PriceActionPattern(side, f"FVG {side} {status}", confirmed))
            break
    return tuple(patterns)


def detect_structure_breaks(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_value: float,
    lookback: int = 30,
) -> tuple[StructureBreak, ...]:
    """Detect recent close-throughs of confirmed swings and classify BOS/CHoCH mechanically."""
    count = len(close)
    if count < 15 or atr_value <= 0:
        return ()
    swing_highs, swing_lows = confirmed_swings(high, low)
    radius = 2
    breaks: list[StructureBreak] = []

    for index in range(max(1, count - lookback), count):
        prior_highs = [point for point in swing_highs if point + radius < index]
        prior_lows = [point for point in swing_lows if point + radius < index]
        if not prior_highs or not prior_lows:
            continue

        prior_trend: str | None = None
        if len(prior_highs) >= 2 and len(prior_lows) >= 2:
            higher_highs = high[prior_highs[-1]] > high[prior_highs[-2]]
            higher_lows = low[prior_lows[-1]] > low[prior_lows[-2]]
            lower_highs = high[prior_highs[-1]] < high[prior_highs[-2]]
            lower_lows = low[prior_lows[-1]] < low[prior_lows[-2]]
            if higher_highs and higher_lows:
                prior_trend = "LONG"
            elif lower_highs and lower_lows:
                prior_trend = "SHORT"

        high_level = float(high[prior_highs[-1]])
        low_level = float(low[prior_lows[-1]])
        bullish_break = (
            close[index - 1] <= high_level
            and close[index] > high_level + 0.02 * atr_value
            and close[index] > open_[index]
            and close[index] - open_[index] >= 0.12 * atr_value
        )
        bearish_break = (
            close[index - 1] >= low_level
            and close[index] < low_level - 0.02 * atr_value
            and close[index] < open_[index]
            and open_[index] - close[index] >= 0.12 * atr_value
        )
        if bullish_break:
            kind = "CHoCH" if prior_trend == "SHORT" else "BOS"
            breaks.append(StructureBreak(index, "LONG", kind, high_level))
        if bearish_break:
            kind = "CHoCH" if prior_trend == "LONG" else "BOS"
            breaks.append(StructureBreak(index, "SHORT", kind, low_level))
    return tuple(breaks)


def detect_order_block_retests(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_value: float,
    structure_breaks: tuple[StructureBreak, ...],
    lookback: int = 30,
) -> tuple[PriceActionPattern, ...]:
    """Find the last opposite candle before a recent break and flag a fresh first-return test."""
    count = len(close)
    if count < 5 or atr_value <= 0:
        return ()
    last = count - 1
    patterns: list[PriceActionPattern] = []

    for side in ("LONG", "SHORT"):
        for event in reversed(structure_breaks):
            if event.side != side or event.index < count - lookback:
                continue
            first = max(0, event.index - 10)
            opposite_candles = [
                index
                for index in range(first, event.index)
                if (close[index] < open_[index] if side == "LONG" else close[index] > open_[index])
            ]
            if not opposite_candles:
                continue
            block_index = opposite_candles[-1]
            zone_low, zone_high = float(low[block_index]), float(high[block_index])
            if zone_high <= zone_low or zone_high - zone_low > 1.5 * atr_value:
                continue

            after_break = range(event.index + 1, count)
            if side == "LONG":
                invalidated = any(close[index] < zone_low for index in after_break)
            else:
                invalidated = any(close[index] > zone_high for index in after_break)
            if invalidated:
                continue
            touched_before = any(
                low[index] <= zone_high and high[index] >= zone_low
                for index in range(event.index + 1, last)
            )
            if touched_before:
                continue
            touched_now = low[last] <= zone_high and high[last] >= zone_low
            midpoint = (zone_low + zone_high) / 2.0
            rejected = (
                close[last] > open_[last] and close[last] >= midpoint
                if side == "LONG"
                else close[last] < open_[last] and close[last] <= midpoint
            )
            if touched_now:
                status = "first retest" if rejected else "retest pending"
            else:
                status = "fresh zone"
            patterns.append(PriceActionPattern(side, f"OB {side} {status}", bool(touched_now and rejected)))
            break
    return tuple(patterns)


def detect_ict_first_retest(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_value: float,
) -> tuple[str | None, str | None]:
    """Detect a mechanical sweep -> displacement/MSS -> FVG -> first retest sequence."""
    count = len(close)
    if count < 20 or atr_value <= 0:
        return None, None
    swing_highs, swing_lows = confirmed_swings(high, low)
    last = count - 1
    if last < 3:
        return None, None

    for fvg_index in range(last - 1, max(2, last - 7) - 1, -1):
        patterns: list[tuple[str, float, float]] = []
        bull_gap = float(low[fvg_index] - high[fvg_index - 2])
        if bull_gap >= 0.1 * atr_value:
            patterns.append(("LONG", float(high[fvg_index - 2]), float(low[fvg_index])))
        bear_gap = float(low[fvg_index - 2] - high[fvg_index])
        if bear_gap >= 0.1 * atr_value:
            patterns.append(("SHORT", float(high[fvg_index]), float(low[fvg_index - 2])))

        for side, zone_low, zone_high in patterns:
            if last - fvg_index > 6:
                continue
            retest_touches = low[last] <= zone_high and high[last] >= zone_low
            bullish_rejection = close[last] > open_[last] and close[last] > zone_low
            bearish_rejection = close[last] < open_[last] and close[last] < zone_high
            if not retest_touches or (side == "LONG" and not bullish_rejection) or (side == "SHORT" and not bearish_rejection):
                continue
            touched_earlier = any(
                low[index] <= zone_high and high[index] >= zone_low
                for index in range(fvg_index + 1, last)
            )
            if touched_earlier:
                continue

            first_sweep = max(2, fvg_index - 10)
            for sweep_index in range(first_sweep, fvg_index):
                if side == "LONG":
                    prior_levels = [index for index in swing_lows if index + 2 < sweep_index]
                    prior_structure = [index for index in swing_highs if index + 2 < sweep_index]
                    if not prior_levels or not prior_structure:
                        continue
                    liquidity = float(low[prior_levels[-1]])
                    structure = float(high[prior_structure[-1]])
                    swept = low[sweep_index] < liquidity and close[sweep_index] > liquidity
                else:
                    prior_levels = [index for index in swing_highs if index + 2 < sweep_index]
                    prior_structure = [index for index in swing_lows if index + 2 < sweep_index]
                    if not prior_levels or not prior_structure:
                        continue
                    liquidity = float(high[prior_levels[-1]])
                    structure = float(low[prior_structure[-1]])
                    swept = high[sweep_index] > liquidity and close[sweep_index] < liquidity
                if not swept:
                    continue

                displacement_found = False
                for index in range(sweep_index + 1, min(fvg_index, sweep_index + 5) + 1):
                    candle_range = float(high[index] - low[index])
                    body = float(close[index] - open_[index])
                    if candle_range <= 0:
                        continue
                    if side == "LONG":
                        displacement_found = close[index] > structure and body >= 0.6 * atr_value and body / candle_range >= 0.6
                    else:
                        displacement_found = close[index] < structure and body <= -0.6 * atr_value and -body / candle_range >= 0.6
                    if displacement_found:
                        break
                if displacement_found:
                    label = "bullish" if side == "LONG" else "bearish"
                    return side, f"ICT {side}: {label} liquidity sweep → displacement/MSS → FVG first retest"
    return None, None


def detect_rtm_first_return(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_value: float,
) -> tuple[str | None, str | None]:
    """Detect a compact base, impulsive departure, and fresh first-return rejection."""
    count = len(close)
    if count < 20 or atr_value <= 0:
        return None, None
    last = count - 1
    first_base_end = max(6, count - 36)
    for base_end in range(last - 2, first_base_end - 1, -1):
        for base_size in (1, 2, 3):
            base_start = base_end - base_size + 1
            if base_start < 4:
                continue
            zone_low = float(np.min(low[base_start : base_end + 1]))
            zone_high = float(np.max(high[base_start : base_end + 1]))
            if zone_high - zone_low > 1.25 * atr_value:
                continue
            approach = float(close[base_start] - close[base_start - 3])
            if abs(approach) < 0.5 * atr_value:
                continue

            departure_limit = min(base_end + 3, last - 1)
            for departure_end in range(base_end + 1, departure_limit + 1):
                move = float(close[departure_end] - close[base_end])
                if move >= 1.2 * atr_value and close[departure_end] > zone_high:
                    side = "LONG"
                elif move <= -1.2 * atr_value and close[departure_end] < zone_low:
                    side = "SHORT"
                else:
                    continue

                touched_before_return = any(
                    low[index] <= zone_high and high[index] >= zone_low
                    for index in range(departure_end + 1, last)
                )
                if touched_before_return:
                    continue
                current_touches = low[last] <= zone_high and high[last] >= zone_low
                if not current_touches:
                    continue
                if side == "LONG" and close[last] > open_[last] and close[last] > zone_high:
                    return side, "RTM LONG proxy: fresh demand base → impulsive departure → first-return rejection"
                if side == "SHORT" and close[last] < open_[last] and close[last] < zone_low:
                    return side, "RTM SHORT proxy: fresh supply base → impulsive departure → first-return rejection"
    return None, None


def closed_candles(rows: list[list[float]], timeframe: str, now_ms: int) -> np.ndarray:
    duration = TIMEFRAME_MS[timeframe]
    normalized: list[list[float]] = []
    previous_timestamp: int | None = None
    for row in rows:
        if len(row) < 6:
            raise ValueError("Malformed OHLCV row")
        try:
            candle = [float(value) for value in row[:6]]
        except (TypeError, ValueError) as exc:
            raise ValueError("Non-numeric OHLCV value") from exc
        if not all(math.isfinite(value) for value in candle):
            raise ValueError("Non-finite OHLCV value")
        timestamp = int(candle[0])
        if candle[0] != timestamp or timestamp < 0:
            raise ValueError("Invalid OHLCV timestamp")
        open_price, high, low, close, volume = candle[1:]
        if (
            min(open_price, high, low, close) <= 0
            or volume < 0
            or high < max(open_price, close)
            or low > min(open_price, close)
            or high < low
        ):
            raise ValueError("Invalid OHLCV price or volume")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("OHLCV timestamps are not strictly increasing")
        normalized.append(candle)
        previous_timestamp = timestamp
    closed = [row for row in normalized if int(row[0]) + duration <= now_ms]
    return np.asarray(closed, dtype=float).reshape((-1, 6))


def require_fresh_candles(rows: np.ndarray, timeframe: str, now_ms: int, grace_ms: int) -> None:
    if rows.ndim != 2 or len(rows) == 0:
        raise ValueError(f"No closed {timeframe} candles")
    latest_close_ms = int(rows[-1, 0]) + TIMEFRAME_MS[timeframe]
    if now_ms - latest_close_ms > TIMEFRAME_MS[timeframe] + grace_ms:
        raise ValueError(f"Stale {timeframe} candles")


def resolve_markets(markets: dict[str, dict[str, Any]], requested: tuple[str, ...], market_type: str) -> tuple[dict[str, str], tuple[str, ...]]:
    resolved: dict[str, str] = {}
    unavailable: list[str] = []
    for requested_symbol in requested:
        candidates = []
        for market in markets.values():
            market_id = str(market.get("id", "")).upper()
            base_quote = f"{market.get('base', '')}{market.get('quote', '')}".upper()
            if requested_symbol.upper() not in {market_id, base_quote}:
                continue
            if market.get("quote") != "USDT" or market.get("active") is False:
                continue
            if market_type == "swap" and not market.get("swap"):
                continue
            if market_type == "future" and not market.get("future"):
                continue
            if market_type == "spot" and not market.get("spot"):
                continue
            candidates.append(market)
        if not candidates:
            unavailable.append(requested_symbol)
            continue
        candidates.sort(key=lambda item: (not item.get("active", True), item.get("symbol", "")))
        resolved[requested_symbol] = str(candidates[0]["symbol"])
    return resolved, tuple(unavailable)


def compute_signal(
    requested_symbol: str,
    market_symbol: str,
    entry_rows: np.ndarray,
    trend_rows: np.ndarray,
    config: Config,
    current_price: float,
) -> tuple[Signal | None, str | None]:
    if entry_rows.ndim != 2 or trend_rows.ndim != 2 or len(entry_rows) < 60 or len(trend_rows) < 60:
        return None, "Insufficient closed candles"

    entry_open, entry_high, entry_low, entry_close, entry_volume = (
        entry_rows[:, index] for index in (1, 2, 3, 4, 5)
    )
    trend_close = trend_rows[:, 4]
    entry_fast = ema(entry_close, 20)
    entry_slow = ema(entry_close, 50)
    trend_fast = ema(trend_close, 20)
    trend_slow = ema(trend_close, 50)
    if not all(np.isfinite(values[-1]) for values in (entry_fast, entry_slow, trend_fast, trend_slow)):
        return None, "EMA calculation unavailable"

    price = float(entry_close[-1])
    if price <= 0:
        return None, "Invalid closing price"
    atr_value, adx_value = atr_and_adx(entry_high, entry_low, entry_close)
    rsi_value = rsi(entry_close)
    if not all(math.isfinite(value) for value in (atr_value, adx_value, rsi_value)) or atr_value <= 0:
        return None, "ATR, ADX, or RSI calculation unavailable"
    frvp_profile = calculate_fixed_range_volume_profile(
        entry_high,
        entry_low,
        entry_close,
        entry_volume,
        config.frvp_lookback_bars,
        config.frvp_rows,
        config.frvp_value_area_percent,
    )
    ict_side, ict_reason = detect_ict_first_retest(entry_open, entry_high, entry_low, entry_close, atr_value)
    rtm_side, rtm_reason = detect_rtm_first_return(entry_open, entry_high, entry_low, entry_close, atr_value)
    structure_breaks = detect_structure_breaks(entry_open, entry_high, entry_low, entry_close, atr_value)
    price_action_patterns: list[PriceActionPattern] = []
    if ict_side and ict_reason:
        price_action_patterns.append(PriceActionPattern(ict_side, ict_reason))
    if rtm_side and rtm_reason:
        price_action_patterns.append(PriceActionPattern(rtm_side, rtm_reason))
    price_action_patterns.extend(detect_recent_fvgs(entry_high, entry_low, entry_close, atr_value))
    recent_breaks = [event for event in structure_breaks if event.index >= len(entry_close) - 4]
    if recent_breaks:
        event = recent_breaks[-1]
        price_action_patterns.append(
            PriceActionPattern(event.side, f"{event.kind} {event.side}: close broke confirmed swing")
        )
    price_action_patterns.extend(
        detect_order_block_retests(
            entry_open, entry_high, entry_low, entry_close, atr_value, structure_breaks
        )
    )

    macd_line = ema(entry_close, 12) - ema(entry_close, 26)
    signal_line = ema(macd_line[np.isfinite(macd_line)], 9)
    macd_hist = float(macd_line[np.isfinite(macd_line)][-1] - signal_line[-1]) if len(signal_line) else 0.0
    previous_hist = float(macd_line[np.isfinite(macd_line)][-2] - signal_line[-2]) if len(signal_line) > 1 else macd_hist

    lookback = min(20, len(entry_close) - 1)
    prior_high = float(np.max(entry_high[-lookback - 1 : -1]))
    prior_low = float(np.min(entry_low[-lookback - 1 : -1]))
    average_volume = float(np.mean(entry_volume[-21:-1]))
    volume_ratio = float(entry_volume[-1] / average_volume) if average_volume > 0 else 0.0
    band_window = entry_close[-20:]
    band_mid = float(np.mean(band_window))
    band_width = float(np.std(band_window, ddof=0) * 2.0)
    band_upper, band_lower = band_mid + band_width, band_mid - band_width
    atr_percent = atr_value / price * 100.0
    if not math.isfinite(current_price) or current_price <= 0:
        current_price = price
    entry_drift_atr = abs(current_price - price) / atr_value
    frvp_position, frvp_pattern = classify_frvp_position(frvp_profile, price, atr_value, volume_ratio)
    if frvp_pattern:
        price_action_patterns.append(frvp_pattern)

    long_score = 0
    short_score = 0
    long_reasons: list[str] = []
    short_reasons: list[str] = []

    htf_up = trend_fast[-1] > trend_slow[-1]
    htf_down = trend_fast[-1] < trend_slow[-1]
    if htf_up:
        long_score += 20
        long_reasons.append(f"{config.trend_timeframe} trend up")
    elif htf_down:
        short_score += 20
        short_reasons.append(f"{config.trend_timeframe} trend down")
    if entry_fast[-1] > entry_slow[-1]:
        long_score += 15
        long_reasons.append(f"{config.entry_timeframe} trend up")
    elif entry_fast[-1] < entry_slow[-1]:
        short_score += 15
        short_reasons.append(f"{config.entry_timeframe} trend down")

    if price > entry_fast[-1]:
        long_score += 5
        long_reasons.append("Price above EMA20")
    else:
        short_score += 5
        short_reasons.append("Price below EMA20")
    if price > entry_slow[-1]:
        long_score += 5
        long_reasons.append("Price above EMA50")
    else:
        short_score += 5
        short_reasons.append("Price below EMA50")

    if macd_hist > 0:
        long_score += 12
        long_reasons.append("MACD positive")
    elif macd_hist < 0:
        short_score += 12
        short_reasons.append("MACD negative")
    if (macd_hist > previous_hist and macd_hist > 0) or (macd_hist < previous_hist and macd_hist < 0):
        if macd_hist > 0:
            long_reasons.append("MACD momentum rising")
        elif macd_hist < 0:
            short_reasons.append("MACD momentum falling")

    if 51 <= rsi_value <= 68:
        long_score += 8
        long_reasons.append(f"RSI supportive ({rsi_value:.0f})")
    elif 68 < rsi_value <= 74:
        long_score += 4
        long_reasons.append(f"RSI strong ({rsi_value:.0f})")
    elif 46 <= rsi_value < 51:
        long_score += 3
    if 32 <= rsi_value < 49:
        short_score += 8
        short_reasons.append(f"RSI supportive ({rsi_value:.0f})")
    elif 26 <= rsi_value < 32:
        short_score += 4
        short_reasons.append(f"RSI weak ({rsi_value:.0f})")
    elif 49 <= rsi_value <= 54:
        short_score += 3

    breakout_side = ""
    if price > prior_high:
        breakout_side = "LONG"
        long_score += 15 if price - prior_high >= atr_value * 0.1 else 8
        long_reasons.append("20-bar high breakout")
    elif price < prior_low:
        breakout_side = "SHORT"
        short_score += 15 if prior_low - price >= atr_value * 0.1 else 8
        short_reasons.append("20-bar low breakdown")

    volume_side = breakout_side or ("LONG" if entry_fast[-1] >= entry_slow[-1] else "SHORT")
    volume_points = 8 if volume_ratio >= 1.5 else 6 if volume_ratio >= 1.2 else 3 if volume_ratio >= 1.0 else 0
    if volume_points:
        if volume_side == "LONG":
            long_score += volume_points
            long_reasons.append(f"Volume {volume_ratio:.1f}x")
        else:
            short_score += volume_points
            short_reasons.append(f"Volume {volume_ratio:.1f}x")

    trend_side = "LONG" if entry_fast[-1] >= entry_slow[-1] else "SHORT"
    if adx_value >= 25:
        if trend_side == "LONG":
            long_score += 7
            long_reasons.append(f"Trend strength ADX {adx_value:.0f}")
        else:
            short_score += 7
            short_reasons.append(f"Trend strength ADX {adx_value:.0f}")
    elif adx_value >= 18:
        if trend_side == "LONG":
            long_score += 5
        else:
            short_score += 5

    if adx_value < 18 and price < band_lower and rsi_value < 35 and not htf_down:
        long_score += 5
        long_reasons.append("Range oversold reversal")
    elif adx_value < 18 and price > band_upper and rsi_value > 65 and not htf_up:
        short_score += 5
        short_reasons.append("Range overbought reversal")

    if atr_percent > 3.0:
        long_score -= 8
        short_score -= 8
    elif atr_percent < 0.04:
        long_score -= 8
        short_score -= 8

    if htf_up and entry_fast[-1] < entry_slow[-1]:
        short_score -= 8
    elif htf_down and entry_fast[-1] > entry_slow[-1]:
        long_score -= 8

    if long_score == short_score:
        return None, None
    side = "LONG" if long_score > short_score else "SHORT"
    score = max(0, min(100, long_score if side == "LONG" else short_score))
    aligned_patterns = tuple(
        pattern.label for pattern in price_action_patterns if pattern.side == side and pattern.confirmed
    )
    confirmed_breakout = breakout_side == side and volume_ratio >= config.breakout_volume_ratio
    filter_failures: list[str] = []
    if volume_ratio < config.min_volume_ratio:
        filter_failures.append(f"Volume below {config.min_volume_ratio:.2f}x ({volume_ratio:.2f}x)")
    if entry_drift_atr > config.max_entry_drift_atr:
        filter_failures.append(
            f"Live price drift {entry_drift_atr:.2f} ATR exceeds {config.max_entry_drift_atr:.2f} ATR"
        )
    if side == "LONG" and rsi_value > config.max_long_rsi and not confirmed_breakout:
        filter_failures.append(f"RSI extended above {config.max_long_rsi:.0f}")
    elif side == "SHORT" and rsi_value < config.min_short_rsi and not confirmed_breakout:
        filter_failures.append(f"RSI extended below {config.min_short_rsi:.0f}")
    if side == "LONG":
        stop = current_price - 1.5 * atr_value
        target_1 = current_price + 1.5 * atr_value
        target_2 = current_price + 3.0 * atr_value
        reasons = tuple(long_reasons + list(aligned_patterns))
    else:
        stop = current_price + 1.5 * atr_value
        target_1 = current_price - 1.5 * atr_value
        target_2 = current_price - 3.0 * atr_value
        reasons = tuple(short_reasons + list(aligned_patterns))
    if not all(math.isfinite(level) and level > 0 for level in (stop, target_1, target_2)):
        filter_failures.append("ATR stop or target is invalid or non-positive")

    signal = Signal(
        requested_symbol=requested_symbol,
        market_symbol=market_symbol,
        candle_timestamp=int(entry_rows[-1, 0]),
        side=side,
        score=score,
        price=current_price,
        stop=stop,
        target_1=target_1,
        target_2=target_2,
        rsi=rsi_value,
        adx=adx_value,
        volume_ratio=volume_ratio,
        atr_percent=atr_percent,
        entry_drift_atr=entry_drift_atr,
        frvp_profile=frvp_profile,
        frvp_position=frvp_position,
        price_action_context=tuple(pattern.label for pattern in price_action_patterns),
        price_action_confirmations=aligned_patterns,
        reasons=reasons,
        filter_failures=tuple(filter_failures),
    )
    return signal, None


async def fetch_candles(exchange: Any, symbol: str, timeframe: str, limit: int, semaphore: asyncio.Semaphore) -> list[list[float]]:
    async with semaphore:
        return await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def latest_candle_price(rows: list[list[float]], timeframe: str, now_ms: int) -> float:
    if not rows or len(rows[-1]) < 6:
        return float("nan")
    timestamp = int(rows[-1][0])
    duration = TIMEFRAME_MS[timeframe]
    if not timestamp <= now_ms < timestamp + duration:
        return float("nan")
    candidate = float(rows[-1][4])
    volume = float(rows[-1][5])
    return candidate if math.isfinite(candidate) and candidate > 0 and volume > 0 else float("nan")


async def inspect_symbol(
    exchange: Any,
    requested_symbol: str,
    market_symbol: str,
    config: Config,
    semaphore: asyncio.Semaphore,
) -> tuple[Signal | None, str | None]:
    try:
        entry_data, trend_data = await asyncio.gather(
            fetch_candles(exchange, market_symbol, config.entry_timeframe, config.candle_limit, semaphore),
            fetch_candles(exchange, market_symbol, config.trend_timeframe, config.candle_limit, semaphore),
        )
        now_ms = exchange.milliseconds()
        entry_rows = closed_candles(entry_data, config.entry_timeframe, now_ms)
        trend_rows = closed_candles(trend_data, config.trend_timeframe, now_ms)
        grace_ms = min(
            max(30_000, config.interval_seconds * 2_000),
            TIMEFRAME_MS[config.entry_timeframe] // 4,
            TIMEFRAME_MS[config.trend_timeframe] // 4,
        )
        require_fresh_candles(entry_rows, config.entry_timeframe, now_ms, grace_ms)
        require_fresh_candles(trend_rows, config.trend_timeframe, now_ms, grace_ms)
        current_price = latest_candle_price(entry_data, config.entry_timeframe, now_ms)
        if not math.isfinite(current_price):
            return None, "Current in-progress entry candle unavailable"
        return compute_signal(requested_symbol, market_symbol, entry_rows, trend_rows, config, current_price)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:120]}"


async def scan(exchange: Any, config: Config, markets: dict[str, dict[str, Any]]) -> ScanResult:
    resolved, unavailable = resolve_markets(markets, SYMBOLS, config.market_type)
    semaphore = asyncio.Semaphore(config.concurrency)
    outcomes = await asyncio.gather(
        *(inspect_symbol(exchange, requested, symbol, config, semaphore) for requested, symbol in resolved.items())
    )
    signals: list[Signal] = []
    failed: list[tuple[str, str]] = []
    for requested, outcome in zip(resolved, outcomes):
        signal, error = outcome
        if signal is not None:
            signals.append(signal)
        elif error:
            failed.append((requested, error))
    signals.sort(
        key=lambda item: (
            is_qualified(item, config),
            item.score,
            len(item.price_action_confirmations),
            item.volume_ratio,
            item.adx,
        ),
        reverse=True,
    )
    return ScanResult(tuple(signals), tuple(failed), unavailable, len(resolved))


def format_price(value: float) -> str:
    if value >= 1_000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.8f}".rstrip("0").rstrip(".")


def format_frvp(profile: FixedRangeVolumeProfile | None, position: str) -> str:
    if profile is None:
        return "Unavailable"
    return (
        f"{profile.bars} bars | POC {format_price(profile.poc)} | "
        f"VAH {format_price(profile.vah)} | VAL {format_price(profile.val)} | {position}"
    )


def concise_data_error(reason: str, limit: int = 100) -> str:
    detail = reason.split(": ", 1)[1] if ": " in reason else reason
    return " ".join(detail.split())[:limit]


def is_qualified(signal: Signal, config: Config) -> bool:
    return signal.score >= config.min_signal_score and not signal.filter_failures


def best_qualified_signal(result: ScanResult, config: Config) -> Signal | None:
    return next((signal for signal in result.signals if is_qualified(signal, config)), None)


def render_console(result: ScanResult, config: Config, cycle: int, elapsed: float) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]🛰️  CRYPTO SIGNAL SCANNER[/bold cyan]\n"
        f"[white]Cycle #{cycle}[/white]  [dim]{timestamp}[/dim]\n"
        f"[green]📡 {result.scanned_count}/{len(SYMBOLS)} matched[/green]  "
        f"[red]⚠️ {len(result.failed)} data issues[/red]  "
        f"[yellow]⏱️ {elapsed:.1f}s[/yellow]  [magenta]🎯 Min score: {config.min_signal_score}/100[/magenta]",
        border_style="bright_blue",
    ))
    table = Table(title="🏆 Top Setups", header_style="bold bright_cyan", border_style="blue", show_lines=False)
    console.print("[dim]Scores are unvalidated heuristics, not probabilities; fees, spread, slippage, funding, positions, and exits are not modeled.[/dim]")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Symbol", style="bold white")
    table.add_column("Side", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("ADX", justify="right")
    table.add_column("Volume", justify="right")
    table.add_column("PA", justify="center")
    table.add_column("FRVP", justify="center")
    table.add_column("Gate", justify="center")
    table.add_column("Entry", justify="right")
    for rank, signal in enumerate(result.signals[:12], start=1):
        side_style = "bold green" if signal.side == "LONG" else "bold red"
        qualified = is_qualified(signal, config)
        score_style = "bold green" if qualified else "red" if signal.filter_failures else "yellow"
        gate = Text("READY", style="bold green") if qualified else Text("BLOCKED", style="bold red") if signal.filter_failures else Text("WATCH", style="yellow")
        price_action = "+".join(
            f"{reason.split(' ', 1)[0]}{'↑' if ' LONG' in reason else '↓'}"
            for reason in signal.price_action_context
        ) or "—"
        table.add_row(
            str(rank), signal.requested_symbol, Text(f"{'🟢' if signal.side == 'LONG' else '🔴'} {signal.side}", style=side_style),
            Text(f"{signal.score}/100", style=score_style), f"{signal.rsi:.1f}", f"{signal.adx:.1f}",
            f"{signal.volume_ratio:.2f}x", price_action, signal.frvp_position, gate, format_price(signal.price),
        )
    if not result.signals:
        table.add_row("—", "No valid candidates", "—", "—", "—", "—", "—", "—", "—", "—", "—")
    console.print(table)

    best = best_qualified_signal(result, config)
    candidate = result.signals[0] if result.signals else None
    if best:
        console.print(Panel(
            f"[bold]{'🚀 LONG' if best.side == 'LONG' else '🔻 SHORT'} {best.requested_symbol}[/bold]  "
            f"[bold bright_green]Score {best.score}/100[/bold bright_green]\n"
            f"💵 Entry: {format_price(best.price)}   🛡️ Stop: {format_price(best.stop)}\n"
            f"🎯 TP1: {format_price(best.target_1)}   🎯 TP2: {format_price(best.target_2)}\n"
            f"🧠 Price action: {' · '.join(best.price_action_context) or 'None detected; advisory only'}\n"
            f"📦 FRVP: {format_frvp(best.frvp_profile, best.frvp_position)}\n"
            f"🧩 {' · '.join(best.reasons[:8])}", title="✨ Best Qualified Setup", border_style="green",
        ))
    elif candidate:
        blocked_by = "; ".join(candidate.filter_failures) or f"Score below {config.min_signal_score}/100"
        console.print(Panel(
            f"🟡 Best candidate: [bold]{candidate.requested_symbol}[/bold] ({candidate.side}, {candidate.score}/100)\n"
            f"📦 FRVP: {format_frvp(candidate.frvp_profile, candidate.frvp_position)}\n"
            f"🛑 Blocked by: {blocked_by}\n[dim]No trade signal.[/dim]",
            title="⏸️ NO TRADE", border_style="yellow",
        ))
    else:
        console.print(Panel("No directional candidate this cycle.", title="⏸️ NO TRADE", border_style="yellow"))

    if result.unavailable:
        missing = ", ".join(result.unavailable)
        console.print(f"[yellow]⚠️ Unlisted/inactive for {config.market_type}: {missing}[/yellow]")
    if result.failed:
        summary = ", ".join(f"{symbol} ({concise_data_error(reason)})" for symbol, reason in result.failed[:10])
        extra = f" +{len(result.failed) - 10} more" if len(result.failed) > 10 else ""
        console.print(f"[red]⚠️ Data errors: {summary}{extra}[/red]")


def telegram_message(
    result: ScanResult,
    config: Config,
    cycle: int,
    elapsed: float,
    selected_signal: Signal | None = None,
) -> str:
    esc = html.escape
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "🛰️ <b>CRYPTO SIGNAL SCANNER</b>",
        f"🔄 Scan #{cycle} | ⏱️ {elapsed:.1f}s | 📡 {result.scanned_count}/{len(SYMBOLS)} matched | ⚠️ {len(result.failed)} data issues",
        f"🕒 {timestamp}",
        f"⏳ Candles: {config.entry_timeframe} entry / {config.trend_timeframe} trend",
        "",
    ]
    best = selected_signal or best_qualified_signal(result, config)
    candidate = result.signals[0] if result.signals else None
    if best:
        emoji = "🟢🚀" if best.side == "LONG" else "🔴🔻"
        lines.extend([
            f"{emoji} <b>{best.side} — {esc(best.requested_symbol)}</b>",
            f"💯 Setup score: <b>{best.score}/100</b> <i>(rule-based score, not a win probability)</i>",
            f"💵 Entry reference: <code>{format_price(best.price)}</code>",
            f"🛡️ ATR stop reference: <code>{format_price(best.stop)}</code>",
            f"🎯 Target 1: <code>{format_price(best.target_1)}</code>",
            f"🎯 Target 2: <code>{format_price(best.target_2)}</code>",
            f"🧠 Price action: {esc(' · '.join(best.price_action_context) or 'None detected; advisory only')}",
            f"📦 FRVP: <code>{esc(format_frvp(best.frvp_profile, best.frvp_position))}</code>",
            f"📊 RSI: {best.rsi:.1f} | ADX: {best.adx:.1f} | Volume: {best.volume_ratio:.2f}x | Entry drift: {best.entry_drift_atr:.2f} ATR",
            f"🧩 <b>Confluence:</b> {esc(' · '.join(best.reasons[:8]) or 'Mixed technical conditions')}",
        ])
    elif candidate:
        blocked_by = "; ".join(candidate.filter_failures) or f"Score below {config.min_signal_score}/100"
        lines.extend([
            "⏸️ <b>NO TRADE — no candidate passed all quality checks</b>",
            f"👀 Best candidate: <b>{esc(candidate.requested_symbol)}</b> · {candidate.side} · {candidate.score}/100",
            f"📦 FRVP: <code>{esc(format_frvp(candidate.frvp_profile, candidate.frvp_position))}</code>",
            f"🛑 Blocked by: {esc(blocked_by)}",
        ])
    else:
        lines.append("⏸️ <b>NO TRADE — no directional candidate this cycle</b>")
    if result.unavailable:
        lines.extend(["", f"⚠️ Unavailable markets ({len(result.unavailable)}): <code>{esc(', '.join(result.unavailable[:12]))}</code>"])
    if result.failed:
        issues = ", ".join(f"{symbol} ({concise_data_error(reason, 70)})" for symbol, reason in result.failed[:5])
        extra = f" +{len(result.failed) - 5} more" if len(result.failed) > 5 else ""
        lines.extend(["", f"⚠️ Data issues: {issues}{extra}"])
    lines.extend(["", "ℹ️ Alert only. ATR levels are references; no orders are placed. Market risk remains."])
    lines.append("⚠️ Score is an unvalidated heuristic, not a probability. Fees, spread, slippage, funding, liquidation, open positions, and exits are not modeled.")
    return "\n".join(lines)


def telegram_dedupe_key(signal: Signal) -> str:
    return f"SIGNAL:{signal.requested_symbol}:{signal.side}:{signal.candle_timestamp}"


def load_telegram_state() -> tuple[str | None, list[str], str | None, int | None]:
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, [], None, None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("Could not read Telegram deduplication state (%s)", type(exc).__name__)
        return None, [], None, None
    key = state.get("last_telegram_key") if isinstance(state, dict) else None
    key = key if isinstance(key, str) else None
    raw_sent_keys = state.get("sent_signal_keys", []) if isinstance(state, dict) else []
    sent_keys = list(dict.fromkeys(item for item in raw_sent_keys if isinstance(item, str)))[-512:] if isinstance(raw_sent_keys, list) else []
    last_state = state.get("last_notification_state") if isinstance(state, dict) else None
    if last_state not in ("READY", "NO_TRADE"):
        if key == "NO_TRADE":
            last_state = "NO_TRADE"
        elif key and key.startswith(("SIGNAL:", "SIGNAL_CANDLE:")):
            last_state = "READY"
        else:
            last_state = None
    candle = state.get("last_signal_candle") if isinstance(state, dict) else None
    if isinstance(candle, bool):
        candle = None
    if not isinstance(candle, int):
        candle = None
    if candle is None and key:
        try:
            candle = int(key.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            pass
    return key, sent_keys, last_state, candle


def save_telegram_state(
    key: str | None,
    sent_signal_keys: list[str],
    notification_state: str | None,
    last_signal_candle: int | None,
) -> None:
    temporary_file = STATE_FILE.with_suffix(".tmp")
    try:
        temporary_file.write_text(
            json.dumps({
                "last_telegram_key": key,
                "last_signal_candle": last_signal_candle,
                "sent_signal_keys": sent_signal_keys[-512:],
                "last_notification_state": notification_state,
            }),
            encoding="utf-8",
        )
        temporary_file.replace(STATE_FILE)
    except OSError as exc:
        logger.warning("Could not save Telegram deduplication state (%s)", type(exc).__name__)


async def send_telegram(
    session: aiohttp.ClientSession,
    config: Config,
    message: str,
    silent: bool = False,
) -> bool:
    if not config.telegram_token or not config.telegram_chat_id:
        return False
    url = f"https://api.telegram.org/bot{config.telegram_token}/sendMessage"
    payload = {
        "chat_id": config.telegram_chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "disable_notification": silent,
    }
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as response:
            if response.status >= 400:
                logger.error("Telegram delivery failed with HTTP %s", response.status)
                return False
            body = await response.json(content_type=None)
            if not body.get("ok"):
                logger.error("Telegram rejected the message: %s", str(body.get("description", "unknown error"))[:160])
                return False
            logger.info("📨 Telegram update sent")
            return True
    except Exception as exc:
        logger.error("Telegram delivery failed: %s", type(exc).__name__)
        return False


async def run() -> None:
    load_dotenv()
    config = Config.from_env()
    exchange_class = getattr(ccxt, config.exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unknown CCXT exchange id: {config.exchange_id}")
    exchange = exchange_class({
        "enableRateLimit": True,
        "timeout": 15_000,
        "options": {"defaultType": config.market_type},
    })
    telegram_configured = bool(config.telegram_token and config.telegram_chat_id)
    if bool(config.telegram_token) != bool(config.telegram_chat_id):
        logger.warning("Set both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable Telegram alerts")
    console.print(Panel.fit(
        f"[bold cyan]🛰️ Starting {config.exchange_id} {config.market_type} scanner[/bold cyan]\n"
        f"🪙 Watchlist: {len(SYMBOLS)} symbols | ⏱️ Every {config.interval_seconds}s | "
        f"📨 Telegram: {'ON' if telegram_configured else 'OFF'}",
        border_style="bright_magenta",
    ))
    http_session = aiohttp.ClientSession()
    cycle = 0
    if telegram_configured:
        last_telegram_key, sent_signal_keys, last_notification_state, last_signal_candle = load_telegram_state()
    else:
        last_telegram_key, sent_signal_keys, last_notification_state, last_signal_candle = None, [], None, None
    try:
        markets = await exchange.load_markets()
        resolved, unavailable = resolve_markets(markets, SYMBOLS, config.market_type)
        logger.info("Loaded %d exchange markets; %d/%d requested symbols are available", len(markets), len(resolved), len(SYMBOLS))
        if unavailable:
            logger.warning("Unavailable symbols: %s", ", ".join(unavailable))
        while True:
            cycle += 1
            started = time.monotonic()
            try:
                result = await scan(exchange, config, markets)
                elapsed = time.monotonic() - started
                render_console(result, config, cycle, elapsed)
                if telegram_configured:
                    best_ready_signal = best_qualified_signal(result, config)
                    ready_signals = [best_ready_signal] if best_ready_signal else []
                    if ready_signals:
                        pending_signals = [
                            signal
                            for signal in ready_signals
                            if telegram_dedupe_key(signal) not in sent_signal_keys
                        ]
                        send_failed = False
                        for signal in pending_signals:
                            notification_key = telegram_dedupe_key(signal)
                            sent = await send_telegram(
                                http_session,
                                config,
                                telegram_message(result, config, cycle, elapsed, selected_signal=signal),
                            )
                            if sent:
                                last_telegram_key = notification_key
                                last_signal_candle = signal.candle_timestamp
                                sent_signal_keys.append(notification_key)
                                sent_signal_keys = sent_signal_keys[-512:]
                                last_notification_state = "READY"
                                save_telegram_state(
                                    last_telegram_key,
                                    sent_signal_keys,
                                    last_notification_state,
                                    last_signal_candle,
                                )
                            else:
                                send_failed = True
                        if not pending_signals:
                            logger.info("Best READY setup was already sent for its signal candle")
                        elif send_failed:
                            logger.warning("Some READY Telegram setups failed; failed sends will retry next cycle")
                        if last_notification_state != "READY":
                            last_notification_state = "READY"
                            save_telegram_state(
                                last_telegram_key,
                                sent_signal_keys,
                                last_notification_state,
                                last_signal_candle,
                            )
                    else:
                        if last_notification_state == "NO_TRADE":
                            logger.info("Repeated NO TRADE Telegram update suppressed")
                        else:
                            sent = await send_telegram(
                                http_session,
                                config,
                                telegram_message(result, config, cycle, elapsed),
                                silent=True,
                            )
                            if sent:
                                last_telegram_key = "NO_TRADE"
                                last_notification_state = "NO_TRADE"
                                save_telegram_state(
                                    last_telegram_key,
                                    sent_signal_keys,
                                    last_notification_state,
                                    last_signal_candle,
                                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scan cycle %d failed; the scanner will retry", cycle)
            delay = max(0.0, config.interval_seconds - (time.monotonic() - started))
            await asyncio.sleep(delay)
    finally:
        await http_session.close()
        await exchange.close()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]👋 Scanner stopped by user.[/bold yellow]")
