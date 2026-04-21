from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import io
import json
import logging
import re
import sys
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import streamlit as st
from statsmodels.tsa.seasonal import STL
import urllib3
import yfinance as yf

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except Exception:
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False


if any(flag in sys.argv for flag in ("--self-test", "--benchmark", "--ml-backtest")):
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)


APP_TITLE = "Single-File Swing Trading Dashboard"
CACHE_TTL_SECONDS = 3600
MACRO_HISTORY_YEARS = 6
OPTIMIZED_TICKERS = ("SPY", "QQQ", "069500.KS")
OPTIMIZED_TICKER_SET = set(OPTIMIZED_TICKERS)
DEFAULT_WATCHLIST = OPTIMIZED_TICKERS
REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
MACRO_TICKERS = ("SPY", "^VIX", "HYG", "IEF", "RSP", "XLY", "XLP", "UUP", "TIPS")
ML_TARGET_SYMBOL = "QQQ"
ML_FEATURE_TICKERS = ("QQQ", "SPY", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "DIA", "IWM", "TLT", "IEF", "HYG", "^VIX")
ML_BATCH_TICKERS = tuple(dict.fromkeys(ML_FEATURE_TICKERS + MACRO_TICKERS))
ML_TRANSACTION_COST = 0.001
ML_BUY_HORIZON_BARS = 15
ML_SELL_HOLDOUT_BARS = 5
ML_THRESHOLD_GRID = (0.50, 0.55, 0.60, 0.65, 0.70)
ML_CV_SPLITS = 6
ML_CV_TEST_SIZE = 63
ML_CV_GAP = 15
ML_RANDOM_STATE = 42
ML_BUY_TARGET_ATR = 1.5
ML_BUY_STOP_ATR = 1.0
ML_MIN_CANDIDATE_SAMPLES = 24
ML_MACRO_FEAR_BUY_VETO = 32.0
ML_DEFAULT_SELL_VETO_PROFILE = "none"
KOREAN_TICKER_PATTERN = re.compile(r"^(?P<code>\d{6})(?:\.(?P<suffix>KS|KQ))?$", re.IGNORECASE)
STATE_COLORS = {
    "Strong Buy": "#15803d",
    "Buy": "#1d4ed8",
    "Weak Buy": "#60a5fa",
    "Hold / Neutral": "#475569",
    "Weak Sell": "#f59e0b",
    "Sell": "#ea580c",
    "Strong Sell": "#dc2626",
}
STATE_PRIORITY = {
    "Strong Buy": 0,
    "Buy": 1,
    "Weak Buy": 2,
    "Hold / Neutral": 3,
    "Weak Sell": 4,
    "Sell": 5,
    "Strong Sell": 6,
}


@dataclass(frozen=True)
class SignalPreset:
    name: str
    strong_threshold: int
    watch_threshold: int
    cooldown_bars: int


@dataclass(frozen=True)
class TickerSnapshot:
    ticker: str
    resolved_symbol: str
    as_of: pd.Timestamp
    last_close: float
    daily_return: float | None
    state: str
    buy_score: int
    sell_score: int
    cycle_score: float | None
    td_label: str
    atr_stretch: float | None
    rsi: float | None
    fear_greed: float | None
    stop_distance: float | None
    note: str
    warning: str | None


@dataclass(frozen=True)
class ReplayEvent:
    ticker: str
    trade: int
    date: pd.Timestamp
    signal: str
    price: float
    state: str
    buy_score: int
    sell_score: int
    position_after: float
    reason: str


@dataclass(frozen=True)
class ReplayTrade:
    ticker: str
    trade: int
    status: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    total_return: float
    hold_days: int
    partial_scaled_out: str


@dataclass(frozen=True)
class MLThresholdSelection:
    buy_threshold: float
    sell_threshold: float
    sell_veto_profile: str
    summary: dict[str, float | int | None]


@dataclass(frozen=True)
class MLSellVetoProfile:
    name: str
    bull_only: bool = False
    max_cycle_score: float | None = None
    max_sell_score: float | None = None
    min_rs_percentile: float | None = None
    max_fear_greed: float | None = None


@dataclass(frozen=True)
class ModelDescriptor:
    name: str
    estimator_name: str


ML_BULL_REGIMES = frozenset({"Bull Trend", "Bull Pullback"})
ML_SELL_VETO_PROFILES = (
    MLSellVetoProfile(name=ML_DEFAULT_SELL_VETO_PROFILE),
    MLSellVetoProfile(name="bull_cycle55", bull_only=True, max_cycle_score=55.0),
)
ML_SELL_VETO_PROFILE_MAP = {profile.name: profile for profile in ML_SELL_VETO_PROFILES}
ML_ACTIVE_SELL_VETO_PROFILES = tuple(
    profile for profile in ML_SELL_VETO_PROFILES if profile.name != ML_DEFAULT_SELL_VETO_PROFILE
)


PROFILE_PRESETS = {
    "Aggressive": SignalPreset(name="Aggressive", strong_threshold=66, watch_threshold=52, cooldown_bars=3),
    "Balanced": SignalPreset(name="Balanced", strong_threshold=70, watch_threshold=56, cooldown_bars=4),
    "Conservative": SignalPreset(name="Conservative", strong_threshold=74, watch_threshold=60, cooldown_bars=6),
}
DEFAULT_SIGNAL_PRESET = PROFILE_PRESETS["Balanced"]

BUY_COMPONENT_WEIGHTS: dict[str, float] = {
    "reversion": 1.00,
    "td": 1.00,
    "macro": 1.00,
    "relative": 1.00,
    "volume": 1.00,
    "macd": 1.00,
    "regime": 1.00,
    "confluence": 1.00,
    "confirmation": 1.00,
}

SELL_COMPONENT_WEIGHTS: dict[str, float] = {
    "exhaustion": 1.00,
    "td": 1.00,
    "heat": 1.00,
    "distribution": 1.00,
    "confluence": 1.00,
}

BUY_CUTOFFS: dict[str, float] = {
    "rsi_reset": 40.0,
    "rsi_extreme": 32.0,
    "mfi_reset": 35.0,
    "mfi_extreme": 18.0,
    "fear_reset": 40.0,
    "fear_extreme": 25.0,
    "bb_low": 0.22,
}

SELL_CUTOFFS: dict[str, float] = {
    "rsi_hot": 66.0,
    "rsi_extreme": 74.0,
    "mfi_hot": 68.0,
    "mfi_extreme": 82.0,
    "greed_hot": 60.0,
    "greed_extreme": 70.0,
    "bb_high": 0.78,
    "bb_extreme": 0.90,
}

DEEP_VALUE_CYCLE_THRESHOLD = 25.0
DEEP_VALUE_EXTREME_CYCLE_THRESHOLD = 18.0
DEEP_VALUE_RSI_THRESHOLD = 42.0
DEEP_VALUE_EXTREME_RSI_THRESHOLD = 38.0
DEEP_VALUE_SETUP_THRESHOLD = 8
DEEP_VALUE_EXTREME_SETUP_THRESHOLD = 9
HEAT_TRIM_FEAR_THRESHOLD = 60.0
HEAT_TRIM_EXTREME_FEAR_THRESHOLD = 70.0
HEAT_TRIM_RSI_THRESHOLD = 66.0
HEAT_TRIM_EXTREME_RSI_THRESHOLD = 72.0
HEAT_TRIM_SETUP_THRESHOLD = 8
HEAT_TRIM_EXTREME_SETUP_THRESHOLD = 9
HEAT_TRIM_REDUCE_WEIGHT = 0.25
HEAT_TRIM_COOLDOWN_BARS = 20
LOW_CONFLUENCE_SCALE_IN_WEIGHT = 0.25
LOW_CONFLUENCE_SCALE_IN_COOLDOWN_BARS = 5

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def preset_min_hold_bars(preset: SignalPreset) -> int:
    return max(2, preset.cooldown_bars - 1)


def preset_reentry_lock_bars(preset: SignalPreset) -> int:
    return max(1, preset.cooldown_bars // 2)


def configure_page() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="SW", layout="wide")


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .hero {
            padding: 1.35rem 1.5rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #0f172a 0%, #14532d 40%, #f5efe6 100%);
            color: #f8fafc;
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.1;
        }
        .hero p {
            margin: 0.55rem 0 0;
            color: rgba(248, 250, 252, 0.9);
        }
        .metric-card {
            padding: 1rem 1.05rem;
            border-radius: 16px;
            background: #f8fafc;
            border: 1px solid rgba(148, 163, 184, 0.24);
            min-height: 108px;
        }
        .metric-card .eyebrow {
            font-size: 0.8rem;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-card .value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 0.22rem;
        }
        .metric-card .caption {
            margin-top: 0.24rem;
            color: #475569;
            font-size: 0.92rem;
        }
        .section-label {
            margin-top: 0.35rem;
            color: #0f172a;
            font-weight: 700;
            font-size: 1.12rem;
        }
        .pill {
            display: inline-block;
            padding: 0.34rem 0.65rem;
            margin: 0.14rem 0.24rem 0.14rem 0;
            border-radius: 999px;
            background: #e2e8f0;
            color: #0f172a;
            font-size: 0.88rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="eyebrow">{title}</div>
            <div class="value">{value}</div>
            <div class="caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


class StreamlitProgressTracker:
    def __init__(self, total_steps: int) -> None:
        self.total_steps = max(1, int(total_steps))
        self.completed_steps = 0
        self.logs: list[str] = []
        self.container = st.container()
        self.container.markdown('<div class="section-label">Run Progress</div>', unsafe_allow_html=True)
        self.message = self.container.empty()
        self.progress = self.container.progress(0, text=f"Preparing... (0/{self.total_steps})")
        self.log_caption = self.container.empty()
        self.log_caption.caption("Recent processing log")
        self.log_box = self.container.empty()
        self.update("Preparing dashboard run", "Initializing controls and cached resources.", advance=0)

    def _render(self, stage: str, detail: str) -> None:
        percent = min(100, int(round((self.completed_steps / self.total_steps) * 100.0)))
        self.progress.progress(percent, text=f"{stage} ({self.completed_steps}/{self.total_steps})")
        self.message.markdown(f"**Current task:** {stage}  \n{detail}")
        recent_logs = self.logs[-10:]
        if recent_logs:
            self.log_box.code("\n".join(recent_logs), language="text")

    def update(self, stage: str, detail: str, *, advance: int = 0) -> None:
        if advance:
            self.completed_steps = min(self.total_steps, self.completed_steps + int(advance))
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {stage}: {detail}")
        self._render(stage, detail)

    def finish(self, detail: str) -> None:
        self.completed_steps = self.total_steps
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] Completed: {detail}")
        self.progress.progress(100, text=f"Completed ({self.total_steps}/{self.total_steps})")
        self.message.success(detail)
        self.log_box.code("\n".join(self.logs[-10:]), language="text")

    def fail(self, detail: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] Failed: {detail}")
        self.progress.progress(min(100, int(round((self.completed_steps / self.total_steps) * 100.0))), text=f"Failed ({self.completed_steps}/{self.total_steps})")
        self.message.error(detail)
        self.log_box.code("\n".join(self.logs[-10:]), language="text")


def parse_tickers(raw: str) -> list[str]:
    tokens = [item.strip().upper() for item in raw.replace("\n", ",").split(",")]
    return [item for item in tokens if item]


def dedupe_preserve_order(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    getter = getattr(row, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(row, key, default)


class TupleRowAccessor:
    __slots__ = ("row",)

    def __init__(self, row: Any) -> None:
        self.row = row

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self.row, key, default)


def is_us_batch_ticker(ticker: str) -> bool:
    normalized = ticker.strip().upper()
    return normalized in ML_BATCH_TICKERS and not bool(KOREAN_TICKER_PATTERN.fullmatch(normalized))


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.{digits}f}"


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100.0:.1f}%"


def _format_score(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.0f}"


def format_snapshot_label(snapshot: TickerSnapshot) -> str:
    resolved = (snapshot.resolved_symbol or "").strip()
    if not resolved or resolved == snapshot.ticker:
        return snapshot.ticker
    return f"{snapshot.ticker} ({resolved})"


def weighted_component(score: float, weight: float) -> float:
    return score * weight


def compute_stop_breach_metrics(
    close_value: float | None,
    stop_value: float | None,
    atr_value: float | None,
) -> tuple[bool, float | None, float | None]:
    if (
        close_value is None
        or stop_value is None
        or pd.isna(close_value)
        or pd.isna(stop_value)
        or stop_value <= 0
    ):
        return False, None, None

    close_breach = close_value <= stop_value
    breach_pct = max(0.0, (stop_value - close_value) / stop_value) if close_breach else 0.0
    if atr_value is None or pd.isna(atr_value) or atr_value <= 0:
        breach_atr = None
    else:
        breach_atr = max(0.0, (stop_value - close_value) / atr_value) if close_breach else 0.0
    return close_breach, breach_pct, breach_atr


def evaluate_stop_exit(
    close_value: float | None,
    stop_value: float | None,
    atr_value: float | None,
    sell_score: int,
    buy_score: int,
    bearish_context: bool,
    sell_context: bool,
    hold_lock_active: bool = False,
    macro_risk_off: bool = False,
) -> tuple[bool, bool, float | None]:
    close_breach, breach_pct, breach_atr = compute_stop_breach_metrics(close_value, stop_value, atr_value)
    if not close_breach:
        return False, False, breach_atr

    soft_atr = 0.25 if not hold_lock_active else 0.55
    hard_atr = 0.55 if not hold_lock_active else 0.90
    soft_pct = 0.0015 if not hold_lock_active else 0.0040
    hard_pct = 0.0040 if not hold_lock_active else 0.0075
    score_gap = 6 if not hold_lock_active else 10
    hard_score_gap = 12 if not hold_lock_active else 18
    if macro_risk_off:
        soft_atr *= 0.85
        hard_atr *= 0.85
        soft_pct *= 0.85
        hard_pct *= 0.85
        score_gap = max(3, score_gap - 1)
        hard_score_gap = max(8, hard_score_gap - 2)

    soft_breach = bool(
        (breach_pct is not None and breach_pct >= soft_pct)
        or (breach_atr is not None and breach_atr >= soft_atr)
    )
    hard_breach = bool(
        (breach_pct is not None and breach_pct >= hard_pct)
        or (breach_atr is not None and breach_atr >= hard_atr)
    )
    score_confirm = sell_score >= (buy_score + score_gap)
    score_emergency = sell_score >= (buy_score + hard_score_gap)

    emergency_exit = bool(hard_breach or score_emergency)
    confirmed_exit = bool(
        emergency_exit
        or (
            soft_breach
            and (sell_context or bearish_context or score_confirm)
        )
    )
    return confirmed_exit, emergency_exit, breach_atr


def calc_13612w_momentum(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    momentum = (
        12.0 * (series / series.shift(5) - 1.0)
        + 4.0 * (series / series.shift(15) - 1.0)
        + 2.0 * (series / series.shift(30) - 1.0)
        + 1.0 * (series / series.shift(60) - 1.0)
    )
    return momentum.replace([np.inf, -np.inf], np.nan)


def build_controls() -> tuple[list[str], int, SignalPreset, bool]:
    st.sidebar.subheader("Analysis Controls")
    watchlist_raw = st.sidebar.text_area(
        "Watchlist tickers",
        value=", ".join(DEFAULT_WATCHLIST),
        help="This version is tuned only for SPY, QQQ, and 069500.KS (KODEX 200 ETF). Other inputs are ignored.",
        height=96,
    )
    history_years = st.sidebar.slider("History lookback (years)", min_value=1, max_value=5, value=3, step=1)
    refresh = st.sidebar.button("Refresh cached data", width="stretch")
    st.sidebar.caption("Balanced tuned mode only. SPY, QQQ, and KODEX 200 are the only supported assets.")

    parsed_watchlist = parse_tickers(watchlist_raw)
    watchlist = [ticker for ticker in parsed_watchlist if ticker in OPTIMIZED_TICKER_SET]
    if not watchlist:
        watchlist = list(DEFAULT_WATCHLIST)
    return watchlist, int(history_years), DEFAULT_SIGNAL_PRESET, refresh


def normalize_datetime_index(index: Any) -> pd.DatetimeIndex:
    normalized = pd.to_datetime(index)
    if isinstance(normalized, pd.DatetimeIndex):
        if normalized.tz is not None:
            normalized = normalized.tz_localize(None)
        return normalized
    return pd.DatetimeIndex(normalized)


def align_series_to_target_index(series: pd.Series, target_index: pd.Index) -> pd.Series:
    normalized_target = normalize_datetime_index(target_index)
    if series is None or series.empty:
        return pd.Series(np.nan, index=normalized_target, dtype=float)

    aligned = pd.to_numeric(series, errors="coerce").copy()
    aligned.index = normalize_datetime_index(aligned.index).normalize()
    aligned = aligned[~aligned.index.duplicated(keep="last")].sort_index()
    target_dates = normalized_target.normalize()
    mapped = aligned.reindex(target_dates).ffill().bfill()
    return pd.Series(mapped.to_numpy(dtype=float), index=normalized_target, name=series.name)


def _select_single_ticker_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame.copy()
    level_zero = set(frame.columns.get_level_values(0))
    if any(column in level_zero for column in REQUIRED_OHLCV_COLUMNS):
        ticker_name = frame.columns.get_level_values(1)[0]
        return frame.xs(ticker_name, axis=1, level=1, drop_level=True)
    ticker_name = frame.columns.get_level_values(0)[0]
    return frame.xs(ticker_name, axis=1, level=0, drop_level=True)


def normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS))

    normalized = _select_single_ticker_frame(frame).copy()
    normalized.index = normalize_datetime_index(normalized.index)
    normalized = normalized.sort_index()
    if "Adj Close" in normalized.columns and "Close" not in normalized.columns:
        normalized = normalized.rename(columns={"Adj Close": "Close"})
    for column in REQUIRED_OHLCV_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    normalized = normalized.loc[:, list(REQUIRED_OHLCV_COLUMNS)]
    normalized = normalized.dropna(subset=["Open", "High", "Low", "Close"])
    normalized["Volume"] = pd.to_numeric(normalized["Volume"], errors="coerce").fillna(0.0)
    return normalized


def quiet_yfinance_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        return yf.download(*args, **kwargs)


def download_chart_api_history(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": int(start_ts.timestamp()),
        "period2": int(end_ts.timestamp()),
        "interval": "1d",
        "includePrePost": "false",
        "events": "div,splits",
    }
    response = requests.get(
        url,
        params=params,
        timeout=30,
        verify=False,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result") or []
    if not result:
        return pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS))
    chart = result[0]
    timestamps = chart.get("timestamp") or []
    quote = ((chart.get("indicators") or {}).get("quote") or [{}])[0]
    if not timestamps:
        return pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS))
    frame = pd.DataFrame(
        {
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Volume": quote.get("volume", []),
        },
        index=pd.to_datetime(timestamps, unit="s"),
    )
    return normalize_ohlcv_frame(frame)


def resolve_yfinance_candidates(ticker: str) -> list[str]:
    normalized = ticker.strip().upper()
    match = KOREAN_TICKER_PATTERN.fullmatch(normalized)
    if not match:
        return [normalized]

    code = match.group("code")
    suffix = match.group("suffix")
    if suffix:
        return dedupe_preserve_order([normalized, f"{code}.KS", f"{code}.KQ", code])
    return dedupe_preserve_order([normalized, f"{code}.KS", f"{code}.KQ"])


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def download_single_history_cached(requested_ticker: str, start_iso: str, end_iso: str) -> dict[str, Any]:
    start_ts = pd.Timestamp(start_iso)
    end_ts = pd.Timestamp(end_iso) + pd.Timedelta(days=1)
    last_error = ""

    for candidate in resolve_yfinance_candidates(requested_ticker):
        try:
            raw = quiet_yfinance_download(
                candidate,
                start=start_ts,
                end=end_ts,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
        except Exception as exc:
            last_error = f"{candidate}: {exc}"
            continue

        frame = normalize_ohlcv_frame(raw)
        if not frame.empty:
            return {
                "requested_ticker": requested_ticker,
                "resolved_symbol": candidate,
                "frame": frame,
                "warning": None,
            }
        try:
            frame = download_chart_api_history(candidate, start_ts, end_ts)
        except Exception as exc:
            last_error = f"{candidate}: {exc}"
            continue
        if not frame.empty:
            return {
                "requested_ticker": requested_ticker,
                "resolved_symbol": candidate,
                "frame": frame,
                "warning": "Used Yahoo chart API fallback due yfinance SSL/rate-limit issue",
            }
        last_error = f"{candidate}: no usable OHLCV rows returned"

    return {
        "requested_ticker": requested_ticker,
        "resolved_symbol": resolve_yfinance_candidates(requested_ticker)[-1],
        "frame": pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS)),
        "warning": last_error or "No usable OHLCV data returned",
    }


def _extract_ticker_history_from_download(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS))
    if not isinstance(raw.columns, pd.MultiIndex):
        return normalize_ohlcv_frame(raw)

    level_zero = set(raw.columns.get_level_values(0))
    level_one = set(raw.columns.get_level_values(1))
    if ticker in level_one and any(column in level_zero for column in REQUIRED_OHLCV_COLUMNS):
        frame = raw.xs(ticker, axis=1, level=1, drop_level=True)
        return normalize_ohlcv_frame(frame)
    if ticker in level_zero:
        frame = raw.xs(ticker, axis=1, level=0, drop_level=True)
        return normalize_ohlcv_frame(frame)
    return pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS))


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def download_batch_history_cached(requested_tickers: tuple[str, ...], start_iso: str, end_iso: str) -> dict[str, dict[str, Any]]:
    start_ts = pd.Timestamp(start_iso)
    end_ts = pd.Timestamp(end_iso) + pd.Timedelta(days=1)
    ordered_tickers = tuple(dedupe_preserve_order([ticker.strip().upper() for ticker in requested_tickers if ticker]))
    payloads: dict[str, dict[str, Any]] = {}
    if not ordered_tickers:
        return payloads

    raw = pd.DataFrame()
    batch_error = ""
    try:
        raw = quiet_yfinance_download(
            list(ordered_tickers),
            start=start_ts,
            end=end_ts,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as exc:
        batch_error = str(exc)

    for ticker in ordered_tickers:
        frame = _extract_ticker_history_from_download(raw, ticker)
        if not frame.empty:
            payloads[ticker] = {
                "requested_ticker": ticker,
                "resolved_symbol": ticker,
                "frame": frame,
                "warning": None,
            }
            continue

        fallback_payload = download_single_history_cached(ticker, start_iso, end_iso)
        warning = fallback_payload.get("warning")
        if batch_error:
            warning = f"Batch download fallback: {batch_error}" if not warning else f"{warning} | Batch download fallback: {batch_error}"
        payloads[ticker] = {
            **fallback_payload,
            "warning": warning,
        }
    return payloads


def load_price_payloads(
    requested_tickers: list[str],
    start_iso: str,
    end_iso: str,
) -> dict[str, dict[str, Any]]:
    ordered_tickers = dedupe_preserve_order([ticker.strip().upper() for ticker in requested_tickers if ticker])
    payloads: dict[str, dict[str, Any]] = {}
    batch_tickers = [ticker for ticker in ordered_tickers if is_us_batch_ticker(ticker)]
    if batch_tickers:
        payloads.update(download_batch_history_cached(tuple(batch_tickers), start_iso, end_iso))
    for ticker in ordered_tickers:
        if ticker in payloads:
            continue
        payloads[ticker] = download_single_history_cached(ticker, start_iso, end_iso)
    return payloads


def _extract_close_panel(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=list(MACRO_TICKERS))

    if isinstance(raw.columns, pd.MultiIndex):
        level_zero = set(raw.columns.get_level_values(0))
        if "Adj Close" in level_zero:
            panel = raw["Adj Close"].copy()
        elif "Close" in level_zero:
            panel = raw["Close"].copy()
        else:
            panel = raw.copy()
    else:
        panel = raw.copy()
    panel.index = normalize_datetime_index(panel.index)
    panel = panel.sort_index().ffill()
    return panel.reindex(columns=list(MACRO_TICKERS))


def download_macro_close_panel_via_chart_api(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    close_map: dict[str, pd.Series] = {}
    for ticker in MACRO_TICKERS:
        try:
            frame = download_chart_api_history(ticker, start_ts, end_ts + pd.Timedelta(days=1))
        except Exception:
            continue
        if frame.empty:
            continue
        close_map[ticker] = frame["Close"]
    if not close_map:
        return pd.DataFrame(columns=list(MACRO_TICKERS))
    panel = pd.concat(close_map, axis=1).sort_index().ffill()
    return panel.reindex(columns=list(MACRO_TICKERS))


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_macro_fear_greed_cached(end_iso: str) -> dict[str, Any]:
    end_ts = pd.Timestamp(end_iso)
    start_ts = end_ts - pd.DateOffset(years=MACRO_HISTORY_YEARS)
    close_panel = pd.DataFrame(columns=list(MACRO_TICKERS))
    try:
        raw = quiet_yfinance_download(
            list(MACRO_TICKERS),
            start=start_ts,
            end=end_ts + pd.Timedelta(days=1),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        close_panel = _extract_close_panel(raw)
    except Exception as exc:
        close_panel = download_macro_close_panel_via_chart_api(start_ts, end_ts)
        if close_panel.dropna(how="all").empty:
            score_index = pd.date_range(end=end_ts, periods=260, freq="B")
            neutral_series = pd.Series(50.0, index=score_index, name="FearGreed")
            tips_series = pd.Series(dtype=float, name="TIPS13612W")
            return {
                "score_series": neutral_series,
                "benchmark_close": pd.Series(dtype=float, name="SPY"),
                "tips_13612w_momentum": tips_series,
                "latest_tips_13612w_momentum": None,
                "plot_df": pd.DataFrame({"Score": neutral_series}),
                "latest_score": 50.0,
                "latest_label": "Neutral",
                "latest_factors": pd.Series(dtype=float),
                "warning": f"Fear & Greed fallback to neutral: {exc}",
            }

    if close_panel.dropna(how="all").empty:
        close_panel = download_macro_close_panel_via_chart_api(start_ts, end_ts)
        if close_panel.dropna(how="all").empty:
            score_index = pd.date_range(end=end_ts, periods=260, freq="B")
            neutral_series = pd.Series(50.0, index=score_index, name="FearGreed")
            tips_series = pd.Series(dtype=float, name="TIPS13612W")
            return {
                "score_series": neutral_series,
                "benchmark_close": pd.Series(dtype=float, name="SPY"),
                "tips_13612w_momentum": tips_series,
                "latest_tips_13612w_momentum": None,
                "plot_df": pd.DataFrame({"Score": neutral_series}),
                "latest_score": 50.0,
                "latest_label": "Neutral",
                "latest_factors": pd.Series(dtype=float),
                "warning": "Fear & Greed fallback to neutral: macro basket returned no usable close data",
            }

    score_series, latest_factors, plot_df = compute_macro_fear_greed(close_panel)
    latest_score = float(score_series.dropna().iloc[-1]) if not score_series.dropna().empty else 50.0
    latest_label = classify_fear_greed(latest_score)
    tips_close = close_panel["TIPS"].copy() if "TIPS" in close_panel.columns else pd.Series(dtype=float, name="TIPS")
    tips_momentum = calc_13612w_momentum(tips_close).rename("TIPS13612W") if not tips_close.empty else pd.Series(dtype=float, name="TIPS13612W")
    latest_tips = float(tips_momentum.dropna().iloc[-1]) if not tips_momentum.dropna().empty else None
    return {
        "score_series": score_series,
        "benchmark_close": close_panel["SPY"].copy() if "SPY" in close_panel.columns else pd.Series(dtype=float, name="SPY"),
        "tips_13612w_momentum": tips_momentum,
        "latest_tips_13612w_momentum": latest_tips,
        "plot_df": plot_df,
        "latest_score": latest_score,
        "latest_label": latest_label,
        "latest_factors": latest_factors,
        "warning": None,
    }


def rolling_percentile(
    series: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
    inverse: bool = False,
) -> pd.Series:
    if min_periods is None:
        min_periods = window
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)

    for index in range(len(values)):
        value = values[index]
        if np.isnan(value):
            continue
        start = max(0, index - window + 1)
        window_values = values[start : index + 1]
        window_values = window_values[~np.isnan(window_values)]
        if len(window_values) < min_periods:
            continue
        less_count = np.sum(window_values < value)
        equal_count = np.sum(window_values == value)
        percentile = (less_count + (0.5 * equal_count)) / len(window_values)
        result[index] = 1.0 - percentile if inverse else percentile
    return pd.Series(result, index=series.index, name=series.name)


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calc_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["Close"].shift(1)
    true_range = np.maximum(
        frame["High"] - frame["Low"],
        np.maximum((frame["High"] - prev_close).abs(), (frame["Low"] - prev_close).abs()),
    )
    return pd.Series(true_range, index=frame.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calc_mfi(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    typical_price = (frame["High"] + frame["Low"] + frame["Close"]) / 3.0
    raw_money_flow = typical_price * pd.to_numeric(frame["Volume"], errors="coerce").fillna(0.0)
    direction = typical_price.diff()
    positive_flow = raw_money_flow.where(direction > 0.0, 0.0)
    negative_flow = raw_money_flow.where(direction < 0.0, 0.0).abs()
    positive_sum = positive_flow.rolling(period, min_periods=period).sum()
    negative_sum = negative_flow.rolling(period, min_periods=period).sum()
    flow_ratio = positive_sum / negative_sum.replace(0.0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + flow_ratio))
    mfi = mfi.where(negative_sum.gt(0.0), 100.0)
    mfi = mfi.where(positive_sum.gt(0.0), 0.0)
    return mfi.fillna(50.0)


def calc_adx(frame: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = frame["High"].diff()
    down_move = -frame["Low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0), index=frame.index)
    atr = calc_atr(frame, period)
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return plus_di, minus_di, adx


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def calc_cmf(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    high_low_range = (frame["High"] - frame["Low"]).replace(0.0, np.nan)
    money_flow_multiplier = (((frame["Close"] - frame["Low"]) - (frame["High"] - frame["Close"])) / high_low_range).fillna(0.0)
    money_flow_volume = money_flow_multiplier * pd.to_numeric(frame["Volume"], errors="coerce").fillna(0.0)
    volume_sum = pd.to_numeric(frame["Volume"], errors="coerce").rolling(period, min_periods=period).sum()
    return money_flow_volume.rolling(period, min_periods=period).sum() / volume_sum.replace(0.0, np.nan)


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = pd.Series(np.sign(close.diff().fillna(0.0)), index=close.index)
    signed_volume = direction * pd.to_numeric(volume, errors="coerce").fillna(0.0)
    return signed_volume.cumsum()


def calc_bb_position(series: pd.Series, period: int = 20, stdev: float = 2.0) -> pd.Series:
    basis = series.rolling(period, min_periods=period).mean()
    dev = series.rolling(period, min_periods=period).std()
    upper = basis + (stdev * dev)
    lower = basis - (stdev * dev)
    position = (series - lower) / (upper - lower).replace(0.0, np.nan)
    return position.clip(-0.5, 1.5)


def compute_td_sequential(frame: pd.DataFrame) -> pd.DataFrame:
    work = pd.DataFrame(index=frame.index)
    work["BuySetup"] = 0
    work["SellSetup"] = 0
    work["BuyCountdown"] = 0
    work["SellCountdown"] = 0
    buy_setup_loc = work.columns.get_loc("BuySetup")
    sell_setup_loc = work.columns.get_loc("SellSetup")
    buy_countdown_loc = work.columns.get_loc("BuyCountdown")
    sell_countdown_loc = work.columns.get_loc("SellCountdown")

    close_values = pd.to_numeric(frame["Close"], errors="coerce").to_numpy(dtype=float)
    active_buy_countdown = False
    active_sell_countdown = False
    buy_count = 0
    sell_count = 0

    for index in range(4, len(work)):
        previous_buy = int(work.iat[index - 1, buy_setup_loc])
        previous_sell = int(work.iat[index - 1, sell_setup_loc])
        work.iat[index, buy_setup_loc] = previous_buy + 1 if close_values[index] < close_values[index - 4] else 0
        work.iat[index, sell_setup_loc] = previous_sell + 1 if close_values[index] > close_values[index - 4] else 0

        if work.iat[index, buy_setup_loc] == 9:
            active_buy_countdown = True
            buy_count = 0
        if work.iat[index, sell_setup_loc] == 9:
            active_sell_countdown = True
            sell_count = 0

        if active_buy_countdown and close_values[index] <= close_values[index - 2]:
            buy_count += 1
            work.iat[index, buy_countdown_loc] = buy_count
            if buy_count == 13:
                active_buy_countdown = False

        if active_sell_countdown and close_values[index] >= close_values[index - 2]:
            sell_count += 1
            work.iat[index, sell_countdown_loc] = sell_count
            if sell_count == 13:
                active_sell_countdown = False

    return work


def compute_stl_cycle(frame: pd.DataFrame) -> pd.DataFrame:
    work = pd.DataFrame(index=frame.index)
    work["Trend"] = np.nan
    work["Residual"] = np.nan
    work["CycleScore"] = np.nan

    close = pd.to_numeric(frame["Close"], errors="coerce").replace(0, np.nan).dropna()
    if len(close) < 120:
        return work

    trend_window = 31
    if trend_window % 2 == 0:
        trend_window += 1
    try:
        result = STL(np.log(close), period=5, robust=True, trend=trend_window).fit()
    except Exception:
        return work

    work.loc[close.index, "Trend"] = np.exp(result.trend)
    work.loc[close.index, "Residual"] = result.resid
    cycle_score = rolling_percentile(work["Residual"], window=252, min_periods=63) * 100.0
    work["CycleScore"] = cycle_score.clip(0, 100)
    return work


def classify_fear_greed(score: float) -> str:
    if score >= 80:
        return "Extreme Greed"
    if score >= 60:
        return "Greed"
    if score <= 20:
        return "Extreme Fear"
    if score <= 40:
        return "Fear"
    return "Neutral"


def compute_macro_fear_greed(close_panel: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    close_panel = close_panel.copy().ffill().dropna(how="all")
    weights = pd.Series(
        {
            "BB Pos": 0.10,
            "RSI Mom": 0.10,
            "MA125 Div": 0.10,
            "Breadth": 0.15,
            "Sector": 0.15,
            "Credit": 0.10,
            "VIX Inv": 0.15,
            "Dollar Inv": 0.15,
        }
    )

    spy = close_panel["SPY"]
    vix = close_panel["^VIX"]
    hyg = close_panel["HYG"]
    ief = close_panel["IEF"]
    rsp = close_panel["RSP"]
    xly = close_panel["XLY"]
    xlp = close_panel["XLP"]
    uup = close_panel["UUP"]

    ma20 = spy.rolling(20).mean()
    std20 = spy.rolling(20).std()
    upper = ma20 + (2.0 * std20)
    lower = ma20 - (2.0 * std20)
    bb_pos = ((spy - lower) / (upper - lower)).replace([np.inf, -np.inf], np.nan)
    ma125_div = (spy - spy.rolling(125).mean()) / spy.rolling(125).mean()

    factor_frame = pd.DataFrame(index=close_panel.index)
    factor_frame["BB Pos"] = rolling_percentile(bb_pos, window=252, min_periods=63)
    factor_frame["RSI Mom"] = rolling_percentile(calc_rsi(spy), window=252, min_periods=63)
    factor_frame["MA125 Div"] = rolling_percentile(ma125_div, window=252, min_periods=63)
    factor_frame["Breadth"] = rolling_percentile(rsp / spy, window=252, min_periods=63)
    factor_frame["Sector"] = rolling_percentile(xly / xlp, window=252, min_periods=63)
    factor_frame["Credit"] = rolling_percentile(hyg / ief, window=252, min_periods=63)
    factor_frame["VIX Inv"] = rolling_percentile(vix, window=252, min_periods=63, inverse=True)
    factor_frame["Dollar Inv"] = rolling_percentile(uup, window=252, min_periods=63, inverse=True)

    weighted_values = factor_frame.mul(weights, axis=1)
    available_weights = factor_frame.notna().mul(weights, axis=1).sum(axis=1)
    score = weighted_values.sum(axis=1, min_count=1).div(available_weights.replace(0, np.nan)).fillna(0.5) * 100.0
    score.name = "FearGreed"

    non_empty_factors = factor_frame.dropna(how="all")
    latest_factors = (
        non_empty_factors.iloc[-1].sort_values(ascending=False) * 100.0
        if not non_empty_factors.empty
        else pd.Series(dtype=float)
    )
    plot_df = pd.DataFrame(
        {
            "Score": score,
            "SPY": spy,
            "Breadth": factor_frame["Breadth"] * 100.0,
            "Sector": factor_frame["Sector"] * 100.0,
            "Credit": factor_frame["Credit"] * 100.0,
        }
    ).dropna(how="all")
    return score, latest_factors, plot_df


def td_label_from_row(row: pd.Series) -> str:
    if pd.isna(row.get("BuySetup")) and pd.isna(row.get("SellSetup")):
        return "Unavailable"
    buy_cd = int(row.get("BuyCountdown", 0) or 0)
    sell_cd = int(row.get("SellCountdown", 0) or 0)
    buy_setup = int(row.get("BuySetup", 0) or 0)
    sell_setup = int(row.get("SellSetup", 0) or 0)
    if buy_cd == 13:
        return "Buy countdown 13"
    if sell_cd == 13:
        return "Sell countdown 13"
    if buy_setup >= 7:
        return f"Buy setup {buy_setup}"
    if sell_setup >= 7:
        return f"Sell setup {sell_setup}"
    return "No exhaustion signal"


def build_td_label_series(frame: pd.DataFrame) -> pd.Series:
    buy_setup = pd.to_numeric(frame["BuySetup"], errors="coerce").fillna(0).astype(int)
    sell_setup = pd.to_numeric(frame["SellSetup"], errors="coerce").fillna(0).astype(int)
    buy_countdown = pd.to_numeric(frame["BuyCountdown"], errors="coerce").fillna(0).astype(int)
    sell_countdown = pd.to_numeric(frame["SellCountdown"], errors="coerce").fillna(0).astype(int)

    labels = pd.Series("No exhaustion signal", index=frame.index, dtype="object")
    sell_setup_mask = sell_setup >= 7
    buy_setup_mask = buy_setup >= 7
    sell_countdown_mask = sell_countdown == 13
    buy_countdown_mask = buy_countdown == 13

    if sell_setup_mask.any():
        labels.loc[sell_setup_mask] = sell_setup.loc[sell_setup_mask].map(lambda value: f"Sell setup {value}")
    if buy_setup_mask.any():
        labels.loc[buy_setup_mask] = buy_setup.loc[buy_setup_mask].map(lambda value: f"Buy setup {value}")
    labels.loc[sell_countdown_mask] = "Sell countdown 13"
    labels.loc[buy_countdown_mask] = "Buy countdown 13"
    return labels


def compute_indicators(
    price_df: pd.DataFrame,
    macro_score_series: pd.Series | None,
    benchmark_close_series: pd.Series | None = None,
    tips_momentum_series: pd.Series | None = None,
    asset_symbol: str | None = None,
) -> pd.DataFrame:
    frame = normalize_ohlcv_frame(price_df)
    if frame.empty:
        return frame

    work = frame.copy()
    reference_symbol = (asset_symbol or "").upper()
    work["AssetSymbol"] = reference_symbol
    work["ReferenceAsset"] = reference_symbol in OPTIMIZED_TICKER_SET
    work["EMA20"] = work["Close"].ewm(span=20, adjust=False, min_periods=20).mean()
    work["EMA50"] = work["Close"].ewm(span=50, adjust=False, min_periods=50).mean()
    work["EMA200"] = work["Close"].ewm(span=200, adjust=False, min_periods=200).mean()
    work["RSI"] = calc_rsi(work["Close"], period=14)
    work["ATR"] = calc_atr(work, period=14)
    work["MFI"] = calc_mfi(work, period=14)
    work["CMF20"] = calc_cmf(work, period=20)
    work["OBV"] = calc_obv(work["Close"], work["Volume"])
    work["OBVSlope10"] = work["OBV"] - work["OBV"].shift(10)
    plus_di, minus_di, adx = calc_adx(work, period=14)
    macd_line, macd_signal, macd_hist = calc_macd(work["Close"])
    work["PlusDI"] = plus_di
    work["MinusDI"] = minus_di
    work["ADX"] = adx
    work["MACDLine"] = macd_line
    work["MACDSignal"] = macd_signal
    work["MACDHist"] = macd_hist
    work["MACDHistDelta"] = work["MACDHist"].diff()
    work["MACDBullCross"] = (work["MACDLine"] > work["MACDSignal"]) & (work["MACDLine"].shift(1) <= work["MACDSignal"].shift(1))
    work["MACDBearCross"] = (work["MACDLine"] < work["MACDSignal"]) & (work["MACDLine"].shift(1) >= work["MACDSignal"].shift(1))
    bb_basis20 = work["Close"].rolling(20, min_periods=20).mean()
    bb_std20 = work["Close"].rolling(20, min_periods=20).std()
    bb_upper20 = bb_basis20 + (2.0 * bb_std20)
    bb_lower20 = bb_basis20 - (2.0 * bb_std20)
    work["BBPos20"] = calc_bb_position(work["Close"], period=20, stdev=2.0)
    work["BBWidth20"] = (bb_upper20 - bb_lower20) / bb_basis20.replace(0.0, np.nan)
    work["BBWidthPct"] = rolling_percentile(work["BBWidth20"], window=252, min_periods=63) * 100.0
    work["ChandelierStop"] = work["High"].rolling(22, min_periods=22).max() - (3.0 * work["ATR"])
    work["ATRStretch"] = (work["Close"] - work["EMA20"]) / work["ATR"].replace(0, np.nan)
    work["StopDistance"] = (work["Close"] / work["ChandelierStop"]) - 1.0
    work["EMA50Slope20"] = work["EMA50"] - work["EMA50"].shift(20)
    work["VolumeSMA20"] = pd.to_numeric(work["Volume"], errors="coerce").rolling(20, min_periods=20).mean()
    work["VolumeRatio20"] = pd.to_numeric(work["Volume"], errors="coerce") / work["VolumeSMA20"].replace(0.0, np.nan)
    work["VolumePct20"] = rolling_percentile(work["VolumeRatio20"], window=252, min_periods=63) * 100.0
    work["VolumeClimaxUp"] = (work["VolumeRatio20"] >= 1.35) & (work["Close"] >= work["Open"])
    work["VolumeClimaxDown"] = (work["VolumeRatio20"] >= 1.35) & (work["Close"] < work["Open"])
    work["VolumeDryUp"] = (work["VolumeRatio20"] <= 0.72).fillna(False)
    work["MFIChange5"] = work["MFI"] - work["MFI"].shift(5)
    work["BuyTriggerPrice"] = work["Close"] > work["High"].shift(1)
    work["SellTriggerPrice"] = work["Close"] < work["Low"].shift(1)
    work["CloseUpDay"] = work["Close"] > work["Close"].shift(1)
    work["CloseDownDay"] = work["Close"] < work["Close"].shift(1)
    work["StopTouched"] = work["Low"] <= work["ChandelierStop"]
    work["StopCloseBreach"] = work["Close"] <= work["ChandelierStop"]

    td = compute_td_sequential(work)
    work = work.join(td, how="left")
    stl = compute_stl_cycle(work)
    work = work.join(stl, how="left")
    work["TrendDelta5"] = work["Trend"] - work["Trend"].shift(5)
    work["TDLabel"] = build_td_label_series(work)

    if benchmark_close_series is None or benchmark_close_series.empty:
        work["RelativeStrength"] = np.nan
        work["RSMomentum63"] = np.nan
        work["RSPercentile"] = np.nan
    else:
        benchmark_close = align_series_to_target_index(benchmark_close_series, work.index)
        relative_strength = work["Close"] / benchmark_close.replace(0.0, np.nan)
        work["RelativeStrength"] = relative_strength
        work["RSMomentum63"] = relative_strength / relative_strength.shift(63) - 1.0
        work["RSPercentile"] = rolling_percentile(relative_strength, window=252, min_periods=63) * 100.0

    if macro_score_series is None or macro_score_series.empty:
        work["FearGreed"] = 50.0
    else:
        aligned = align_series_to_target_index(macro_score_series, work.index)
        work["FearGreed"] = aligned.fillna(50.0)
    if tips_momentum_series is None or tips_momentum_series.empty:
        work["TIPS13612WMomentum"] = np.nan
        work["TIPSRiskOff"] = False
    else:
        aligned_tips = align_series_to_target_index(tips_momentum_series, work.index)
        work["TIPS13612WMomentum"] = aligned_tips
        work["TIPSRiskOff"] = aligned_tips.lt(0.0).fillna(False)
    return work


def attach_rule_only_columns(
    state_frame: pd.DataFrame,
    model_mode: str = "rules_only",
    sell_veto_profile: str = ML_DEFAULT_SELL_VETO_PROFILE,
) -> pd.DataFrame:
    if state_frame.empty:
        frame = state_frame.copy()
        for column, default in [
            ("MLBuyProb", np.nan),
            ("MLSellProb", np.nan),
            ("MLBuyApproved", True),
            ("MLSellApproved", True),
            ("ModelMode", model_mode),
            ("SellVetoProfile", sell_veto_profile),
            ("StrategySource", "RuleOnly"),
            ("SignalNote", ""),
        ]:
            frame[column] = default
        return frame

    frame = state_frame.copy()
    frame["MLBuyProb"] = pd.to_numeric(frame.get("MLBuyProb", np.nan), errors="coerce")
    frame["MLSellProb"] = pd.to_numeric(frame.get("MLSellProb", np.nan), errors="coerce")
    frame["MLBuyApproved"] = frame.get("MLBuyApproved", True)
    frame["MLSellApproved"] = frame.get("MLSellApproved", True)
    frame["ModelMode"] = frame.get("ModelMode", model_mode)
    frame["SellVetoProfile"] = frame.get("SellVetoProfile", sell_veto_profile)
    frame["StrategySource"] = frame.get("StrategySource", "RuleOnly")
    frame["SignalNote"] = frame.get("SignalNote", "")
    frame["MLBuyApproved"] = frame["MLBuyApproved"].fillna(True).astype(bool)
    frame["MLSellApproved"] = frame["MLSellApproved"].fillna(True).astype(bool)
    frame["SellVetoProfile"] = frame["SellVetoProfile"].fillna(sell_veto_profile).astype(str)
    return frame


def sanitize_feature_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", symbol.upper()).strip("_")


def build_cross_asset_feature_frame(
    target_index: pd.Index,
    price_payloads: dict[str, dict[str, Any]],
    *,
    target_symbol: str,
) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    normalized_target = target_symbol.upper()
    for symbol in ML_FEATURE_TICKERS:
        if symbol.upper() == normalized_target:
            continue
        payload = price_payloads.get(symbol.upper()) or price_payloads.get(symbol)
        if not payload:
            continue
        frame = payload.get("frame", pd.DataFrame())
        if frame.empty or "Close" not in frame.columns:
            continue
        close = align_series_to_target_index(frame["Close"], target_index)
        prefix = sanitize_feature_symbol(symbol)
        returns = close.pct_change()
        sma20 = close.rolling(20, min_periods=20).mean()
        sma50 = close.rolling(50, min_periods=50).mean()
        std20 = close.rolling(20, min_periods=20).std().replace(0.0, np.nan)
        features[f"{prefix}_RET1"] = close.pct_change(1)
        features[f"{prefix}_RET5"] = close.pct_change(5)
        features[f"{prefix}_RET20"] = close.pct_change(20)
        features[f"{prefix}_RET63"] = close.pct_change(63)
        features[f"{prefix}_VOL20"] = returns.rolling(20, min_periods=20).std() * np.sqrt(252.0)
        features[f"{prefix}_SMA20_GAP"] = close / sma20 - 1.0
        features[f"{prefix}_SMA50_GAP"] = close / sma50 - 1.0
        features[f"{prefix}_Z20"] = (close - sma20) / std20
    if not features:
        return pd.DataFrame(index=normalize_datetime_index(target_index))
    return pd.DataFrame(features, index=normalize_datetime_index(target_index)).replace([np.inf, -np.inf], np.nan)


def build_ml_feature_frame(
    indicator_frame: pd.DataFrame,
    baseline_state_frame: pd.DataFrame,
    price_payloads: dict[str, dict[str, Any]],
    *,
    target_symbol: str,
) -> pd.DataFrame:
    joined = indicator_frame.join(
        baseline_state_frame.loc[
            :,
            [
                "BuyScore",
                "SellScore",
                "BuySetupActive",
                "SellSetupActive",
                "BuyTrigger",
                "SellTrigger",
                "HoldLockActive",
                "ReentryLockActive",
                "StopExitConfirmed",
                "StopEmergencyExit",
            ],
        ],
        how="left",
    ).copy()
    regime_map = {
        "Bull Trend": 2,
        "Bull Pullback": 1,
        "Range / Transition": 0,
        "Bear Rally": -1,
        "Bear Trend": -2,
    }
    joined["RegimeCode"] = joined.get("Regime", pd.Series(index=joined.index, dtype=object)).map(regime_map).fillna(0).astype(float)
    drop_columns = {
        "AssetSymbol",
        "TDLabel",
        "Regime",
        "State",
        "SignalNote",
        "BuyReasonText",
        "SellReasonText",
    }
    feature_frame = joined.drop(columns=[column for column in drop_columns if column in joined.columns], errors="ignore").copy()
    for column in feature_frame.columns:
        if pd.api.types.is_bool_dtype(feature_frame[column]):
            feature_frame[column] = feature_frame[column].astype(int)
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.join(
        build_cross_asset_feature_frame(feature_frame.index, price_payloads, target_symbol=target_symbol),
        how="left",
    )
    return feature_frame.replace([np.inf, -np.inf], np.nan)


def infer_position_flags(state_frame: pd.DataFrame, preset: SignalPreset) -> pd.DataFrame:
    if state_frame.empty:
        return pd.DataFrame(index=state_frame.index, columns=["InPositionBefore", "InPositionAfter"], dtype=bool)

    rows = state_frame.reset_index(drop=True)
    in_position_before: list[bool] = []
    in_position_after: list[bool] = []
    in_position = False
    last_buy_index = -10_000
    last_sell_index = -10_000
    entry_index = -10_000
    min_hold_bars = preset_min_hold_bars(preset)

    for position, row in enumerate(rows.itertuples(index=False, name="PositionRow")):
        in_position_before.append(in_position)
        buy_trigger = _coerce_bool(getattr(row, "BuyTrigger", False))
        sell_trigger = _coerce_bool(getattr(row, "SellTrigger", False))
        buy_ready = (not in_position) and buy_trigger and (position - last_buy_index) >= preset.cooldown_bars
        hold_bars_elapsed = position - entry_index if in_position else 10_000
        force_exit = _coerce_bool(getattr(row, "StopEmergencyExit", False))
        sell_ready = (
            in_position
            and sell_trigger
            and ((hold_bars_elapsed >= min_hold_bars) or force_exit)
            and (position - last_sell_index) >= preset.cooldown_bars
        )

        if buy_ready:
            in_position = True
            last_buy_index = position
            entry_index = position
        elif sell_ready:
            in_position = False
            last_sell_index = position
            entry_index = -10_000
        in_position_after.append(in_position)

    return pd.DataFrame({"InPositionBefore": in_position_before, "InPositionAfter": in_position_after}, index=state_frame.index)


def build_ml_label_frame(state_frame: pd.DataFrame, preset: SignalPreset) -> pd.DataFrame:
    if state_frame.empty:
        return pd.DataFrame(index=state_frame.index)

    frame = state_frame.copy()
    position_flags = infer_position_flags(frame, preset)
    frame = frame.join(position_flags, how="left")

    buy_candidate = frame["BuyTrigger"].fillna(False).astype(bool)
    sell_candidate = (
        frame["InPositionBefore"].fillna(False).astype(bool)
        & frame["SellTrigger"].fillna(False).astype(bool)
        & ~frame["StopExitConfirmed"].fillna(False).astype(bool)
        & ~frame["StopEmergencyExit"].fillna(False).astype(bool)
    )

    buy_label = pd.Series(False, index=frame.index, dtype=bool)
    sell_label = pd.Series(False, index=frame.index, dtype=bool)
    buy_forward_return = pd.Series(np.nan, index=frame.index, dtype=float)
    sell_edge = pd.Series(np.nan, index=frame.index, dtype=float)

    rows = frame.reset_index(drop=True)
    for position, row in enumerate(rows.itertuples(index=False, name="LabelRow")):
        if buy_candidate.iloc[position]:
            entry_position = position + 1
            if entry_position < len(rows):
                entry_open = _coerce_float(getattr(rows.iloc[entry_position], "Open", np.nan))
                atr_value = _coerce_float(getattr(row, "ATR", np.nan))
                if entry_open is not None and atr_value is not None and atr_value > 0:
                    target_price = entry_open + (atr_value * ML_BUY_TARGET_ATR)
                    stop_price = entry_open - (atr_value * ML_BUY_STOP_ATR)
                    hit_target_first = False
                    horizon_end = min(len(rows), entry_position + ML_BUY_HORIZON_BARS)
                    for future_position in range(entry_position, horizon_end):
                        future_row = rows.iloc[future_position]
                        future_high = _coerce_float(getattr(future_row, "High", np.nan))
                        future_low = _coerce_float(getattr(future_row, "Low", np.nan))
                        if future_high is None or future_low is None:
                            continue
                        if future_low <= stop_price and future_high >= target_price:
                            hit_target_first = False
                            break
                        if future_low <= stop_price:
                            hit_target_first = False
                            break
                        if future_high >= target_price:
                            net_return = target_price / entry_open - 1.0 - (2.0 * ML_TRANSACTION_COST)
                            hit_target_first = net_return > 0.0
                            break
                    exit_position = min(len(rows) - 1, horizon_end - 1)
                    exit_close = _coerce_float(getattr(rows.iloc[exit_position], "Close", np.nan))
                    if exit_close is not None:
                        buy_forward_return.iloc[position] = exit_close / entry_open - 1.0 - (2.0 * ML_TRANSACTION_COST)
                    buy_label.iloc[position] = hit_target_first

        if sell_candidate.iloc[position]:
            exit_position = position + 1
            hold_position = min(len(rows) - 1, position + ML_SELL_HOLDOUT_BARS)
            if exit_position < len(rows):
                next_open = _coerce_float(getattr(rows.iloc[exit_position], "Open", np.nan))
                anchor_close = _coerce_float(getattr(row, "Close", np.nan))
                hold_close = _coerce_float(getattr(rows.iloc[hold_position], "Close", np.nan))
                if next_open is not None and anchor_close is not None and hold_close is not None and anchor_close > 0:
                    immediate_exit_return = next_open / anchor_close - 1.0 - ML_TRANSACTION_COST
                    hold_return = hold_close / anchor_close - 1.0 - ML_TRANSACTION_COST
                    sell_edge.iloc[position] = immediate_exit_return - hold_return
                    sell_label.iloc[position] = immediate_exit_return > hold_return

    return pd.DataFrame(
        {
            "BuyCandidate": buy_candidate,
            "SellCandidate": sell_candidate,
            "BuyLabel": buy_label,
            "SellLabel": sell_label,
            "BuyForwardReturn": buy_forward_return,
            "SellEdge": sell_edge,
            "InPositionBefore": frame["InPositionBefore"].fillna(False).astype(bool),
            "InPositionAfter": frame["InPositionAfter"].fillna(False).astype(bool),
        },
        index=frame.index,
    )


def resolve_boosting_descriptor(prefer_lightgbm: bool = True) -> ModelDescriptor:
    if prefer_lightgbm and LIGHTGBM_AVAILABLE and LGBMClassifier is not None:
        return ModelDescriptor(name="lightgbm", estimator_name="LGBMClassifier")
    return ModelDescriptor(name="hist_gradient_boosting", estimator_name="HistGradientBoostingClassifier")


def build_random_forest_estimator() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=280,
        max_depth=7,
        min_samples_leaf=6,
        min_samples_split=12,
        class_weight="balanced_subsample",
        random_state=ML_RANDOM_STATE,
        n_jobs=-1,
    )


def build_boosting_estimator(prefer_lightgbm: bool = True) -> HistGradientBoostingClassifier | Any:
    descriptor = resolve_boosting_descriptor(prefer_lightgbm=prefer_lightgbm)
    if descriptor.name == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            n_estimators=220,
            learning_rate=0.045,
            num_leaves=31,
            min_child_samples=18,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=ML_RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    return HistGradientBoostingClassifier(
        loss="log_loss",
        max_depth=6,
        max_iter=220,
        learning_rate=0.045,
        min_samples_leaf=18,
        l2_regularization=0.1,
        max_leaf_nodes=31,
        random_state=ML_RANDOM_STATE,
    )


def prepare_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    reference_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    numeric = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    reference_numeric = numeric if reference_frame is None else reference_frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill_values = reference_numeric.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return numeric.fillna(fill_values).astype(float)


def fit_calibrated_binary_classifier(
    estimator_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
) -> Any | None:
    if X.empty or y.empty:
        return None
    y_int = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    class_counts = y_int.value_counts()
    if len(class_counts) < 2 or int(class_counts.min()) < 2:
        return None
    cv_folds = min(3, int(class_counts.min()))
    if cv_folds < 2:
        return None
    try:
        model = CalibratedClassifierCV(
            estimator=estimator_factory(),
            method="sigmoid",
            cv=cv_folds,
            ensemble=False,
        )
        model.fit(X, y_int)
    except Exception:
        return None
    return model


def predict_binary_probabilities(model: Any | None, X: pd.DataFrame) -> np.ndarray:
    if model is None or X.empty:
        return np.full(len(X), np.nan, dtype=float)
    try:
        probabilities = model.predict_proba(X)[:, 1]
    except Exception:
        return np.full(len(X), np.nan, dtype=float)
    return np.asarray(probabilities, dtype=float)


def _combine_probability_columns(columns: list[np.ndarray]) -> np.ndarray:
    if not columns:
        return np.array([], dtype=float)
    matrix = np.column_stack(columns)
    with np.errstate(invalid="ignore"):
        combined = np.nanmean(matrix, axis=1)
    if matrix.shape[1] == 0:
        return np.full(matrix.shape[0], np.nan, dtype=float)
    all_nan_mask = np.isnan(matrix).all(axis=1)
    combined[all_nan_mask] = np.nan
    return combined


def build_probability_views_for_target(
    train_feature_frame: pd.DataFrame,
    train_candidate_mask: pd.Series,
    train_label: pd.Series,
    test_feature_frame: pd.DataFrame,
    test_candidate_mask: pd.Series,
    feature_columns: list[str],
    *,
    prefer_lightgbm: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    descriptor = resolve_boosting_descriptor(prefer_lightgbm=prefer_lightgbm)
    model_names = ["random_forest", descriptor.name, "soft_ensemble"]
    train_views = {name: np.full(len(train_feature_frame), np.nan, dtype=float) for name in model_names}
    test_views = {name: np.full(len(test_feature_frame), np.nan, dtype=float) for name in model_names}

    candidate_train = train_candidate_mask.fillna(False).astype(bool)
    candidate_test = test_candidate_mask.fillna(False).astype(bool)
    if int(candidate_train.sum()) < ML_MIN_CANDIDATE_SAMPLES:
        return train_views, test_views, descriptor.name

    y_train = pd.to_numeric(train_label.loc[candidate_train], errors="coerce").fillna(0).astype(int)
    if y_train.nunique() < 2:
        return train_views, test_views, descriptor.name

    reference_frame = train_feature_frame.loc[:, feature_columns]
    X_train_candidates = prepare_feature_matrix(train_feature_frame.loc[candidate_train], feature_columns, reference_frame=reference_frame)
    X_test_candidates = prepare_feature_matrix(test_feature_frame.loc[candidate_test], feature_columns, reference_frame=reference_frame) if candidate_test.any() else pd.DataFrame(columns=feature_columns)

    rf_model = fit_calibrated_binary_classifier(build_random_forest_estimator, X_train_candidates, y_train)
    boost_model = fit_calibrated_binary_classifier(lambda: build_boosting_estimator(prefer_lightgbm=prefer_lightgbm), X_train_candidates, y_train)

    rf_train_probs = predict_binary_probabilities(rf_model, X_train_candidates)
    boost_train_probs = predict_binary_probabilities(boost_model, X_train_candidates)
    train_views["random_forest"][candidate_train.to_numpy()] = rf_train_probs
    train_views[descriptor.name][candidate_train.to_numpy()] = boost_train_probs
    train_views["soft_ensemble"][candidate_train.to_numpy()] = _combine_probability_columns([rf_train_probs, boost_train_probs])

    if candidate_test.any():
        rf_test_probs = predict_binary_probabilities(rf_model, X_test_candidates)
        boost_test_probs = predict_binary_probabilities(boost_model, X_test_candidates)
        test_views["random_forest"][candidate_test.to_numpy()] = rf_test_probs
        test_views[descriptor.name][candidate_test.to_numpy()] = boost_test_probs
        test_views["soft_ensemble"][candidate_test.to_numpy()] = _combine_probability_columns([rf_test_probs, boost_test_probs])

    return train_views, test_views, descriptor.name


def resolve_ml_sell_veto_profile(profile_name: str) -> MLSellVetoProfile:
    return ML_SELL_VETO_PROFILE_MAP.get(
        str(profile_name or ML_DEFAULT_SELL_VETO_PROFILE),
        ML_SELL_VETO_PROFILE_MAP[ML_DEFAULT_SELL_VETO_PROFILE],
    )


def build_ml_sell_veto_mask(
    filtered: pd.DataFrame,
    *,
    baseline_sell: pd.Series,
    stop_protected: pd.Series,
    ml_sell_approved: pd.Series,
    profile: MLSellVetoProfile,
) -> pd.Series:
    veto_mask = pd.Series(False, index=filtered.index, dtype=bool)
    if profile.name == ML_DEFAULT_SELL_VETO_PROFILE:
        return veto_mask

    veto_mask = baseline_sell & ml_sell_approved & ~stop_protected
    if not veto_mask.any():
        return veto_mask

    if profile.bull_only:
        regime = filtered.get("Regime", pd.Series("", index=filtered.index, dtype=object))
        veto_mask &= regime.isin(ML_BULL_REGIMES)
    if profile.max_cycle_score is not None:
        cycle_score = pd.to_numeric(filtered.get("CycleScore", pd.Series(np.nan, index=filtered.index)), errors="coerce")
        veto_mask &= cycle_score.le(profile.max_cycle_score).fillna(False)
    if profile.max_sell_score is not None:
        sell_score = pd.to_numeric(filtered.get("SellScore", pd.Series(np.nan, index=filtered.index)), errors="coerce")
        veto_mask &= sell_score.le(profile.max_sell_score).fillna(False)
    if profile.min_rs_percentile is not None:
        rs_percentile = pd.to_numeric(filtered.get("RSPercentile", pd.Series(np.nan, index=filtered.index)), errors="coerce")
        veto_mask &= rs_percentile.ge(profile.min_rs_percentile).fillna(False)
    if profile.max_fear_greed is not None:
        fear_greed = pd.to_numeric(filtered.get("FearGreed", pd.Series(np.nan, index=filtered.index)), errors="coerce")
        veto_mask &= fear_greed.le(profile.max_fear_greed).fillna(False)
    return veto_mask.fillna(False)


def apply_ml_thresholds_to_state_frame(
    base_state_frame: pd.DataFrame,
    buy_probabilities: pd.Series | np.ndarray,
    sell_probabilities: pd.Series | np.ndarray,
    buy_threshold: float,
    sell_threshold: float,
    *,
    model_mode: str,
    sell_veto_profile: str = ML_DEFAULT_SELL_VETO_PROFILE,
) -> pd.DataFrame:
    if base_state_frame.empty:
        return attach_rule_only_columns(
            base_state_frame.copy(),
            model_mode=model_mode,
            sell_veto_profile=sell_veto_profile,
        )

    filtered = base_state_frame.copy()
    sell_veto = resolve_ml_sell_veto_profile(sell_veto_profile)
    buy_prob_series = pd.Series(buy_probabilities, index=base_state_frame.index, dtype=float)
    sell_prob_series = pd.Series(sell_probabilities, index=base_state_frame.index, dtype=float)
    baseline_buy = filtered["BuyTrigger"].fillna(False).astype(bool)
    baseline_sell = filtered["SellTrigger"].fillna(False).astype(bool)
    stop_protected = filtered["StopExitConfirmed"].fillna(False).astype(bool) | filtered["StopEmergencyExit"].fillna(False).astype(bool)
    fear_greed = pd.Series(
        pd.to_numeric(filtered.get("FearGreed", pd.Series(np.nan, index=filtered.index)), errors="coerce"),
        index=filtered.index,
        dtype=float,
    )

    ml_buy_approved = pd.Series(True, index=filtered.index, dtype=bool)
    ml_sell_approved = pd.Series(True, index=filtered.index, dtype=bool)

    buy_known = buy_prob_series.notna()
    sell_known = sell_prob_series.notna()
    ml_buy_approved.loc[baseline_buy & buy_known] = buy_prob_series.loc[baseline_buy & buy_known] >= buy_threshold
    ml_sell_approved.loc[baseline_sell & sell_known] = sell_prob_series.loc[baseline_sell & sell_known] >= sell_threshold
    ml_sell_approved.loc[stop_protected] = True
    macro_fear_veto = baseline_buy & fear_greed.lt(ML_MACRO_FEAR_BUY_VETO).fillna(False)
    ml_buy_approved.loc[macro_fear_veto] = False
    sell_veto_mask = build_ml_sell_veto_mask(
        filtered,
        baseline_sell=baseline_sell,
        stop_protected=stop_protected,
        ml_sell_approved=ml_sell_approved,
        profile=sell_veto,
    )
    ml_sell_approved.loc[sell_veto_mask] = False

    filtered["MLBuyProb"] = buy_prob_series
    filtered["MLSellProb"] = sell_prob_series
    filtered["MLBuyApproved"] = ml_buy_approved
    filtered["MLSellApproved"] = ml_sell_approved
    filtered["ModelMode"] = filtered.get("ModelMode", model_mode)
    filtered["SellVetoProfile"] = filtered.get("SellVetoProfile", sell_veto.name)
    filtered["StrategySource"] = "RuleOnly"

    filtered.loc[baseline_buy & ml_buy_approved, "StrategySource"] = "MLApprovedBuy"
    filtered.loc[baseline_buy & ~ml_buy_approved, "StrategySource"] = "MLBlockedBuy"
    filtered.loc[baseline_sell & ml_sell_approved & ~stop_protected, "StrategySource"] = "MLApprovedSell"
    filtered.loc[baseline_sell & ~ml_sell_approved & ~stop_protected, "StrategySource"] = "MLBlockedSell"
    filtered.loc[stop_protected, "StrategySource"] = "HardStop"
    filtered.loc[macro_fear_veto, "StrategySource"] = "MacroFearVeto"
    filtered.loc[sell_veto_mask, "StrategySource"] = "MLSellVeto"

    filtered["BuyTrigger"] = baseline_buy & ml_buy_approved
    filtered["SellTrigger"] = baseline_sell & (ml_sell_approved | stop_protected)
    buy_side = filtered["State"].isin(["Strong Buy", "Buy", "Weak Buy"])
    sell_side = filtered["State"].isin(["Strong Sell", "Sell", "Weak Sell"])
    filtered.loc[buy_side & baseline_buy & ~filtered["BuyTrigger"], "State"] = "Hold / Neutral"
    filtered.loc[sell_side & baseline_sell & ~filtered["SellTrigger"] & ~stop_protected, "State"] = "Hold / Neutral"
    filtered["SignalNote"] = ""
    return attach_rule_only_columns(
        filtered,
        model_mode=model_mode,
        sell_veto_profile=sell_veto.name,
    )


def select_ml_strategy_for_state_frame(
    symbol: str,
    base_state_frame: pd.DataFrame,
    buy_prob_views: dict[str, np.ndarray],
    sell_prob_views: dict[str, np.ndarray],
    preset: SignalPreset,
) -> tuple[str, MLThresholdSelection] | None:
    best_selection: tuple[str, MLThresholdSelection] | None = None
    best_score = (-10_000.0, -10_000.0, -1.0, -1)

    candidate_model_names = [name for name in buy_prob_views if name in sell_prob_views]
    for model_name in candidate_model_names:
        if np.isnan(buy_prob_views[model_name]).all() and np.isnan(sell_prob_views[model_name]).all():
            continue
        for buy_threshold in ML_THRESHOLD_GRID:
            for sell_threshold in ML_THRESHOLD_GRID:
                for sell_veto_profile in ML_ACTIVE_SELL_VETO_PROFILES:
                    filtered = apply_ml_thresholds_to_state_frame(
                        base_state_frame,
                        buy_prob_views[model_name],
                        sell_prob_views[model_name],
                        buy_threshold,
                        sell_threshold,
                        model_mode=model_name,
                        sell_veto_profile=sell_veto_profile.name,
                    )
                    summary = build_trade_replay_from_state_frame(symbol, filtered, preset)[2]
                    excess_return = -10_000.0 if summary.get("excess_return") is None else float(summary["excess_return"])
                    strategy_total_return = -10_000.0 if summary.get("strategy_total_return") is None else float(summary["strategy_total_return"])
                    win_rate = -1.0 if summary.get("win_rate") is None else float(summary["win_rate"])
                    closed_trades = int(summary["closed_trades"] or 0)
                    score = (excess_return, strategy_total_return, win_rate, closed_trades)
                    if score > best_score:
                        best_score = score
                        best_selection = (
                            model_name,
                            MLThresholdSelection(
                                buy_threshold=buy_threshold,
                                sell_threshold=sell_threshold,
                                sell_veto_profile=sell_veto_profile.name,
                                summary=summary,
                            ),
                        )
    return best_selection


def build_ml_walkforward_bundle(
    symbol: str,
    indicator_frame: pd.DataFrame,
    baseline_state_frame: pd.DataFrame,
    price_payloads: dict[str, dict[str, Any]],
    preset: SignalPreset,
) -> dict[str, Any]:
    if symbol.upper() != ML_TARGET_SYMBOL or indicator_frame.empty or baseline_state_frame.empty:
        return {
            "enabled": False,
            "model_mode": "rules_only",
            "reason": "Rule-only asset",
            "state_frame": attach_rule_only_columns(baseline_state_frame.copy()),
            "comparison": {},
            "selection_frame": pd.DataFrame(),
            "oos_start": None,
        }

    feature_frame = build_ml_feature_frame(indicator_frame, baseline_state_frame, price_payloads, target_symbol=symbol)
    label_frame = build_ml_label_frame(baseline_state_frame, preset)
    feature_columns = [column for column in feature_frame.columns if feature_frame[column].notna().sum() >= 30]
    if not feature_columns:
        return {
            "enabled": False,
            "model_mode": "rules_only",
            "reason": "No usable ML features",
            "state_frame": attach_rule_only_columns(baseline_state_frame.copy()),
            "comparison": {},
            "selection_frame": pd.DataFrame(),
            "oos_start": None,
        }

    try:
        splitter = TimeSeriesSplit(
            n_splits=ML_CV_SPLITS,
            test_size=ML_CV_TEST_SIZE,
            gap=ML_CV_GAP,
        )
        splits = list(splitter.split(feature_frame))
    except Exception as exc:
        return {
            "enabled": False,
            "model_mode": "rules_only",
            "reason": f"TimeSeriesSplit unavailable: {exc}",
            "state_frame": attach_rule_only_columns(baseline_state_frame.copy()),
            "comparison": {},
            "selection_frame": pd.DataFrame(),
            "oos_start": None,
        }

    if not splits:
        return {
            "enabled": False,
            "model_mode": "rules_only",
            "reason": "Insufficient rows for walk-forward split",
            "state_frame": attach_rule_only_columns(baseline_state_frame.copy()),
            "comparison": {},
            "selection_frame": pd.DataFrame(),
            "oos_start": None,
        }

    index = feature_frame.index
    buy_prob_series = pd.Series(np.nan, index=index, dtype=float)
    sell_prob_series = pd.Series(np.nan, index=index, dtype=float)
    buy_approved_series = pd.Series(True, index=index, dtype=bool)
    sell_approved_series = pd.Series(True, index=index, dtype=bool)
    model_mode_series = pd.Series("rules_only", index=index, dtype=object)
    sell_veto_profile_series = pd.Series(ML_DEFAULT_SELL_VETO_PROFILE, index=index, dtype=object)
    strategy_source_series = pd.Series("RuleOnlyWarmup", index=index, dtype=object)
    selection_rows: list[dict[str, Any]] = []

    for fold_number, (train_positions, test_positions) in enumerate(splits, start=1):
        train_features = feature_frame.iloc[train_positions]
        test_features = feature_frame.iloc[test_positions]
        train_labels = label_frame.iloc[train_positions]
        test_labels = label_frame.iloc[test_positions]

        buy_train_views, buy_test_views, boosting_name = build_probability_views_for_target(
            train_features,
            train_labels["BuyCandidate"],
            train_labels["BuyLabel"],
            test_features,
            test_labels["BuyCandidate"],
            feature_columns,
        )
        sell_train_views, sell_test_views, _ = build_probability_views_for_target(
            train_features,
            train_labels["SellCandidate"],
            train_labels["SellLabel"],
            test_features,
            test_labels["SellCandidate"],
            feature_columns,
        )
        selection = select_ml_strategy_for_state_frame(symbol, baseline_state_frame.iloc[train_positions], buy_train_views, sell_train_views, preset)
        if selection is None:
            continue
        model_mode, threshold_selection = selection
        filtered_test = apply_ml_thresholds_to_state_frame(
            baseline_state_frame.iloc[test_positions],
            buy_test_views[model_mode],
            sell_test_views[model_mode],
            threshold_selection.buy_threshold,
            threshold_selection.sell_threshold,
            model_mode=model_mode,
            sell_veto_profile=threshold_selection.sell_veto_profile,
        )
        test_index = index[test_positions]
        buy_prob_series.loc[test_index] = filtered_test["MLBuyProb"].to_numpy(dtype=float)
        sell_prob_series.loc[test_index] = filtered_test["MLSellProb"].to_numpy(dtype=float)
        buy_approved_series.loc[test_index] = filtered_test["MLBuyApproved"].to_numpy(dtype=bool)
        sell_approved_series.loc[test_index] = filtered_test["MLSellApproved"].to_numpy(dtype=bool)
        model_mode_series.loc[test_index] = model_mode
        sell_veto_profile_series.loc[test_index] = threshold_selection.sell_veto_profile
        strategy_source_series.loc[test_index] = filtered_test["StrategySource"].to_numpy(dtype=object)
        selection_rows.append(
            {
                "Fold": fold_number,
                "TrainStart": index[train_positions[0]].date().isoformat(),
                "TrainEnd": index[train_positions[-1]].date().isoformat(),
                "TestStart": index[test_positions[0]].date().isoformat(),
                "TestEnd": index[test_positions[-1]].date().isoformat(),
                "ModelMode": model_mode,
                "BoostingMode": boosting_name,
                "BuyThreshold": threshold_selection.buy_threshold,
                "SellThreshold": threshold_selection.sell_threshold,
                "SellVetoProfile": threshold_selection.sell_veto_profile,
                "WinRate": threshold_selection.summary["win_rate"],
                "StrategyTotalReturn": threshold_selection.summary.get("strategy_total_return"),
                "BuyHoldReturn": threshold_selection.summary.get("buy_hold_return"),
                "ExcessReturn": threshold_selection.summary.get("excess_return"),
                "AvgClosedReturn": threshold_selection.summary["avg_closed_return"],
                "ClosedTrades": threshold_selection.summary["closed_trades"],
            }
        )

    full_buy_views, _, boosting_name = build_probability_views_for_target(
        feature_frame,
        label_frame["BuyCandidate"],
        label_frame["BuyLabel"],
        feature_frame.iloc[0:0],
        pd.Series(dtype=bool),
        feature_columns,
    )
    full_sell_views, _, _ = build_probability_views_for_target(
        feature_frame,
        label_frame["SellCandidate"],
        label_frame["SellLabel"],
        feature_frame.iloc[0:0],
        pd.Series(dtype=bool),
        feature_columns,
    )
    full_selection = select_ml_strategy_for_state_frame(symbol, baseline_state_frame, full_buy_views, full_sell_views, preset)
    if full_selection is None:
        final_state = attach_rule_only_columns(baseline_state_frame.copy())
        return {
            "enabled": False,
            "model_mode": "rules_only",
            "reason": "ML models could not fit candidate labels",
            "state_frame": final_state,
            "comparison": {},
            "selection_frame": pd.DataFrame(selection_rows),
            "oos_start": None,
        }

    final_model_mode, final_thresholds = full_selection
    final_filtered = apply_ml_thresholds_to_state_frame(
        baseline_state_frame,
        full_buy_views[final_model_mode],
        full_sell_views[final_model_mode],
        final_thresholds.buy_threshold,
        final_thresholds.sell_threshold,
        model_mode=final_model_mode,
        sell_veto_profile=final_thresholds.sell_veto_profile,
    )
    missing_buy = buy_prob_series.isna() & final_filtered["MLBuyProb"].notna()
    missing_sell = sell_prob_series.isna() & final_filtered["MLSellProb"].notna()
    buy_prob_series.loc[missing_buy] = final_filtered.loc[missing_buy, "MLBuyProb"]
    sell_prob_series.loc[missing_sell] = final_filtered.loc[missing_sell, "MLSellProb"]
    buy_approved_series.loc[missing_buy] = final_filtered.loc[missing_buy, "MLBuyApproved"]
    sell_approved_series.loc[missing_sell] = final_filtered.loc[missing_sell, "MLSellApproved"]
    model_mode_series.loc[model_mode_series.eq("rules_only")] = final_model_mode
    warmup_mask = strategy_source_series.eq("RuleOnlyWarmup")
    sell_veto_profile_series.loc[warmup_mask] = final_thresholds.sell_veto_profile
    strategy_source_series.loc[strategy_source_series.eq("RuleOnlyWarmup")] = final_filtered.loc[strategy_source_series.eq("RuleOnlyWarmup"), "StrategySource"]

    final_state = baseline_state_frame.copy()
    final_state["MLBuyProb"] = buy_prob_series
    final_state["MLSellProb"] = sell_prob_series
    final_state["MLBuyApproved"] = buy_approved_series
    final_state["MLSellApproved"] = sell_approved_series
    final_state["ModelMode"] = model_mode_series
    final_state["SellVetoProfile"] = sell_veto_profile_series
    final_state["StrategySource"] = strategy_source_series
    final_state = apply_ml_thresholds_to_state_frame(
        final_state,
        final_state["MLBuyProb"],
        final_state["MLSellProb"],
        final_thresholds.buy_threshold,
        final_thresholds.sell_threshold,
        model_mode=final_model_mode,
        sell_veto_profile=final_thresholds.sell_veto_profile,
    )

    oos_start = index[splits[0][1][0]]
    baseline_oos = attach_rule_only_columns(baseline_state_frame.loc[oos_start:].copy())
    final_oos = final_state.loc[oos_start:].copy()
    rule_summary = build_trade_replay_from_state_frame(symbol, baseline_oos, preset)[2]
    ml_summary = build_trade_replay_from_state_frame(symbol, final_oos, preset)[2]
    rule_total_return = rule_summary.get("strategy_total_return")
    ml_total_return = ml_summary.get("strategy_total_return")
    if (
        rule_total_return is not None
        and ml_total_return is not None
        and float(ml_total_return) <= float(rule_total_return)
    ):
        fallback_state = attach_rule_only_columns(baseline_state_frame.copy())
        return {
            "enabled": False,
            "model_mode": "rules_only",
            "reason": "Rule-only outperformed walk-forward ML; keeping rule strategy",
            "state_frame": fallback_state,
            "comparison": {
                "rule_summary": rule_summary,
                "ml_summary": ml_summary,
                "buy_threshold": final_thresholds.buy_threshold,
                "sell_threshold": final_thresholds.sell_threshold,
                "sell_veto_profile": final_thresholds.sell_veto_profile,
            },
            "selection_frame": pd.DataFrame(selection_rows),
            "oos_start": oos_start,
        }
    return {
        "enabled": True,
        "model_mode": final_model_mode,
        "reason": f"Walk-forward ML filter active ({boosting_name}, sell veto={final_thresholds.sell_veto_profile})",
        "state_frame": final_state,
        "comparison": {
            "rule_summary": rule_summary,
            "ml_summary": ml_summary,
            "buy_threshold": final_thresholds.buy_threshold,
            "sell_threshold": final_thresholds.sell_threshold,
            "sell_veto_profile": final_thresholds.sell_veto_profile,
        },
        "selection_frame": pd.DataFrame(selection_rows),
        "oos_start": oos_start,
    }


def analyze_symbol_state(
    ticker: str,
    payload: dict[str, Any],
    macro_data: dict[str, Any],
    preset: SignalPreset,
    *,
    feature_payloads: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    price_frame = payload.get("frame", pd.DataFrame())
    indicators = compute_indicators(
        price_frame,
        macro_data["score_series"],
        macro_data.get("benchmark_close"),
        macro_data.get("tips_13612w_momentum"),
        payload.get("resolved_symbol"),
    )
    baseline_state_frame = attach_rule_only_columns(build_state_frame(indicators, preset))
    ml_bundle = {
        "enabled": False,
        "model_mode": "rules_only",
        "reason": "Rule-only asset",
        "state_frame": baseline_state_frame,
        "comparison": {},
        "selection_frame": pd.DataFrame(),
        "oos_start": None,
    }
    if payload.get("resolved_symbol", ticker).upper() == ML_TARGET_SYMBOL and feature_payloads is not None:
        ml_bundle = build_ml_walkforward_bundle(ML_TARGET_SYMBOL, indicators, baseline_state_frame, feature_payloads, preset)
    state_frame = attach_rule_only_columns(ml_bundle.get("state_frame", baseline_state_frame))
    snapshot = build_snapshot(ticker, payload.get("resolved_symbol", ticker), state_frame, payload.get("warning"))
    return {
        "indicators": indicators,
        "baseline_state_frame": baseline_state_frame,
        "state_frame": state_frame,
        "snapshot": snapshot,
        "ml_bundle": ml_bundle,
    }


@st.cache_resource(show_spinner=False)
def load_cached_qqq_analysis(start_iso: str, end_iso: str, preset_name: str) -> dict[str, Any]:
    preset = PROFILE_PRESETS.get(preset_name, DEFAULT_SIGNAL_PRESET)
    macro_data = load_macro_fear_greed_cached(end_iso)
    feature_payloads = load_price_payloads(list(ML_BATCH_TICKERS), start_iso, end_iso)
    qqq_payload = feature_payloads.get(ML_TARGET_SYMBOL, {"frame": pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS)), "resolved_symbol": ML_TARGET_SYMBOL, "warning": "QQQ payload unavailable"})
    return analyze_symbol_state(
        ML_TARGET_SYMBOL,
        qqq_payload,
        macro_data,
        preset,
        feature_payloads=feature_payloads,
    )


def _coerce_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _coerce_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return False if pd.isna(value) else bool(value)


def classify_market_regime(row: Any) -> str:
    close_value = _coerce_float(row.get("Close"))
    ema50 = _coerce_float(row.get("EMA50"))
    ema200 = _coerce_float(row.get("EMA200"))
    trend_delta = _coerce_float(row.get("TrendDelta5"))
    adx = _coerce_float(row.get("ADX"))
    plus_di = _coerce_float(row.get("PlusDI"))
    minus_di = _coerce_float(row.get("MinusDI"))

    strong_trend = adx is not None and adx >= 23
    bull_context = all(value is not None for value in [close_value, ema50, ema200, trend_delta]) and bool(
        close_value >= ema200 and ema50 >= ema200 and trend_delta >= 0
    )
    bear_context = all(value is not None for value in [close_value, ema50, ema200, trend_delta]) and bool(
        close_value < ema200 and ema50 < ema200 and trend_delta < 0
    )
    plus_bias = plus_di is not None and minus_di is not None and plus_di >= minus_di
    minus_bias = plus_di is not None and minus_di is not None and minus_di > plus_di

    if bull_context and strong_trend and plus_bias:
        return "Bull Trend"
    if bear_context and strong_trend and minus_bias:
        return "Bear Trend"
    if bull_context:
        return "Bull Pullback"
    if bear_context:
        return "Bear Rally"
    return "Range / Transition"


def _is_bull_regime(regime: str) -> bool:
    return regime in {"Bull Trend", "Bull Pullback"}


def _is_bear_regime(regime: str) -> bool:
    return regime in {"Bear Trend", "Bear Rally"}


def detect_deep_value_confluence(row: Any) -> tuple[bool, bool]:
    cycle_score = _coerce_float(row_get(row, "CycleScore"))
    rsi = _coerce_float(row_get(row, "RSI"))
    buy_setup = int(row_get(row, "BuySetup", 0) or 0)
    buy_countdown = int(row_get(row, "BuyCountdown", 0) or 0)
    close_value = _coerce_float(row_get(row, "Close"))
    ema20 = _coerce_float(row_get(row, "EMA20"))
    ema50 = _coerce_float(row_get(row, "EMA50"))
    close_up_day = _coerce_bool(row_get(row, "CloseUpDay", False))

    base_overlap = bool(
        cycle_score is not None
        and cycle_score <= DEEP_VALUE_CYCLE_THRESHOLD
        and rsi is not None
        and rsi <= DEEP_VALUE_RSI_THRESHOLD
        and (buy_countdown == 13 or buy_setup >= DEEP_VALUE_SETUP_THRESHOLD)
    )
    ma_context = bool(
        close_up_day
        or (
            close_value is not None
            and ema20 is not None
            and close_value <= (ema20 * 1.02)
        )
        or (
            close_value is not None
            and ema50 is not None
            and close_value >= (ema50 * 0.98)
        )
    )
    extreme_overlap = bool(
        cycle_score is not None
        and cycle_score <= DEEP_VALUE_EXTREME_CYCLE_THRESHOLD
        and rsi is not None
        and rsi <= DEEP_VALUE_EXTREME_RSI_THRESHOLD
        and (
            buy_countdown == 13
            or buy_setup >= DEEP_VALUE_EXTREME_SETUP_THRESHOLD
            or close_up_day
        )
    )
    return base_overlap and ma_context, extreme_overlap


def detect_heat_trim_confluence(row: Any) -> tuple[bool, bool]:
    cycle_score = _coerce_float(row_get(row, "CycleScore"))
    rsi = _coerce_float(row_get(row, "RSI"))
    fear_greed = _coerce_float(row_get(row, "FearGreed"))
    sell_setup = int(row_get(row, "SellSetup", 0) or 0)
    sell_countdown = int(row_get(row, "SellCountdown", 0) or 0)
    close_value = _coerce_float(row_get(row, "Close"))
    ema20 = _coerce_float(row_get(row, "EMA20"))
    ema50 = _coerce_float(row_get(row, "EMA50"))

    ma_context = bool(
        close_value is not None
        and ema20 is not None
        and ema50 is not None
        and close_value >= ema20
        and ema20 >= ema50
    )
    base_overlap = bool(
        fear_greed is not None
        and fear_greed >= HEAT_TRIM_FEAR_THRESHOLD
        and rsi is not None
        and rsi >= HEAT_TRIM_RSI_THRESHOLD
        and (
            sell_countdown == 13
            or sell_setup >= HEAT_TRIM_SETUP_THRESHOLD
            or (cycle_score is not None and cycle_score >= 72.0)
        )
    )
    extreme_overlap = bool(
        fear_greed is not None
        and fear_greed >= HEAT_TRIM_EXTREME_FEAR_THRESHOLD
        and rsi is not None
        and rsi >= HEAT_TRIM_EXTREME_RSI_THRESHOLD
        and (
            sell_countdown == 13
            or sell_setup >= HEAT_TRIM_EXTREME_SETUP_THRESHOLD
            or (cycle_score is not None and cycle_score >= 80.0)
        )
    )
    return base_overlap and ma_context, extreme_overlap and ma_context


def detect_macro_trend_exit(row: Any) -> bool:
    fear_greed = _coerce_float(row_get(row, "FearGreed"))
    macd_bear_cross = _coerce_bool(row_get(row, "MACDBearCross", False))
    return bool(
        macd_bear_cross
        and fear_greed is not None
        and fear_greed < BUY_CUTOFFS["fear_reset"]
    )


def score_buy_signal(row: pd.Series) -> tuple[int, list[str]]:
    score = 0.0
    reasons: list[str] = []
    regime = classify_market_regime(row)
    bull_regime = _is_bull_regime(regime)
    bear_regime = _is_bear_regime(regime)
    reference_asset = _coerce_bool(row.get("ReferenceAsset"))

    cycle_score = _coerce_float(row.get("CycleScore"))
    buy_setup = int(row.get("BuySetup", 0) or 0)
    buy_countdown = int(row.get("BuyCountdown", 0) or 0)
    atr_stretch = _coerce_float(row.get("ATRStretch"))
    rsi = _coerce_float(row.get("RSI"))
    mfi = _coerce_float(row.get("MFI"))
    mfi_change = _coerce_float(row.get("MFIChange5"))
    fear_greed = _coerce_float(row.get("FearGreed"))
    tips_momentum = _coerce_float(row.get("TIPS13612WMomentum"))
    rs_percentile = _coerce_float(row.get("RSPercentile"))
    rs_momentum = _coerce_float(row.get("RSMomentum63"))
    volume_ratio = _coerce_float(row.get("VolumeRatio20"))
    cmf20 = _coerce_float(row.get("CMF20"))
    obv_slope10 = _coerce_float(row.get("OBVSlope10"))
    macd_hist = _coerce_float(row.get("MACDHist"))
    macd_delta = _coerce_float(row.get("MACDHistDelta"))
    macd_bull_cross = _coerce_bool(row.get("MACDBullCross"))
    macd_bear_cross = _coerce_bool(row.get("MACDBearCross"))
    bb_pos = _coerce_float(row.get("BBPos20"))
    bb_width_pct = _coerce_float(row.get("BBWidthPct"))
    volume_climax_up = _coerce_bool(row.get("VolumeClimaxUp"))
    volume_climax_down = _coerce_bool(row.get("VolumeClimaxDown"))
    volume_dry_up = _coerce_bool(row.get("VolumeDryUp"))
    close_value = _coerce_float(row.get("Close"))
    open_value = _coerce_float(row.get("Open"))
    ema20 = _coerce_float(row.get("EMA20"))
    deep_value_confluence, deep_value_extreme = detect_deep_value_confluence(row)
    buy_rsi_reset = bool(rsi is not None and rsi <= BUY_CUTOFFS["rsi_reset"])
    buy_rsi_extreme = bool(rsi is not None and rsi <= BUY_CUTOFFS["rsi_extreme"])
    buy_mfi_reset = bool(mfi is not None and mfi <= BUY_CUTOFFS["mfi_reset"])
    buy_mfi_extreme = bool(mfi is not None and mfi <= BUY_CUTOFFS["mfi_extreme"])
    buy_fear_reset = bool(fear_greed is not None and fear_greed <= BUY_CUTOFFS["fear_reset"])
    buy_fear_extreme = bool(fear_greed is not None and fear_greed <= BUY_CUTOFFS["fear_extreme"])
    buy_bb_low = bool(bb_pos is not None and bb_pos <= BUY_CUTOFFS["bb_low"])
    buy_bb_extreme = bool(bb_pos is not None and bb_pos <= 0.10)
    buy_cutoff_hits = int(sum([buy_rsi_reset, buy_mfi_reset, buy_fear_reset, buy_bb_low]))
    buy_extreme_hits = int(sum([buy_rsi_extreme, buy_mfi_extreme, buy_fear_extreme, buy_bb_extreme]))

    deep_reversal = cycle_score is not None and cycle_score <= 15 and buy_countdown == 13
    oversold_cluster = False
    dryup_reversal = bool(
        volume_dry_up
        and cycle_score is not None
        and cycle_score <= 25
        and rsi is not None
        and rsi <= 44
    )
    accumulation_flow = bool(
        volume_climax_up
        or (
            cmf20 is not None
            and cmf20 >= 0.05
            and obv_slope10 is not None
            and obv_slope10 > 0
        )
    )
    distribution_pressure = bool(
        volume_climax_down
        or (
            cmf20 is not None
            and cmf20 <= -0.05
            and obv_slope10 is not None
            and obv_slope10 < 0
        )
    )
    volume_support = bool(
        (volume_ratio is not None and volume_ratio >= 1.05 and close_value is not None and open_value is not None and close_value >= open_value)
        or (mfi_change is not None and mfi_change >= 5.0)
        or accumulation_flow
        or dryup_reversal
    )
    relative_support = bool(
        (rs_percentile is not None and rs_percentile >= (40.0 if reference_asset else 50.0))
        or (rs_momentum is not None and rs_momentum >= (-0.03 if reference_asset else -0.01))
    )
    tips_risk_off = bool(tips_momentum is not None and tips_momentum < 0.0)
    chop_zone = bool(bb_width_pct is not None and bb_width_pct <= 30)

    reversion_score = 0.0
    if cycle_score is not None:
        if cycle_score <= 15:
            reversion_score += 18
            reasons.append("STL capitulation low")
        elif cycle_score <= 25:
            reversion_score += 14
            reasons.append("STL cycle low")
        elif cycle_score <= 35:
            reversion_score += 9
            reasons.append("STL accumulation zone")
        elif cycle_score <= 45:
            reversion_score += 5
            reasons.append("STL early reset")

    if atr_stretch is not None:
        if atr_stretch <= -1.8:
            reversion_score += 10
            oversold_cluster = True
            reasons.append("ATR deep pullback")
        elif atr_stretch <= -1.1:
            reversion_score += 7
            oversold_cluster = True
            reasons.append("ATR pullback")
        elif atr_stretch <= -0.6:
            reversion_score += 3

    if buy_rsi_extreme:
        reversion_score += 10
        oversold_cluster = True
        reasons.append("RSI washed out")
    elif buy_rsi_reset:
        reversion_score += 7
        oversold_cluster = True
        reasons.append("RSI reset")

    if buy_mfi_extreme:
        reversion_score += 10
        oversold_cluster = True
        reasons.append("MFI capitulation")
    elif buy_mfi_reset:
        reversion_score += 7
        oversold_cluster = True
        reasons.append("MFI oversold")

    if buy_bb_extreme:
        reversion_score += 10
        oversold_cluster = True
        reasons.append("Bollinger lower-tag")
    elif buy_bb_low:
        reversion_score += 7
        oversold_cluster = True
        reasons.append("Bollinger low-zone")

    if reference_asset and cycle_score is not None and rsi is not None:
        if cycle_score <= 20 and rsi <= 45:
            reversion_score += 12
            reasons.append("Reference ETF washout")
        elif cycle_score <= 30 and rsi <= 48:
            reversion_score += 8
            reasons.append("ETF low-zone reset")

    if deep_value_confluence:
        reversion_score += 12 if deep_value_extreme else 8
        reasons.append("STL+TD+RSI washout")

    reversion_cap = 28 if regime == "Range / Transition" else 24 if bull_regime else 20
    if reference_asset:
        reversion_cap += 8
    reversion_score = min(reversion_cap, reversion_score)
    score += weighted_component(reversion_score, BUY_COMPONENT_WEIGHTS["reversion"])

    td_score = 0.0
    if buy_countdown == 13:
        td_score = 24
        reasons.append("TD buy countdown 13")
    elif buy_setup >= 9:
        td_score = 18
        reasons.append(f"TD buy setup {buy_setup}")
    elif buy_setup >= 7:
        td_score = 10
        reasons.append(f"TD buy setup {buy_setup}")
    elif reference_asset and buy_setup >= 5 and cycle_score is not None and cycle_score <= 35:
        td_score = 8
        reasons.append(f"TD early buy setup {buy_setup}")
    score += weighted_component(td_score, BUY_COMPONENT_WEIGHTS["td"])

    macro_score = 0.0
    if buy_fear_extreme:
        macro_score += 8
        reasons.append("Macro fear tailwind")
    elif buy_fear_reset:
        macro_score += 4
        reasons.append("Macro risk reset")
    if tips_risk_off:
        if deep_reversal:
            macro_score -= 2
        elif tips_momentum is not None and tips_momentum <= -0.03:
            macro_score -= 8
            reasons.append("TIPS 13612W risk-off")
        else:
            macro_score -= 5
            reasons.append("TIPS macro headwind")
    score += weighted_component(macro_score, BUY_COMPONENT_WEIGHTS["macro"])

    relative_score = 0.0
    if relative_support:
        relative_score += 8 if rs_percentile is not None and rs_percentile >= 60 else 5
        reasons.append("Relative strength supportive")
    elif rs_percentile is not None and rs_percentile < 25:
        relative_score -= 10
        reasons.append("Relative strength weak")
    elif rs_percentile is not None and rs_percentile < 40:
        relative_score -= 4
    score += weighted_component(relative_score, BUY_COMPONENT_WEIGHTS["relative"])

    volume_score = 0.0
    if accumulation_flow:
        volume_score += 9
        reasons.append("Accumulation flow")
    elif dryup_reversal:
        volume_score += 5
        reasons.append("Volume dry-up reset")
    elif volume_support:
        volume_score += 7
        reasons.append("Volume-backed reversal")
    elif mfi is not None and mfi < 25 and mfi_change is not None and mfi_change < 0 and bear_regime:
        volume_score -= 8
        reasons.append("Falling knife flow")
    if distribution_pressure and not deep_reversal:
        volume_score -= 8
        reasons.append("Distribution pressure")
    score += weighted_component(volume_score, BUY_COMPONENT_WEIGHTS["volume"])

    macd_score = 0.0
    if macd_bull_cross:
        macd_score += 12 if macd_hist is not None and macd_hist <= 0 else 9
        reasons.append("MACD bullish cross")
    elif macd_delta is not None:
        if macd_delta > 0 and macd_hist is not None and macd_hist <= 0:
            macd_score += 8
            reasons.append("MACD momentum turn")
        elif macd_delta > 0 and macd_hist is not None and macd_hist > 0:
            macd_score += 10
            reasons.append("MACD bullish expansion")
        elif macd_delta < 0 and bear_regime and not deep_reversal:
            macd_score -= 5
            reasons.append("MACD still falling")
    if macd_bear_cross and not deep_reversal:
        macd_score -= 6
        reasons.append("MACD bear cross")
    score += weighted_component(macd_score, BUY_COMPONENT_WEIGHTS["macd"])

    regime_score = 0.0
    if chop_zone:
        if macd_bull_cross or accumulation_flow or deep_reversal:
            regime_score += 3
            reasons.append("Squeeze reversal support")
        else:
            regime_score -= 7
            reasons.append("Bollinger squeeze chop")

    if bull_regime:
        regime_score += 8
        reasons.append("Bull regime support")
    elif regime == "Range / Transition":
        regime_score += 3
    elif not deep_reversal:
        regime_score -= 6 if reference_asset else 12
        reasons.append("Bear regime penalty")

    if bull_regime and atr_stretch is not None and atr_stretch <= -0.5 and close_value is not None and ema20 is not None and close_value >= ema20:
        regime_score += 6
        reasons.append("Trend dip setup")
    score += weighted_component(regime_score, BUY_COMPONENT_WEIGHTS["regime"])

    confluence_count = int(sum(
        [
            bool(cycle_score is not None and cycle_score <= 35),
            bool(buy_setup >= 8 or buy_countdown == 13),
            oversold_cluster,
            accumulation_flow or volume_support,
            relative_support,
            bool(buy_cutoff_hits >= 2 or buy_fear_reset),
            bull_regime,
            macd_bull_cross,
            deep_value_confluence,
        ]
    ))
    if confluence_count >= 3:
        confluence_bonus = min(20.0, 6.0 + ((confluence_count - 3) * 4.0))
        score += weighted_component(confluence_bonus, BUY_COMPONENT_WEIGHTS["confluence"])
        reasons.append(f"{confluence_count}x buy confluence")
    if buy_extreme_hits >= 2:
        score += weighted_component(8.0, BUY_COMPONENT_WEIGHTS["confluence"])
        reasons.append("Extreme oversold cutoffs")
    elif buy_cutoff_hits >= 3:
        score += weighted_component(5.0, BUY_COMPONENT_WEIGHTS["confluence"])
        reasons.append("Multi-cutoff washout")

    if _coerce_bool(row.get("BuyTriggerPrice")):
        score += weighted_component(6.0, BUY_COMPONENT_WEIGHTS["confirmation"])
        reasons.append("Price confirmation")

    if bear_regime and not deep_reversal:
        score = min(score, 70.0 if reference_asset else 62.0)
    return max(0, min(100, int(round(score)))), reasons[:5]


def score_sell_signal(row: pd.Series) -> tuple[int, list[str]]:
    score = 0.0
    reasons: list[str] = []
    regime = classify_market_regime(row)
    bull_regime = _is_bull_regime(regime)
    bear_regime = _is_bear_regime(regime)
    reference_asset = _coerce_bool(row.get("ReferenceAsset"))

    cycle_score = _coerce_float(row.get("CycleScore"))
    sell_setup = int(row.get("SellSetup", 0) or 0)
    sell_countdown = int(row.get("SellCountdown", 0) or 0)
    atr_stretch = _coerce_float(row.get("ATRStretch"))
    rsi = _coerce_float(row.get("RSI"))
    mfi = _coerce_float(row.get("MFI"))
    mfi_change = _coerce_float(row.get("MFIChange5"))
    fear_greed = _coerce_float(row.get("FearGreed"))
    tips_momentum = _coerce_float(row.get("TIPS13612WMomentum"))
    rs_percentile = _coerce_float(row.get("RSPercentile"))
    rs_momentum = _coerce_float(row.get("RSMomentum63"))
    volume_ratio = _coerce_float(row.get("VolumeRatio20"))
    cmf20 = _coerce_float(row.get("CMF20"))
    obv_slope10 = _coerce_float(row.get("OBVSlope10"))
    macd_hist = _coerce_float(row.get("MACDHist"))
    macd_delta = _coerce_float(row.get("MACDHistDelta"))
    macd_bull_cross = _coerce_bool(row.get("MACDBullCross"))
    macd_bear_cross = _coerce_bool(row.get("MACDBearCross"))
    bb_pos = _coerce_float(row.get("BBPos20"))
    bb_width_pct = _coerce_float(row.get("BBWidthPct"))
    volume_climax_up = _coerce_bool(row.get("VolumeClimaxUp"))
    volume_climax_down = _coerce_bool(row.get("VolumeClimaxDown"))
    close_value = _coerce_float(row.get("Close"))
    open_value = _coerce_float(row.get("Open"))
    ema20 = _coerce_float(row.get("EMA20"))
    ema50 = _coerce_float(row.get("EMA50"))
    stop_touched = _coerce_bool(row.get("StopTouched"))
    stop_close_breach = _coerce_bool(row.get("StopCloseBreach"))
    sell_trigger_price = _coerce_bool(row.get("SellTriggerPrice"))
    heat_trim_confluence, heat_trim_extreme = detect_heat_trim_confluence(row)
    ma_overheat = bool(
        close_value is not None
        and ema20 is not None
        and ema50 is not None
        and close_value >= ema20
        and ema20 >= ema50
    )
    sell_rsi_hot = bool(rsi is not None and rsi >= SELL_CUTOFFS["rsi_hot"])
    sell_rsi_extreme = bool(rsi is not None and rsi >= SELL_CUTOFFS["rsi_extreme"])
    sell_mfi_hot = bool(mfi is not None and mfi >= SELL_CUTOFFS["mfi_hot"])
    sell_mfi_extreme = bool(mfi is not None and mfi >= SELL_CUTOFFS["mfi_extreme"])
    sell_greed_hot = bool(fear_greed is not None and fear_greed >= SELL_CUTOFFS["greed_hot"])
    sell_greed_extreme = bool(fear_greed is not None and fear_greed >= SELL_CUTOFFS["greed_extreme"])
    sell_bb_hot = bool(bb_pos is not None and bb_pos >= SELL_CUTOFFS["bb_high"])
    sell_bb_extreme = bool(bb_pos is not None and bb_pos >= SELL_CUTOFFS["bb_extreme"])
    sell_cutoff_hits = int(sum([sell_rsi_hot, sell_mfi_hot, sell_greed_hot, sell_bb_hot]))
    sell_extreme_hits = int(sum([sell_rsi_extreme, sell_mfi_extreme, sell_greed_extreme, sell_bb_extreme]))
    panic_low = bool(
        cycle_score is not None
        and rsi is not None
        and (
            (cycle_score <= 15 and rsi <= 40)
            or (cycle_score <= 25 and rsi <= 35)
        )
    )
    tips_risk_off = bool(tips_momentum is not None and tips_momentum < 0.0)
    accumulation_support = bool(
        (cmf20 is not None and cmf20 >= 0.05 and obv_slope10 is not None and obv_slope10 > 0)
        or volume_climax_up
    )
    blowoff_risk = bool(
        volume_climax_up
        and bb_pos is not None
        and bb_pos >= 0.90
        and (
            (rsi is not None and rsi >= 66)
            or (atr_stretch is not None and atr_stretch >= 1.1)
        )
    )
    chop_zone = bool(bb_width_pct is not None and bb_width_pct <= 30)

    exhaustion_score = 0.0
    if cycle_score is not None:
        if cycle_score >= 85:
            exhaustion_score += 16
            reasons.append("STL overheated")
        elif cycle_score >= 72:
            exhaustion_score += 12
            reasons.append("STL extended")
        elif cycle_score >= 60:
            exhaustion_score += 7
            reasons.append("STL rich")
    score += weighted_component(exhaustion_score, SELL_COMPONENT_WEIGHTS["exhaustion"])

    td_score = 0.0
    if sell_countdown == 13:
        td_score = 18
        reasons.append("TD sell countdown 13")
    elif sell_setup >= 9:
        td_score = 12
        reasons.append(f"TD sell setup {sell_setup}")
    elif sell_setup >= 7:
        td_score = 8
        reasons.append(f"TD sell setup {sell_setup}")
    score += weighted_component(td_score, SELL_COMPONENT_WEIGHTS["td"])

    heat_score = 0.0
    heat_signal = False
    if atr_stretch is not None:
        if atr_stretch >= 1.8:
            heat_score += 12
            heat_signal = True
            reasons.append("ATR overextension")
        elif atr_stretch >= 1.1:
            heat_score += 8
            heat_signal = True
            reasons.append("ATR stretched")
        elif atr_stretch >= 0.7:
            heat_score += 4

    if sell_rsi_extreme:
        heat_score += 10 if (ma_overheat or bear_regime) else 5
        heat_signal = True
        reasons.append("RSI overheated")
    elif sell_rsi_hot:
        heat_score += 6 if (ma_overheat or bear_regime) else 3
        heat_signal = True
        reasons.append("RSI hot")

    if sell_mfi_extreme:
        heat_score += 8 if (ma_overheat or bear_regime) else 4
        heat_signal = True
        reasons.append("MFI euphoric")
    elif sell_mfi_hot:
        heat_score += 4 if (ma_overheat or bear_regime) else 2
        heat_signal = True
        reasons.append("MFI rich")

    if sell_bb_extreme:
        heat_score += 8 if (ma_overheat or bear_regime) else 4
        heat_signal = True
        reasons.append("Bollinger upper-tag")
    elif sell_bb_hot:
        heat_score += 4 if (ma_overheat or bear_regime) else 2
        heat_signal = True
        reasons.append("Bollinger hot-zone")

    if heat_trim_confluence:
        heat_score += 8 if heat_trim_extreme else 5
        heat_signal = True
        reasons.append("Fear/RSI/MA trim cluster")

    heat_score = min(18.0, heat_score)
    score += weighted_component(heat_score, SELL_COMPONENT_WEIGHTS["heat"])

    distribution_score = 0.0
    distribution_flow = bool(
        (volume_ratio is not None and volume_ratio >= 1.05 and close_value is not None and open_value is not None and close_value < open_value)
        or (mfi_change is not None and mfi_change <= -5.0 and mfi is not None and mfi >= 60.0)
        or volume_climax_down
        or (
            cmf20 is not None
            and cmf20 <= -0.05
            and obv_slope10 is not None
            and obv_slope10 < 0
        )
    )
    if distribution_flow:
        distribution_score += 8
        reasons.append("Distribution flow")
    if blowoff_risk:
        distribution_score += 6
        reasons.append("Blowoff volume")
    if sell_greed_extreme:
        distribution_score += 8 if (ma_overheat or bear_regime) else 3
        reasons.append("Macro greed risk")
    elif sell_greed_hot:
        distribution_score += 4 if (ma_overheat or bear_regime) else 1
        reasons.append("Macro greed warning")
    if tips_risk_off:
        if tips_momentum is not None and tips_momentum <= -0.03:
            distribution_score += 8
            reasons.append("TIPS 13612W negative")
        else:
            distribution_score += 5
            reasons.append("TIPS risk-off backdrop")
    if rs_percentile is not None and rs_percentile <= 25:
        distribution_score += 8
        reasons.append("Relative strength breakdown")
    elif rs_percentile is not None and rs_percentile <= 40:
        distribution_score += 4
    if rs_momentum is not None and rs_momentum <= -0.03:
        distribution_score += 6
    if macd_bear_cross:
        distribution_score += 10 if macd_hist is not None and macd_hist >= 0 else 8
        reasons.append("MACD bear cross")
    elif macd_delta is not None:
        if macd_delta < 0 and macd_hist is not None and macd_hist <= 0:
            distribution_score += 8
            reasons.append("MACD breakdown")
        elif macd_delta < 0 and macd_hist is not None and macd_hist > 0:
            distribution_score += 4
            reasons.append("MACD rollover")
    if macd_bull_cross and bull_regime and not stop_close_breach:
        distribution_score -= 6
        reasons.append("MACD support")
    if sell_trigger_price:
        distribution_score += 12
        reasons.append("Price breakdown")
    if stop_close_breach:
        distribution_score += 10 if reference_asset and panic_low else 16
        reasons.append("Trailing stop close breach")
    elif stop_touched:
        distribution_score += 4
        reasons.append("Intraday stop probe")
    if bear_regime:
        distribution_score += 12
        reasons.append("Bear regime pressure")
    if accumulation_support and not stop_close_breach and not bear_regime:
        distribution_score -= 6
        reasons.append("Accumulation support")
    if chop_zone:
        if distribution_flow or macd_bear_cross or stop_close_breach:
            distribution_score += 3
            reasons.append("Squeeze breakdown risk")
        else:
            distribution_score -= 7
            reasons.append("Bollinger squeeze chop")
    score += weighted_component(distribution_score, SELL_COMPONENT_WEIGHTS["distribution"])

    confluence_count = int(sum(
        [
            bool(cycle_score is not None and cycle_score >= 72),
            bool(sell_setup >= 8 or sell_countdown == 13),
            heat_signal,
            distribution_flow,
            sell_trigger_price,
            stop_close_breach,
            bear_regime,
            macd_bear_cross or sell_cutoff_hits >= 2,
            heat_trim_confluence,
        ]
    ))
    if confluence_count >= 3:
        confluence_bonus = min(16.0, 5.0 + ((confluence_count - 3) * 3.0))
        score += weighted_component(confluence_bonus, SELL_COMPONENT_WEIGHTS["confluence"])
        reasons.append(f"{confluence_count}x sell confluence")
    if sell_extreme_hits >= 2:
        score += weighted_component(6.0, SELL_COMPONENT_WEIGHTS["confluence"])
        reasons.append("Extreme hot cutoffs")
    elif sell_cutoff_hits >= 3:
        score += weighted_component(4.0, SELL_COMPONENT_WEIGHTS["confluence"])
        reasons.append("Multi-cutoff heat")

    if bull_regime and not stop_close_breach and not sell_trigger_price:
        score = min(score, 56.0 if sell_cutoff_hits < 2 else 62.0)
        if rs_percentile is not None and rs_percentile >= 55:
            score -= 6
        reasons.append("Bull trend trim bias")

    if reference_asset and panic_low and not stop_close_breach and not sell_trigger_price:
        score = min(score, 32.0)
        reasons.append("Capitulation sell cap")

    return max(0, min(100, int(round(score)))), reasons[:5]


def build_signal_state(row: Any, preset: SignalPreset) -> dict[str, Any]:
    buy_score, buy_reasons = score_buy_signal(row)
    sell_score, sell_reasons = score_sell_signal(row)
    regime = classify_market_regime(row)
    bull_regime = _is_bull_regime(regime)
    bear_regime = _is_bear_regime(regime)
    reference_asset = _coerce_bool(row.get("ReferenceAsset"))

    close_value = _coerce_float(row.get("Close"))
    open_value = _coerce_float(row.get("Open"))
    cycle_score = _coerce_float(row.get("CycleScore"))
    rsi = _coerce_float(row.get("RSI"))
    mfi = _coerce_float(row.get("MFI"))
    buy_countdown = int(row.get("BuyCountdown", 0) or 0)
    fear_greed = _coerce_float(row.get("FearGreed"))
    tips_momentum = _coerce_float(row.get("TIPS13612WMomentum"))
    rs_percentile = _coerce_float(row.get("RSPercentile"))
    volume_ratio = _coerce_float(row.get("VolumeRatio20"))
    cmf20 = _coerce_float(row.get("CMF20"))
    obv_slope10 = _coerce_float(row.get("OBVSlope10"))
    mfi_change = _coerce_float(row.get("MFIChange5"))
    macd_delta = _coerce_float(row.get("MACDHistDelta"))
    macd_bull_cross = _coerce_bool(row.get("MACDBullCross"))
    macd_bear_cross = _coerce_bool(row.get("MACDBearCross"))
    bb_pos = _coerce_float(row.get("BBPos20"))
    bb_width_pct = _coerce_float(row.get("BBWidthPct"))
    volume_climax_up = _coerce_bool(row.get("VolumeClimaxUp"))
    volume_climax_down = _coerce_bool(row.get("VolumeClimaxDown"))
    volume_dry_up = _coerce_bool(row.get("VolumeDryUp"))
    stop_touched = _coerce_bool(row.get("StopTouched"))
    stop_close_breach = _coerce_bool(row.get("StopCloseBreach"))
    stop_price = _coerce_float(row.get("ChandelierStop"))
    atr_value = _coerce_float(row.get("ATR"))
    buy_trigger_price = _coerce_bool(row.get("BuyTriggerPrice"))
    sell_trigger_price = _coerce_bool(row.get("SellTriggerPrice"))
    close_up_day = _coerce_bool(row.get("CloseUpDay"))
    close_down_day = _coerce_bool(row.get("CloseDownDay"))
    buy_rsi_cutoff = bool(rsi is not None and rsi <= BUY_CUTOFFS["rsi_reset"])
    buy_rsi_extreme = bool(rsi is not None and rsi <= BUY_CUTOFFS["rsi_extreme"])
    buy_mfi_cutoff = bool(mfi is not None and mfi <= BUY_CUTOFFS["mfi_reset"])
    buy_mfi_extreme = bool(mfi is not None and mfi <= BUY_CUTOFFS["mfi_extreme"])
    buy_fear_cutoff = bool(fear_greed is not None and fear_greed <= BUY_CUTOFFS["fear_reset"])
    buy_fear_extreme = bool(fear_greed is not None and fear_greed <= BUY_CUTOFFS["fear_extreme"])
    buy_bb_cutoff = bool(bb_pos is not None and bb_pos <= BUY_CUTOFFS["bb_low"])
    buy_bb_extreme = bool(bb_pos is not None and bb_pos <= 0.10)
    buy_cutoff_hits = int(sum([buy_rsi_cutoff, buy_mfi_cutoff, buy_fear_cutoff, buy_bb_cutoff]))
    buy_extreme_hits = int(sum([buy_rsi_extreme, buy_mfi_extreme, buy_fear_extreme, buy_bb_extreme]))
    sell_rsi_cutoff = bool(rsi is not None and rsi >= SELL_CUTOFFS["rsi_hot"])
    sell_rsi_extreme = bool(rsi is not None and rsi >= SELL_CUTOFFS["rsi_extreme"])
    sell_mfi_cutoff = bool(mfi is not None and mfi >= SELL_CUTOFFS["mfi_hot"])
    sell_mfi_extreme = bool(mfi is not None and mfi >= SELL_CUTOFFS["mfi_extreme"])
    sell_greed_cutoff = bool(fear_greed is not None and fear_greed >= SELL_CUTOFFS["greed_hot"])
    sell_greed_extreme = bool(fear_greed is not None and fear_greed >= SELL_CUTOFFS["greed_extreme"])
    sell_bb_cutoff = bool(bb_pos is not None and bb_pos >= SELL_CUTOFFS["bb_high"])
    sell_bb_extreme = bool(bb_pos is not None and bb_pos >= SELL_CUTOFFS["bb_extreme"])
    sell_cutoff_hits = int(sum([sell_rsi_cutoff, sell_mfi_cutoff, sell_greed_cutoff, sell_bb_cutoff]))
    sell_extreme_hits = int(sum([sell_rsi_extreme, sell_mfi_extreme, sell_greed_extreme, sell_bb_extreme]))
    deep_value_confluence, deep_value_extreme = detect_deep_value_confluence(row)
    heat_trim_confluence, heat_trim_extreme = detect_heat_trim_confluence(row)
    macro_trend_exit = detect_macro_trend_exit(row)

    deep_reversal = bool(cycle_score is not None and cycle_score <= 15 and buy_countdown == 13)
    panic_low = bool(
        cycle_score is not None
        and rsi is not None
        and (
            (cycle_score <= 15 and rsi <= 40)
            or (cycle_score <= 25 and rsi <= 35)
        )
    )
    dryup_reversal = bool(
        volume_dry_up
        and cycle_score is not None
        and cycle_score <= 25
        and rsi is not None
        and rsi <= 44
    )
    accumulation_flow = bool(
        volume_climax_up
        or (
            cmf20 is not None
            and cmf20 >= 0.03
            and obv_slope10 is not None
            and obv_slope10 > 0
        )
    )
    distribution_flow = bool(
        volume_climax_down
        or (
            cmf20 is not None
            and cmf20 <= -0.03
            and obv_slope10 is not None
            and obv_slope10 < 0
        )
    )
    flow_conflict = distribution_flow and not accumulation_flow
    volume_support = bool(
        (volume_ratio is not None and volume_ratio >= 1.05 and close_value is not None and open_value is not None and close_value >= open_value)
        or (mfi_change is not None and mfi_change >= 5.0)
        or accumulation_flow
        or dryup_reversal
    )
    relative_support = bool(rs_percentile is not None and rs_percentile >= (40.0 if reference_asset else 50.0))
    bullish_close = bool(close_value is not None and open_value is not None and close_value >= open_value)
    tips_risk_off = bool(tips_momentum is not None and tips_momentum < 0.0)
    chop_zone = bool(bb_width_pct is not None and bb_width_pct <= 30)
    reference_washout = bool(
        reference_asset
        and cycle_score is not None
        and cycle_score <= 30
        and buy_cutoff_hits >= 1
    )
    strong_trend_dip = bool(
        bull_regime
        and cycle_score is not None
        and cycle_score <= 38
        and (buy_cutoff_hits >= 1 or cycle_score <= 25)
        and volume_support
        and (macd_bull_cross or buy_trigger_price)
        and not flow_conflict
    )
    buy_cutoff_ready = bool(
        deep_reversal
        or reference_washout
        or buy_extreme_hits >= 1
        or (cycle_score is not None and cycle_score <= 35 and buy_cutoff_hits >= 1)
        or (
            cycle_score is not None
            and cycle_score <= 45
            and buy_cutoff_hits >= 2
            and (volume_support or macd_bull_cross or bull_regime)
        )
        or strong_trend_dip
        or deep_value_confluence
    )
    sell_capitulation_block = bool(
        deep_reversal
        or deep_value_extreme
        or buy_extreme_hits >= 2
        or (cycle_score is not None and cycle_score <= 30 and buy_cutoff_hits >= 2)
    )
    sell_cutoff_ready = bool(
        sell_trigger_price
        or sell_extreme_hits >= 2
        or (stop_close_breach and not sell_capitulation_block)
        or (
            bull_regime
            and cycle_score is not None
            and cycle_score >= 72
            and sell_cutoff_hits >= 2
            and (distribution_flow or macd_bear_cross or tips_risk_off or stop_touched)
        )
        or (
            not bull_regime
            and cycle_score is not None
            and cycle_score >= 68
            and sell_cutoff_hits >= 1
            and (distribution_flow or macd_bear_cross or bear_regime or tips_risk_off or stop_touched)
        )
    )

    buy_setup_threshold = preset.watch_threshold
    if bull_regime and relative_support:
        buy_setup_threshold -= 3
    if reference_asset:
        buy_setup_threshold -= 5
    if buy_cutoff_hits >= 2 or deep_reversal:
        buy_setup_threshold -= 2
    if deep_value_confluence:
        buy_setup_threshold -= 4
    if tips_risk_off and not deep_reversal:
        buy_setup_threshold += 2
    if chop_zone and not (macd_bull_cross or accumulation_flow or deep_reversal):
        buy_setup_threshold += 3
    if flow_conflict and not deep_reversal:
        buy_setup_threshold += 4
    buy_setup_active = (
        (
            buy_score >= buy_setup_threshold
            and buy_cutoff_ready
            and (buy_score >= sell_score - 8 or deep_reversal)
        )
        or reference_washout
        or strong_trend_dip
    )
    sell_setup_threshold = preset.watch_threshold + (4 if reference_asset else 0)
    if sell_cutoff_hits == 0 and not stop_close_breach and not sell_trigger_price:
        sell_setup_threshold += 4
    if chop_zone and not (macd_bear_cross or distribution_flow or stop_close_breach):
        sell_setup_threshold += 3
    sell_setup_active = bool(
        (not sell_capitulation_block)
        and (
            stop_close_breach
            or (sell_score >= sell_setup_threshold and sell_cutoff_ready and sell_score >= buy_score - 2 and not panic_low)
        )
    )
    stop_exit_confirmed, stop_emergency_exit, stop_gap_atr = evaluate_stop_exit(
        close_value,
        stop_price,
        atr_value,
        sell_score=sell_score,
        buy_score=buy_score,
        bearish_context=bear_regime,
        sell_context=bool(sell_setup_active or sell_trigger_price),
        hold_lock_active=False,
        macro_risk_off=tips_risk_off,
    )
    buy_trigger = bool(
        buy_setup_active
        and not stop_exit_confirmed
        and (buy_cutoff_ready or strong_trend_dip)
        and (not chop_zone or macd_bull_cross or accumulation_flow or deep_reversal or buy_trigger_price)
        and (not flow_conflict or deep_reversal or macd_bull_cross)
        and (
            buy_trigger_price
            or macd_bull_cross
            or (deep_reversal and bullish_close and (volume_support or relative_support))
            or (
                close_up_day
                and buy_score >= max(preset.watch_threshold + 4, sell_score + 6)
                and (
                    accumulation_flow
                    or (volume_support and not flow_conflict)
                    or deep_reversal
                    or bull_regime
                    or (macd_delta is not None and macd_delta > 0)
                )
            )
            or (bull_regime and buy_score >= max(preset.watch_threshold, preset.strong_threshold - 8) and bullish_close and volume_support and not flow_conflict)
            or (deep_value_confluence and (buy_trigger_price or macd_bull_cross or close_up_day))
        )
    )
    sell_trigger = bool(
        stop_emergency_exit
        or (
            sell_setup_active
            and sell_cutoff_ready
            and (not chop_zone or macd_bear_cross or distribution_flow or stop_exit_confirmed or sell_trigger_price)
            and (
                stop_exit_confirmed
                or sell_trigger_price
                or macd_bear_cross
                or (close_down_day and sell_score >= max(preset.watch_threshold + 6, buy_score + 8) and bear_regime)
            )
        )
    )
    trim_ready = bool(
        sell_setup_active
        and sell_cutoff_ready
        and not sell_capitulation_block
        and not sell_trigger
        and not stop_exit_confirmed
        and (not chop_zone or macd_bear_cross or distribution_flow)
        and (
            bull_regime
            or (rs_percentile is not None and rs_percentile >= 50.0)
            or sell_greed_cutoff
            or heat_trim_confluence
        )
    )

    buy_strong_threshold = preset.strong_threshold - 5 if bull_regime and relative_support else preset.strong_threshold
    sell_strong_threshold = preset.strong_threshold + 6 if bull_regime and not stop_exit_confirmed else preset.strong_threshold

    if stop_emergency_exit:
        state = "Strong Sell"
        key_reasons = sell_reasons
    elif reference_asset and panic_low and buy_setup_active and not sell_trigger:
        if buy_trigger and buy_score >= preset.strong_threshold - 6:
            state = "Strong Buy"
        elif buy_trigger or buy_score >= preset.watch_threshold + 2:
            state = "Buy"
        else:
            state = "Weak Buy"
        key_reasons = buy_reasons
    elif sell_trigger and (stop_emergency_exit or sell_score >= max(sell_strong_threshold, buy_score + 4)):
        state = "Strong Sell"
        key_reasons = sell_reasons
    elif sell_trigger and sell_score >= max(preset.watch_threshold + 8, buy_score + 6):
        state = "Sell"
        key_reasons = sell_reasons
    elif buy_trigger and buy_score >= max(preset.watch_threshold + 10, buy_strong_threshold) and buy_score >= sell_score - 3:
        state = "Strong Buy"
        key_reasons = buy_reasons
    elif buy_trigger and buy_score >= max(preset.watch_threshold + 2, sell_score):
        state = "Buy"
        key_reasons = buy_reasons
    elif buy_setup_active:
        state = "Weak Buy"
        key_reasons = buy_reasons
    elif trim_ready or (sell_setup_active and sell_score > buy_score + 2):
        state = "Weak Sell"
        key_reasons = sell_reasons
    else:
        state = "Hold / Neutral"
        key_reasons = buy_reasons[:2] + sell_reasons[:2]

    if bull_regime and heat_trim_confluence and not macro_trend_exit and state in {"Strong Sell", "Sell"}:
        state = "Weak Sell"
        sell_trigger = False
        key_reasons = sell_reasons

    td_label = str(row.get("TDLabel", "Unavailable"))
    cycle_text = f"Cycle {_format_float(cycle_score, 1)}" if cycle_score is not None else "Cycle n/a"
    fear_text = f"Fear & Greed {_format_float(fear_greed, 0)}" if fear_greed is not None else "Fear & Greed n/a"
    reason_text = ", ".join(key_reasons[:3]) if key_reasons else "No dominant edge"
    note = (
        f"{state} | {regime} | B{buy_score}/S{sell_score} | "
        f"Setup {int(buy_setup_active)}/{int(sell_setup_active)} | Trigger {int(buy_trigger)}/{int(sell_trigger)} | "
        f"{cycle_text} | {td_label} | {fear_text} | {reason_text}"
    )

    return {
        "State": state,
        "Regime": regime,
        "BuyScore": buy_score,
        "SellScore": sell_score,
        "BuySetupActive": buy_setup_active,
        "SellSetupActive": sell_setup_active,
        "BuyTrigger": buy_trigger,
        "SellTrigger": sell_trigger,
        "StopExitConfirmed": stop_exit_confirmed,
        "StopEmergencyExit": stop_emergency_exit,
        "StopGapATR": stop_gap_atr,
        "SignalNote": note,
        "BuyReasonText": " | ".join(buy_reasons),
        "SellReasonText": " | ".join(sell_reasons),
        "DeepValueConfluence": deep_value_confluence,
        "DeepValueExtreme": deep_value_extreme,
        "HeatTrimConfluence": heat_trim_confluence,
        "HeatTrimExtreme": heat_trim_extreme,
        "MacroTrendExit": macro_trend_exit,
    }


def _compose_signal_note(row: pd.Series, state: str) -> str:
    buy_score = int(row.get("BuyScore", 0) or 0)
    sell_score = int(row.get("SellScore", 0) or 0)
    regime = str(row.get("Regime", "n/a"))
    td_label = str(row.get("TDLabel", "Unavailable"))
    cycle_score = _coerce_float(row.get("CycleScore"))
    fear_greed = _coerce_float(row.get("FearGreed"))
    if state in {"Strong Buy", "Buy", "Weak Buy"}:
        reason_source = str(row.get("BuyReasonText", ""))
    elif state in {"Strong Sell", "Sell", "Weak Sell"}:
        reason_source = str(row.get("SellReasonText", ""))
    else:
        neutral_parts = [str(row.get("BuyReasonText", "")).split(" | ")[0], str(row.get("SellReasonText", "")).split(" | ")[0]]
        reason_source = " | ".join(part for part in neutral_parts if part)
    reason_parts = [part for part in reason_source.split(" | ") if part][:3]
    cycle_text = f"Cycle {_format_float(cycle_score, 1)}" if cycle_score is not None else "Cycle n/a"
    fear_text = f"Fear & Greed {_format_float(fear_greed, 0)}" if fear_greed is not None else "Fear & Greed n/a"
    reason_text = ", ".join(reason_parts) if reason_parts else "No dominant edge"
    return (
        f"{state} | {regime} | B{buy_score}/S{sell_score} | "
        f"Setup {int(_coerce_bool(row.get('BuySetupWindow', row.get('BuySetupActive', False))))}/"
        f"{int(_coerce_bool(row.get('SellSetupWindow', row.get('SellSetupActive', False))))} | "
        f"Trigger {int(_coerce_bool(row.get('BuyTrigger', False)))}/{int(_coerce_bool(row.get('SellTrigger', False)))} | "
        f"{cycle_text} | {td_label} | {fear_text} | {reason_text}"
    )


def build_state_frame(indicator_frame: pd.DataFrame, preset: SignalPreset) -> pd.DataFrame:
    frame = indicator_frame.dropna(subset=["Close"]).copy()
    if frame.empty:
        return frame

    min_hold_bars = preset_min_hold_bars(preset)
    reentry_lock_bars = preset_reentry_lock_bars(preset)
    state_rows = [build_signal_state(TupleRowAccessor(row), preset) for row in frame.itertuples(index=False, name="IndicatorRow")]
    state_frame = frame.join(pd.DataFrame(state_rows, index=frame.index), how="left")
    setup_window = max(3, min(6, preset.cooldown_bars + 1))
    state_frame["BuySetupWindow"] = state_frame["BuySetupActive"].rolling(setup_window, min_periods=1).max().fillna(0.0).astype(bool)
    state_frame["SellSetupWindow"] = state_frame["SellSetupActive"].rolling(setup_window, min_periods=1).max().fillna(0.0).astype(bool)
    state_frame["BuySetupPeakScore"] = state_frame["BuyScore"].where(state_frame["BuySetupActive"]).rolling(setup_window, min_periods=1).max()
    state_frame["SellSetupPeakScore"] = state_frame["SellScore"].where(state_frame["SellSetupActive"]).rolling(setup_window, min_periods=1).max()

    updated_buy_trigger: list[bool] = []
    updated_sell_trigger: list[bool] = []
    updated_state: list[str] = []
    hold_lock_flags: list[bool] = []
    reentry_lock_flags: list[bool] = []
    updated_stop_exit_confirmed: list[bool] = []
    updated_stop_emergency_exit: list[bool] = []
    updated_stop_gap_atr: list[float | None] = []
    in_virtual_position = False
    entry_index = -10_000
    last_exit_index = -10_000

    for index, row_tuple in enumerate(state_frame.itertuples(index=False, name="StateRow")):
        row = TupleRowAccessor(row_tuple)
        buy_score = int(row.get("BuyScore", 0) or 0)
        sell_score = int(row.get("SellScore", 0) or 0)
        regime = str(row.get("Regime", "Range / Transition"))
        bull_regime = _is_bull_regime(regime)
        bear_regime = _is_bear_regime(regime)
        reference_asset = _coerce_bool(row.get("ReferenceAsset"))
        buy_trigger = _coerce_bool(row.get("BuyTrigger", False))
        sell_trigger = _coerce_bool(row.get("SellTrigger", False))
        buy_window = _coerce_bool(row.get("BuySetupWindow", False))
        sell_window = _coerce_bool(row.get("SellSetupWindow", False))
        close_up_day = _coerce_bool(row.get("CloseUpDay", False))
        close_down_day = _coerce_bool(row.get("CloseDownDay", False))
        stop_touched = _coerce_bool(row.get("StopTouched", False))
        stop_close_breach = _coerce_bool(row.get("StopCloseBreach", False))
        close_value = _coerce_float(row.get("Close"))
        stop_price = _coerce_float(row.get("ChandelierStop"))
        atr_value = _coerce_float(row.get("ATR"))
        cycle_score = _coerce_float(row.get("CycleScore"))
        rsi = _coerce_float(row.get("RSI"))
        mfi = _coerce_float(row.get("MFI"))
        fear_greed = _coerce_float(row.get("FearGreed"))
        atr_stretch = _coerce_float(row.get("ATRStretch"))
        panic_low = bool(
            cycle_score is not None
            and rsi is not None
            and (
                (cycle_score <= 15 and rsi <= 40)
                or (cycle_score <= 25 and rsi <= 35)
            )
        )
        volume_ratio = _coerce_float(row.get("VolumeRatio20"))
        mfi_change = _coerce_float(row.get("MFIChange5"))
        rs_percentile = _coerce_float(row.get("RSPercentile"))
        cmf20 = _coerce_float(row.get("CMF20"))
        obv_slope10 = _coerce_float(row.get("OBVSlope10"))
        macd_bull_cross = _coerce_bool(row.get("MACDBullCross", False))
        macd_bear_cross = _coerce_bool(row.get("MACDBearCross", False))
        bb_pos = _coerce_float(row.get("BBPos20"))
        bb_width_pct = _coerce_float(row.get("BBWidthPct"))
        volume_climax_up = _coerce_bool(row.get("VolumeClimaxUp", False))
        volume_climax_down = _coerce_bool(row.get("VolumeClimaxDown", False))
        volume_dry_up = _coerce_bool(row.get("VolumeDryUp", False))
        tips_risk_off = _coerce_bool(row.get("TIPSRiskOff", False))
        buy_peak_score = _coerce_float(row.get("BuySetupPeakScore")) or 0.0
        sell_peak_score = _coerce_float(row.get("SellSetupPeakScore")) or 0.0
        buy_rsi_cutoff = bool(rsi is not None and rsi <= BUY_CUTOFFS["rsi_reset"])
        buy_rsi_extreme = bool(rsi is not None and rsi <= BUY_CUTOFFS["rsi_extreme"])
        buy_mfi_cutoff = bool(mfi is not None and mfi <= BUY_CUTOFFS["mfi_reset"])
        buy_mfi_extreme = bool(mfi is not None and mfi <= BUY_CUTOFFS["mfi_extreme"])
        buy_fear_cutoff = bool(fear_greed is not None and fear_greed <= BUY_CUTOFFS["fear_reset"])
        buy_fear_extreme = bool(fear_greed is not None and fear_greed <= BUY_CUTOFFS["fear_extreme"])
        buy_bb_cutoff = bool(bb_pos is not None and bb_pos <= BUY_CUTOFFS["bb_low"])
        buy_bb_extreme = bool(bb_pos is not None and bb_pos <= 0.10)
        buy_cutoff_hits = int(sum([buy_rsi_cutoff, buy_mfi_cutoff, buy_fear_cutoff, buy_bb_cutoff]))
        buy_extreme_hits = int(sum([buy_rsi_extreme, buy_mfi_extreme, buy_fear_extreme, buy_bb_extreme]))
        dryup_reversal = bool(
            volume_dry_up
            and cycle_score is not None
            and cycle_score <= 25
            and rsi is not None
            and rsi <= 44
        )
        accumulation_flow = bool(
            volume_climax_up
            or (
                cmf20 is not None
                and cmf20 >= 0.03
                and obv_slope10 is not None
                and obv_slope10 > 0
            )
        )
        distribution_flow = bool(
            volume_climax_down
            or (
                cmf20 is not None
                and cmf20 <= -0.03
                and obv_slope10 is not None
                and obv_slope10 < 0
            )
        )
        flow_conflict = distribution_flow and not accumulation_flow
        chop_zone = bool(bb_width_pct is not None and bb_width_pct <= 30)
        buy_flow_support = bool(
            (volume_ratio is not None and volume_ratio >= 1.02)
            or (mfi_change is not None and mfi_change >= 2.0)
            or (rs_percentile is not None and rs_percentile >= (42.0 if reference_asset else 45.0))
            or accumulation_flow
            or dryup_reversal
        )
        if flow_conflict and not panic_low and not macd_bull_cross:
            buy_flow_support = False
        sell_flow_support = bool(
            (volume_ratio is not None and volume_ratio >= 1.02)
            or (mfi_change is not None and mfi_change <= -2.0)
            or (rs_percentile is not None and rs_percentile <= 40.0)
            or distribution_flow
        )
        reversal_zone = bool(
            panic_low
            or (cycle_score is not None and cycle_score <= 35)
            or buy_score >= preset.watch_threshold
        )
        reversal_confirmation = bool(
            _coerce_bool(row.get("BuyTriggerPrice", False))
            or macd_bull_cross
            or (close_up_day and buy_flow_support and reversal_zone)
            or (reference_asset and close_up_day and reversal_zone and buy_peak_score >= preset.watch_threshold - 6)
        )
        breakdown_confirmation = bool(
            _coerce_bool(row.get("SellTriggerPrice", False))
            or macd_bear_cross
            or (close_down_day and sell_flow_support)
        )
        hold_lock_active = in_virtual_position and (index - entry_index) < min_hold_bars
        reentry_lock_active = (not in_virtual_position) and (index - last_exit_index) < reentry_lock_bars
        stop_exit_confirmed, stop_emergency_exit, stop_gap_atr = evaluate_stop_exit(
            close_value,
            stop_price,
            atr_value,
            sell_score=int(round(sell_peak_score or sell_score)),
            buy_score=buy_score,
            bearish_context=bear_regime,
            sell_context=bool(sell_window or _coerce_bool(row.get("SellSetupActive", False)) or _coerce_bool(row.get("SellTriggerPrice", False))),
            hold_lock_active=hold_lock_active,
            macro_risk_off=tips_risk_off,
        )
        sell_capitulation_block = bool(
            panic_low
            or buy_extreme_hits >= 2
            or (cycle_score is not None and cycle_score <= 30 and buy_cutoff_hits >= 2)
        )
        if sell_capitulation_block and not stop_emergency_exit:
            sell_window = False
            if not _coerce_bool(row.get("SellTriggerPrice", False)):
                sell_trigger = False
                breakdown_confirmation = False

        if not buy_trigger and buy_window and reversal_confirmation and reversal_zone:
            if (
                buy_peak_score >= preset.watch_threshold - (6 if reference_asset else 2)
                and sell_score <= max(buy_peak_score + 8, preset.strong_threshold + 2)
                and not (bear_regime and breakdown_confirmation and not panic_low)
                and not stop_exit_confirmed
                and (not chop_zone or macd_bull_cross or buy_flow_support or panic_low)
                and (not flow_conflict or panic_low or macd_bull_cross)
            ):
                buy_trigger = True
        if not sell_trigger and sell_window and breakdown_confirmation and not panic_low and not sell_capitulation_block:
            if sell_peak_score >= preset.watch_threshold + (6 if reference_asset else 4):
                if sell_flow_support and (bear_regime or stop_exit_confirmed or _coerce_bool(row.get("SellTriggerPrice", False)) or macd_bear_cross):
                    sell_trigger = True
        if reference_asset and panic_low and buy_window and reversal_confirmation and buy_peak_score >= preset.watch_threshold - 8:
            buy_trigger = True
            sell_trigger = False

        force_exit = bool(
            stop_emergency_exit
            or (
                sell_trigger
                and (
                    sell_peak_score >= max(preset.strong_threshold + 8, buy_score + 10)
                    or str(row.get("State", "")) == "Strong Sell"
                )
            )
        )
        if hold_lock_active and not force_exit:
            sell_trigger = False
        if reentry_lock_active and buy_trigger and buy_peak_score < max(preset.strong_threshold + 2, sell_score + 10):
            buy_trigger = False

        buy_strong_threshold = preset.strong_threshold - (8 if reference_asset else 5) if bull_regime else preset.strong_threshold - (4 if reference_asset else 0)
        sell_strong_threshold = preset.strong_threshold + 10 if bull_regime and not stop_exit_confirmed else preset.strong_threshold
        if reference_asset and panic_low and (buy_trigger or buy_window):
            if buy_peak_score >= preset.strong_threshold - 6 and reversal_confirmation:
                state = "Strong Buy"
            elif reversal_confirmation or buy_peak_score >= preset.watch_threshold + 2:
                state = "Buy"
            else:
                state = "Weak Buy"
        elif sell_trigger and (stop_emergency_exit or sell_peak_score >= max(sell_strong_threshold, buy_score + 10)):
            state = "Strong Sell"
        elif sell_trigger and sell_peak_score >= max(preset.watch_threshold + 8, buy_score + 6):
            state = "Sell"
        elif buy_trigger and buy_peak_score >= max(preset.watch_threshold + 10, buy_strong_threshold) and sell_score <= buy_peak_score + 6:
            state = "Strong Buy"
        elif buy_trigger and buy_peak_score >= max(preset.watch_threshold + 2, sell_score):
            state = "Buy"
        elif buy_window and buy_peak_score >= preset.watch_threshold - (8 if reference_asset else 6) and sell_score <= buy_peak_score + 12:
            state = "Weak Buy"
        elif sell_window and not panic_low and not sell_capitulation_block and sell_peak_score >= preset.watch_threshold + (8 if reference_asset else 2) and sell_score > buy_score + (6 if reference_asset else 4):
            state = "Sell"
        elif sell_window and not panic_low and not sell_capitulation_block and sell_peak_score >= preset.watch_threshold - (2 if reference_asset else 3) and sell_score > buy_score + (4 if reference_asset else 2):
            state = "Weak Sell"
        else:
            state = "Hold / Neutral"

        if hold_lock_active and not force_exit:
            if state in {"Strong Sell", "Sell"}:
                state = "Weak Buy" if buy_window else "Hold / Neutral"
            elif state == "Weak Sell":
                state = "Hold / Neutral"
        if reentry_lock_active and not buy_trigger:
            if state in {"Strong Buy", "Buy"}:
                state = "Weak Buy"
        if stop_emergency_exit:
            buy_trigger = False
            sell_trigger = True
            state = "Strong Sell"
        elif state in {"Strong Buy", "Buy", "Weak Buy"}:
            sell_trigger = False
        elif state in {"Strong Sell", "Sell", "Weak Sell"}:
            buy_trigger = False

        updated_buy_trigger.append(buy_trigger)
        updated_sell_trigger.append(sell_trigger)
        updated_state.append(state)
        hold_lock_flags.append(hold_lock_active)
        reentry_lock_flags.append(reentry_lock_active)
        updated_stop_exit_confirmed.append(stop_exit_confirmed)
        updated_stop_emergency_exit.append(stop_emergency_exit)
        updated_stop_gap_atr.append(stop_gap_atr)

        if not in_virtual_position and buy_trigger:
            in_virtual_position = True
            entry_index = index
        elif in_virtual_position and sell_trigger:
            in_virtual_position = False
            last_exit_index = index
            entry_index = -10_000

    state_frame["BuyTrigger"] = updated_buy_trigger
    state_frame["SellTrigger"] = updated_sell_trigger
    state_frame["State"] = updated_state
    state_frame["SignalNote"] = ""
    state_frame["HoldLockActive"] = hold_lock_flags
    state_frame["ReentryLockActive"] = reentry_lock_flags
    state_frame["StopExitConfirmed"] = updated_stop_exit_confirmed
    state_frame["StopEmergencyExit"] = updated_stop_emergency_exit
    state_frame["StopGapATR"] = updated_stop_gap_atr
    return state_frame


def build_snapshot(
    ticker: str,
    resolved_symbol: str,
    state_frame: pd.DataFrame,
    warning: str | None,
) -> TickerSnapshot:
    latest = state_frame.dropna(subset=["Close"]).iloc[-1]
    previous_close = state_frame["Close"].iloc[-2] if len(state_frame) > 1 else latest["Close"]
    daily_return = None if previous_close in {None, 0} else float(latest["Close"] / previous_close - 1.0)
    latest_note = _compose_signal_note(latest, str(latest.get("State", "Hold / Neutral")))

    return TickerSnapshot(
        ticker=ticker,
        resolved_symbol=resolved_symbol,
        as_of=pd.Timestamp(state_frame.index[-1]),
        last_close=float(latest["Close"]),
        daily_return=daily_return,
        state=str(latest["State"]),
        buy_score=int(latest["BuyScore"]),
        sell_score=int(latest["SellScore"]),
        cycle_score=_coerce_float(latest.get("CycleScore")),
        td_label=str(latest.get("TDLabel", "Unavailable")),
        atr_stretch=_coerce_float(latest.get("ATRStretch")),
        rsi=_coerce_float(latest.get("RSI")),
        fear_greed=_coerce_float(latest.get("FearGreed")),
        stop_distance=_coerce_float(latest.get("StopDistance")),
        note=latest_note,
        warning=warning,
    )


def build_watchlist_frame(snapshots: list[TickerSnapshot]) -> pd.DataFrame:
    if not snapshots:
        return pd.DataFrame()
    frame = pd.DataFrame([asdict(snapshot) for snapshot in snapshots])
    frame["state_priority"] = frame["state"].map(STATE_PRIORITY)
    frame = frame.sort_values(["state_priority", "sell_score", "buy_score", "ticker"], ascending=[True, False, False, True])
    return frame.drop(columns=["state_priority"])


def compute_strategy_total_return(trade_frame: pd.DataFrame) -> float | None:
    if trade_frame.empty:
        return None
    equity = 1.0
    closed = trade_frame.loc[trade_frame["Status"] == "Closed"].copy()
    for value in closed["Return"].tolist():
        equity *= 1.0 + float(value)
    open_trade = trade_frame.loc[trade_frame["Status"] == "Open"].copy()
    if not open_trade.empty:
        equity *= 1.0 + float(open_trade.iloc[-1]["Return"])
    return float(equity - 1.0)


def compute_buy_hold_return(state_frame: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    if state_frame.empty or "Close" not in state_frame.columns:
        return None, None, None
    close = pd.to_numeric(state_frame["Close"], errors="coerce").dropna()
    if close.empty:
        return None, None, None
    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])
    if start_price <= 0:
        return None, start_price, end_price
    return float(end_price / start_price - 1.0), start_price, end_price


def build_replay_summary(trade_frame: pd.DataFrame) -> dict[str, float | int | None]:
    closed = trade_frame.loc[trade_frame["Status"] == "Closed"].copy() if not trade_frame.empty else pd.DataFrame()
    open_trade = trade_frame.loc[trade_frame["Status"] == "Open"].copy() if not trade_frame.empty else pd.DataFrame()
    return {
        "closed_trades": int(len(closed)),
        "win_rate": float(closed["Return"].gt(0.0).mean()) if not closed.empty else None,
        "avg_closed_return": float(closed["Return"].mean()) if not closed.empty else None,
        "median_closed_return": float(closed["Return"].median()) if not closed.empty else None,
        "avg_hold_days": float(closed["HoldDays"].mean()) if not closed.empty else None,
        "open_trade_return": float(open_trade["Return"].iloc[-1]) if not open_trade.empty else None,
        "strategy_total_return": compute_strategy_total_return(trade_frame),
    }


def build_strategy_vs_benchmark_summary(
    state_frame: pd.DataFrame,
    trade_frame: pd.DataFrame,
) -> dict[str, float | int | None]:
    summary = build_replay_summary(trade_frame)
    buy_hold_return, buy_hold_start_price, buy_hold_end_price = compute_buy_hold_return(state_frame)
    strategy_total_return = summary.get("strategy_total_return")
    summary["buy_hold_return"] = buy_hold_return
    summary["buy_hold_start_price"] = buy_hold_start_price
    summary["buy_hold_end_price"] = buy_hold_end_price
    summary["excess_return"] = (
        None
        if strategy_total_return is None or buy_hold_return is None
        else float(strategy_total_return - buy_hold_return)
    )
    summary["relative_alpha"] = (
        None
        if strategy_total_return is None or buy_hold_return is None
        else float((1.0 + strategy_total_return) / (1.0 + buy_hold_return) - 1.0)
    )
    return summary


def build_buy_hold_trade_frame(ticker: str, state_frame: pd.DataFrame) -> pd.DataFrame:
    if state_frame.empty or "Close" not in state_frame.columns:
        return pd.DataFrame()
    close = pd.to_numeric(state_frame["Close"], errors="coerce").dropna()
    if close.empty:
        return pd.DataFrame()
    entry_date = pd.Timestamp(close.index[0])
    exit_date = pd.Timestamp(close.index[-1])
    entry_price = float(close.iloc[0])
    exit_price = float(close.iloc[-1])
    if entry_price <= 0:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "Ticker": ticker,
                "Trade": 1,
                "Status": "Closed",
                "EntryDate": entry_date,
                "ExitDate": exit_date,
                "EntryPrice": entry_price,
                "ExitPrice": exit_price,
                "Return": float(exit_price / entry_price - 1.0),
                "HoldDays": int((exit_date - entry_date).days),
                "PartialScaledOut": "No",
            }
        ]
    )


def build_trade_replay_from_state_frame(
    ticker: str,
    state_frame: pd.DataFrame,
    preset: SignalPreset,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | None]]:
    if state_frame.empty:
        empty_summary: dict[str, float | int | None] = {
            "closed_trades": 0,
            "win_rate": None,
            "avg_closed_return": None,
            "median_closed_return": None,
            "avg_hold_days": None,
            "open_trade_return": None,
            "strategy_total_return": None,
            "buy_hold_return": None,
            "buy_hold_start_price": None,
            "buy_hold_end_price": None,
            "excess_return": None,
            "relative_alpha": None,
        }
        return pd.DataFrame(), pd.DataFrame(), empty_summary

    events: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    in_position = False
    partial_taken = False
    trade_id = 0
    entry_date: pd.Timestamp | None = None
    entry_price: float | None = None
    position_remaining = 0.0
    realized_return = 0.0
    last_buy_index = -10_000
    last_sell_index = -10_000
    last_partial_index = -10_000
    last_scale_index = -10_000
    entry_index = -10_000
    min_hold_bars = preset_min_hold_bars(preset)
    partial_hold_bars = max(1, min_hold_bars - 1)
    iter_frame = state_frame.reset_index().rename(columns={"index": "Date"})

    for position, row in enumerate(iter_frame.itertuples(index=False)):
        timestamp = pd.Timestamp(row.Date)
        state = str(row.State)
        state_sell_side = state in {"Weak Sell", "Sell", "Strong Sell"}
        price = float(row.Close)
        low_price = float(getattr(row, "Low", price))
        stop_price = float(row.ChandelierStop) if pd.notna(getattr(row, "ChandelierStop", np.nan)) else np.nan
        stop_touched = bool(
            _coerce_bool(getattr(row, "StopTouched", False))
            or (pd.notna(stop_price) and low_price <= stop_price)
        )
        note = str(getattr(row, "SignalNote", "") or "")
        if not note:
            note = _compose_signal_note(TupleRowAccessor(row), state)
        buy_score = int(row.BuyScore)
        sell_score = int(row.SellScore)
        atr_value = _coerce_float(getattr(row, "ATR", np.nan))
        buy_trigger = _coerce_bool(getattr(row, "BuyTrigger", state in {"Buy", "Strong Buy"}))
        sell_setup_active = _coerce_bool(getattr(row, "SellSetupActive", state_sell_side))
        regime = str(getattr(row, "Regime", "") or "")
        regime_known = bool(regime)
        bull_regime = regime in ML_BULL_REGIMES
        close_up_day = _coerce_bool(getattr(row, "CloseUpDay", False))
        deep_value_confluence = _coerce_bool(getattr(row, "DeepValueConfluence", False))
        heat_trim_confluence = _coerce_bool(getattr(row, "HeatTrimConfluence", False))
        macro_trend_exit = _coerce_bool(getattr(row, "MacroTrendExit", False))
        if regime_known and not deep_value_confluence:
            deep_value_confluence, _ = detect_deep_value_confluence(TupleRowAccessor(row))
        if regime_known and not heat_trim_confluence:
            heat_trim_confluence, _ = detect_heat_trim_confluence(TupleRowAccessor(row))
        if regime_known and not macro_trend_exit:
            macro_trend_exit = detect_macro_trend_exit(TupleRowAccessor(row))
        hold_bars_elapsed = position - entry_index if in_position else 10_000
        hold_lock_active = in_position and hold_bars_elapsed < min_hold_bars
        stop_exit_confirmed = _coerce_bool(getattr(row, "StopExitConfirmed", False))
        stop_emergency_exit = _coerce_bool(getattr(row, "StopEmergencyExit", False))
        if not stop_exit_confirmed and not stop_emergency_exit:
            stop_exit_confirmed, stop_emergency_exit, _ = evaluate_stop_exit(
                price,
                None if pd.isna(stop_price) else stop_price,
                atr_value,
                sell_score=sell_score,
                buy_score=buy_score,
                bearish_context=state in {"Sell", "Strong Sell"},
                sell_context=bool(sell_setup_active or state_sell_side),
                hold_lock_active=hold_lock_active,
            )
        sell_trigger = (
            _coerce_bool(getattr(row, "SellTrigger", state in {"Sell", "Strong Sell"}))
            or stop_emergency_exit
            or (stop_exit_confirmed and state_sell_side)
        )
        force_exit = bool(
            (not regime_known and stop_emergency_exit)
            or (
                not regime_known
                and state == "Strong Sell"
                and sell_score >= max(preset.strong_threshold + 8, buy_score + 10)
            )
        )

        buy_ready = (not in_position) and buy_trigger and (position - last_buy_index) >= preset.cooldown_bars
        scale_in_ready = (
            in_position
            and position_remaining < 1.0
            and deep_value_confluence
            and (buy_trigger or close_up_day)
            and (position - last_scale_index) >= LOW_CONFLUENCE_SCALE_IN_COOLDOWN_BARS
        )
        partial_ready = (
            in_position
            and position_remaining > (1.0 - HEAT_TRIM_REDUCE_WEIGHT)
            and (
                heat_trim_confluence
                or (
                    not regime_known
                    and (not partial_taken)
                    and state == "Weak Sell"
                    and sell_setup_active
                    and not stop_exit_confirmed
                )
            )
            and (hold_bars_elapsed >= partial_hold_bars)
            and (position - last_partial_index) >= HEAT_TRIM_COOLDOWN_BARS
        )
        sell_ready = (
            in_position
            and (macro_trend_exit or ((not regime_known) and sell_trigger))
            and ((hold_bars_elapsed >= min_hold_bars) or force_exit or macro_trend_exit)
            and (position - last_sell_index) >= preset.cooldown_bars
        )

        if buy_ready:
            trade_id += 1
            in_position = True
            partial_taken = False
            entry_date = pd.Timestamp(timestamp)
            entry_price = price
            position_remaining = 1.0
            realized_return = 0.0
            last_buy_index = position
            entry_index = position
            events.append(
                {
                    "Ticker": ticker,
                    "Trade": trade_id,
                    "Date": timestamp,
                    "Signal": "BUY",
                    "Price": price,
                    "State": state,
                    "BuyScore": buy_score,
                    "SellScore": sell_score,
                    "PositionAfter": position_remaining,
                    "Reason": note,
                }
            )
            continue

        if scale_in_ready and entry_price is not None and position_remaining > 0:
            buy_weight = min(LOW_CONFLUENCE_SCALE_IN_WEIGHT, 1.0 - position_remaining)
            if buy_weight > 0:
                entry_price = ((entry_price * position_remaining) + (price * buy_weight)) / (position_remaining + buy_weight)
                position_remaining += buy_weight
                last_scale_index = position
                events.append(
                    {
                        "Ticker": ticker,
                        "Trade": trade_id,
                        "Date": timestamp,
                        "Signal": "SCALE IN BUY",
                        "Price": price,
                        "State": state,
                        "BuyScore": buy_score,
                        "SellScore": sell_score,
                        "PositionAfter": position_remaining,
                        "Reason": "Deep value confluence scale-in",
                    }
                )
                continue

        if partial_ready and entry_price is not None and position_remaining > 0:
            sell_weight = max(0.0, position_remaining - (1.0 - HEAT_TRIM_REDUCE_WEIGHT))
            if sell_weight <= 0:
                sell_weight = min(HEAT_TRIM_REDUCE_WEIGHT, position_remaining)
            realized_return += sell_weight * (price / entry_price - 1.0)
            position_remaining -= sell_weight
            partial_taken = True
            last_partial_index = position
            events.append(
                {
                    "Ticker": ticker,
                    "Trade": trade_id,
                    "Date": timestamp,
                    "Signal": "PARTIAL SELL",
                    "Price": price,
                    "State": state,
                    "BuyScore": buy_score,
                    "SellScore": sell_score,
                    "PositionAfter": position_remaining,
                    "Reason": "Fear/RSI/MA trim confluence" if heat_trim_confluence else note,
                }
            )
            continue

        if sell_ready and entry_price is not None and entry_date is not None and position_remaining > 0:
            exit_price = price
            realized_return += position_remaining * (exit_price / entry_price - 1.0)
            last_sell_index = position
            events.append(
                {
                    "Ticker": ticker,
                    "Trade": trade_id,
                    "Date": timestamp,
                    "Signal": "SELL",
                    "Price": exit_price,
                    "State": state,
                    "BuyScore": buy_score,
                    "SellScore": sell_score,
                    "PositionAfter": 0.0,
                    "Reason": "MACD bear + fear regime exit" if macro_trend_exit else note,
                }
            )
            trades.append(
                {
                    "Ticker": ticker,
                    "Trade": trade_id,
                    "Status": "Closed",
                    "EntryDate": entry_date,
                    "ExitDate": timestamp,
                    "EntryPrice": entry_price,
                    "ExitPrice": exit_price,
                    "Return": realized_return,
                    "HoldDays": int((timestamp - entry_date).days),
                    "PartialScaledOut": "Yes" if partial_taken else "No",
                }
            )
            in_position = False
            partial_taken = False
            entry_date = None
            entry_price = None
            position_remaining = 0.0
            realized_return = 0.0
            entry_index = -10_000
            last_scale_index = -10_000

    if in_position and entry_price is not None and entry_date is not None:
        last_timestamp = pd.Timestamp(state_frame.index[-1])
        last_price = float(state_frame["Close"].iloc[-1])
        mark_to_market = realized_return + (position_remaining * (last_price / entry_price - 1.0))
        trades.append(
            {
                "Ticker": ticker,
                "Trade": trade_id,
                "Status": "Open",
                "EntryDate": entry_date,
                "ExitDate": last_timestamp,
                "EntryPrice": entry_price,
                "ExitPrice": last_price,
                "Return": mark_to_market,
                "HoldDays": int((last_timestamp - entry_date).days),
                "PartialScaledOut": "Yes" if partial_taken else "No",
            }
        )

    event_frame = pd.DataFrame(events)
    trade_frame = pd.DataFrame(trades)
    return event_frame, trade_frame, build_strategy_vs_benchmark_summary(state_frame, trade_frame)


def build_chart_signal_frame(state_frame: pd.DataFrame, preset: SignalPreset) -> pd.DataFrame:
    if state_frame.empty:
        return pd.DataFrame(columns=["Date", "Signal", "Price", "State", "BuyScore", "SellScore"])

    markers: list[dict[str, Any]] = []
    last_buy_index = -10_000
    last_sell_index = -10_000
    iter_frame = state_frame.reset_index().rename(columns={"index": "Date"})

    for position, row in enumerate(iter_frame.itertuples(index=False)):
        price = float(row.Close)
        stop_price = float(row.ChandelierStop) if pd.notna(getattr(row, "ChandelierStop", np.nan)) else np.nan
        atr_value = _coerce_float(getattr(row, "ATR", np.nan))
        buy_trigger = _coerce_bool(getattr(row, "BuyTrigger", str(row.State) in {"Buy", "Strong Buy"}))
        stop_exit_confirmed = _coerce_bool(getattr(row, "StopExitConfirmed", False))
        stop_emergency_exit = _coerce_bool(getattr(row, "StopEmergencyExit", False))
        if not stop_exit_confirmed and not stop_emergency_exit:
            stop_exit_confirmed, stop_emergency_exit, _ = evaluate_stop_exit(
                price,
                None if pd.isna(stop_price) else stop_price,
                atr_value,
                sell_score=int(row.SellScore),
                buy_score=int(row.BuyScore),
                bearish_context=str(row.State) in {"Sell", "Strong Sell"},
                sell_context=bool(_coerce_bool(getattr(row, "SellSetupActive", False)) or str(row.State) in {"Weak Sell", "Sell", "Strong Sell"}),
                hold_lock_active=_coerce_bool(getattr(row, "HoldLockActive", False)),
            )
        sell_trigger = (
            _coerce_bool(getattr(row, "SellTrigger", str(row.State) in {"Sell", "Strong Sell"}))
            or stop_emergency_exit
            or (stop_exit_confirmed and str(row.State) in {"Weak Sell", "Sell", "Strong Sell"})
        )

        if buy_trigger and sell_trigger:
            if int(row.SellScore) >= int(row.BuyScore):
                buy_trigger = False
            else:
                sell_trigger = False

        if buy_trigger:
            if (position - last_buy_index) >= preset.cooldown_bars:
                last_buy_index = position
                markers.append(
                    {
                        "Date": pd.Timestamp(row.Date),
                        "Signal": "BUY",
                        "Price": price,
                        "State": str(row.State),
                        "BuyScore": int(row.BuyScore),
                        "SellScore": int(row.SellScore),
                    }
                )
        if sell_trigger:
            if (position - last_sell_index) >= preset.cooldown_bars:
                last_sell_index = position
                markers.append(
                    {
                        "Date": pd.Timestamp(row.Date),
                        "Signal": "SELL",
                        "Price": price,
                        "State": str(row.State),
                        "BuyScore": int(row.BuyScore),
                        "SellScore": int(row.SellScore),
                    }
                )

    return pd.DataFrame(markers)


def build_price_context_figure(
    ticker: str,
    state_frame: pd.DataFrame,
    event_frame: pd.DataFrame,
    preset: SignalPreset,
) -> go.Figure:
    del preset
    plot_frame = state_frame.loc[:, ["Close", "BuyScore", "SellScore", "State"]].copy().reset_index().rename(columns={"index": "Date"})
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_frame["Date"],
            y=plot_frame["Close"],
            mode="lines",
            name="Close",
            line={"color": "#0f172a", "width": 2.2},
            hovertemplate="Date %{x|%Y-%m-%d}<br>Close %{y:.2f}<br>Buy %{customdata[0]} / Sell %{customdata[1]}<extra></extra>",
            customdata=plot_frame[["BuyScore", "SellScore"]],
        )
    )

    if not event_frame.empty:
        marker_styles = {
            "BUY": {"color": "#16a34a", "symbol": "triangle-up", "textposition": "bottom center"},
            "SELL": {"color": "#dc2626", "symbol": "triangle-down", "textposition": "top center"},
        }
        for signal, style in marker_styles.items():
            subset = event_frame.loc[event_frame["Signal"] == signal]
            if subset.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=subset["Date"],
                    y=subset["Price"],
                    mode="markers+text",
                    text=subset["Signal"],
                    textposition=style["textposition"],
                    name=signal,
                    marker={
                        "size": 14,
                        "color": style["color"],
                        "symbol": style["symbol"],
                        "line": {"width": 1, "color": "#ffffff"},
                    },
                    customdata=subset[["State", "BuyScore", "SellScore"]],
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>%{customdata[0]}<br>Price %{y:.2f}"
                        "<br>Buy %{customdata[1]} / Sell %{customdata[2]}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=f"{ticker} Buy / Sell Signal Overlay",
        height=520,
        margin={"l": 12, "r": 12, "t": 52, "b": 10},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.04},
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price")
    return fig


def build_macro_figure(macro_data: dict[str, Any], history_years: int) -> go.Figure:
    plot_df = macro_data["plot_df"].tail(max(history_years * 252, 120)).copy()
    latest_factors = macro_data["latest_factors"]
    score = float(macro_data["latest_score"])
    normalized_spy = (plot_df["SPY"] / plot_df["SPY"].iloc[0] - 1.0) * 100.0 if "SPY" in plot_df.columns and not plot_df["SPY"].dropna().empty else pd.Series(dtype=float)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "indicator"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Fear & Greed Gauge", "Score vs SPY", "Latest Factor Rank", "Internal Health"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#102a43"},
                "steps": [
                    {"range": [0, 20], "color": "#dbeafe"},
                    {"range": [20, 40], "color": "#dcfce7"},
                    {"range": [40, 60], "color": "#f8fafc"},
                    {"range": [60, 80], "color": "#fef3c7"},
                    {"range": [80, 100], "color": "#fee2e2"},
                ],
            },
            title={"text": macro_data["latest_label"]},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Score"], name="Fear & Greed", line={"color": "#0f766e", "width": 2.2}), row=1, col=2)
    if not normalized_spy.empty:
        fig.add_trace(go.Scatter(x=plot_df.index, y=normalized_spy, name="SPY return %", line={"color": "#dd6b20", "width": 1.6, "dash": "dash"}), row=1, col=2)
    fig.add_trace(go.Bar(x=latest_factors.index, y=latest_factors.values, marker_color="#2563eb", name="Factor score"), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Breadth"], name="Breadth", line={"color": "#dd6b20", "width": 1.8}), row=2, col=2)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Sector"], name="Sector", line={"color": "#2563eb", "width": 1.8}), row=2, col=2)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Credit"], name="Credit", line={"color": "#0f766e", "width": 1.8}), row=2, col=2)
    fig.add_trace(go.Scatter(x=plot_df.index, y=[80] * len(plot_df), mode="lines", line={"color": "#dc2626", "width": 1, "dash": "dot"}, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=plot_df.index, y=[20] * len(plot_df), mode="lines", line={"color": "#2563eb", "width": 1, "dash": "dot"}, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=plot_df.index, y=[50] * len(plot_df), mode="lines", line={"color": "#64748b", "width": 1, "dash": "dot"}, showlegend=False), row=2, col=2)
    fig.update_layout(
        title="Macro Fear & Greed Overview",
        height=700,
        margin={"l": 12, "r": 12, "t": 52, "b": 10},
        legend={"orientation": "h", "y": 1.08},
    )
    fig.update_yaxes(title_text="Score", row=1, col=2)
    fig.update_yaxes(title_text="Percentile", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=2)
    return fig


def render_header(macro_data: dict[str, Any], snapshots: list[TickerSnapshot]) -> None:
    latest_as_of = max(snapshot.as_of for snapshot in snapshots).date().isoformat() if snapshots else "n/a"
    warning_text = ""
    if macro_data.get("warning"):
        warning_text = " Macro model is using a neutral fallback for unavailable fields."
    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_TITLE}</h1>
            <p>Latest completed daily bar: {latest_as_of} | Macro Fear & Greed: {macro_data['latest_label']} ({macro_data['latest_score']:.0f}) | yfinance-only deployment path.{warning_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_metrics(watchlist_frame: pd.DataFrame, macro_data: dict[str, Any], preset: SignalPreset) -> None:
    strong_buy = int((watchlist_frame["state"] == "Strong Buy").sum()) if not watchlist_frame.empty else 0
    buy_count = int((watchlist_frame["state"] == "Buy").sum()) if not watchlist_frame.empty else 0
    strong_sell = int((watchlist_frame["state"] == "Strong Sell").sum()) if not watchlist_frame.empty else 0
    sell_count = int((watchlist_frame["state"] == "Sell").sum()) if not watchlist_frame.empty else 0
    avg_buy = float(watchlist_frame["buy_score"].mean()) if not watchlist_frame.empty else 0.0
    cards = st.columns(4)
    with cards[0]:
        render_metric_card("Macro Fear & Greed", f"{macro_data['latest_score']:.0f}", macro_data["latest_label"])
    with cards[1]:
        render_metric_card("Buy Bias", str(strong_buy + buy_count), f"Strong {strong_buy} / Buy {buy_count}")
    with cards[2]:
        render_metric_card("Sell Bias", str(strong_sell + sell_count), f"Strong {strong_sell} / Sell {sell_count}")
    with cards[3]:
        render_metric_card("Average BuyScore", f"{avg_buy:.0f}", f"Watch threshold {preset.watch_threshold}")


def render_watchlist_summary(watchlist_frame: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">Watchlist Summary</div>', unsafe_allow_html=True)
    if watchlist_frame.empty:
        st.warning("No usable watchlist data was returned.")
        return

    display = watchlist_frame.loc[
        :,
        [
            "ticker",
            "resolved_symbol",
            "as_of",
            "last_close",
            "daily_return",
            "state",
            "buy_score",
            "sell_score",
            "cycle_score",
            "td_label",
            "atr_stretch",
            "rsi",
            "fear_greed",
            "stop_distance",
            "warning",
        ],
    ].copy()
    display.columns = [
        "Ticker",
        "Resolved",
        "As Of",
        "Close",
        "1D Return",
        "State",
        "BuyScore",
        "SellScore",
        "CycleScore",
        "TDLabel",
        "ATRStretch",
        "RSI",
        "FearGreed",
        "StopDistance",
        "Warning",
    ]
    display["Close"] = display["Close"].map(_format_float)
    display["1D Return"] = display["1D Return"].map(_format_pct)
    display["CycleScore"] = display["CycleScore"].map(lambda value: _format_float(value, 1))
    display["ATRStretch"] = display["ATRStretch"].map(lambda value: _format_float(value, 2))
    display["RSI"] = display["RSI"].map(lambda value: _format_float(value, 1))
    display["FearGreed"] = display["FearGreed"].map(lambda value: _format_float(value, 0))
    display["StopDistance"] = display["StopDistance"].map(_format_pct)
    st.dataframe(display, width="stretch", hide_index=True)


def render_warning_panel(warnings: list[str]) -> None:
    active_warnings = [item for item in warnings if item]
    if not active_warnings:
        return
    with st.expander("Warnings", expanded=False):
        for message in active_warnings:
            st.warning(message)


def render_signal_review(event_frame: pd.DataFrame, trade_frame: pd.DataFrame, summary: dict[str, float | int | None]) -> None:
    st.markdown('<div class="section-label">Historical Signal Replay</div>', unsafe_allow_html=True)
    st.caption(
        "Signals below replay the same rule set through history. This is a rule-based timing review, not a broker-grade fill simulator."
    )

    avg_hold_days = "n/a" if summary["avg_hold_days"] is None else f"{summary['avg_hold_days']:.0f}d"
    cards = st.columns(6)
    with cards[0]:
        render_metric_card("Closed Trades", str(summary["closed_trades"]), "Completed round trips")
    with cards[1]:
        render_metric_card("Win Rate", _format_pct(summary["win_rate"]), "Closed trades only")
    with cards[2]:
        render_metric_card("Avg Closed Return", _format_pct(summary["avg_closed_return"]), "Closed trades only")
    with cards[3]:
        render_metric_card("Median Return", _format_pct(summary["median_closed_return"]), "Closed trades only")
    with cards[4]:
        render_metric_card("Avg Hold", avg_hold_days, "Closed trades only")
    with cards[5]:
        open_caption = "No open simulated position" if summary["open_trade_return"] is None else "Current mark-to-market"
        render_metric_card("Open Trade", _format_pct(summary["open_trade_return"]), open_caption)

    benchmark_cards = st.columns(3)
    with benchmark_cards[0]:
        render_metric_card("Strategy Total", _format_pct(summary.get("strategy_total_return")), "Compounded closed + open trade")
    with benchmark_cards[1]:
        render_metric_card("Buy & Hold", _format_pct(summary.get("buy_hold_return")), "First close to latest close")
    with benchmark_cards[2]:
        render_metric_card("Alpha vs Hold", _format_pct(summary.get("excess_return")), "Strategy total minus buy & hold")

    left, right = st.columns([1.25, 1.0])
    with left:
        if event_frame.empty:
            st.info("No historical signal events were generated for the selected ticker.")
        else:
            display = event_frame.sort_values("Date", ascending=False).copy()
            display["Date"] = display["Date"].dt.date
            display["Price"] = display["Price"].map(_format_float)
            st.dataframe(
                display.loc[:, ["Date", "Signal", "Price", "State", "BuyScore", "SellScore"]],
                width="stretch",
                hide_index=True,
            )
    with right:
        if trade_frame.empty:
            st.info("No simulated trades to summarize yet.")
        else:
            display = trade_frame.sort_values(["EntryDate", "Trade"], ascending=[False, False]).copy()
            display["EntryDate"] = display["EntryDate"].dt.date
            display["ExitDate"] = display["ExitDate"].dt.date
            display["EntryPrice"] = display["EntryPrice"].map(_format_float)
            display["ExitPrice"] = display["ExitPrice"].map(_format_float)
            display["Return"] = display["Return"].map(_format_pct)
            display["HoldDays"] = display["HoldDays"].map(lambda value: f"{int(value)}d")
            st.dataframe(
                display.loc[:, ["Trade", "Status", "EntryDate", "ExitDate", "EntryPrice", "ExitPrice", "Return", "HoldDays", "PartialScaledOut"]],
                width="stretch",
                hide_index=True,
            )


def render_strategy_comparison(comparison: dict[str, Any], selection_frame: pd.DataFrame) -> None:
    if not comparison:
        return
    rule_summary = comparison.get("rule_summary", {})
    ml_summary = comparison.get("ml_summary", {})
    if not rule_summary and not ml_summary:
        return

    st.markdown('<div class="section-label">Rule vs ML Filter</div>', unsafe_allow_html=True)
    comparison_frame = pd.DataFrame(
        [
            {
                "Metric": "Closed Trades",
                "Rule Only": rule_summary.get("closed_trades"),
                "ML Filter": ml_summary.get("closed_trades"),
            },
            {
                "Metric": "Strategy Total",
                "Rule Only": _format_pct(rule_summary.get("strategy_total_return")),
                "ML Filter": _format_pct(ml_summary.get("strategy_total_return")),
            },
            {
                "Metric": "Buy & Hold",
                "Rule Only": _format_pct(rule_summary.get("buy_hold_return")),
                "ML Filter": _format_pct(ml_summary.get("buy_hold_return")),
            },
            {
                "Metric": "Alpha vs Hold",
                "Rule Only": _format_pct(rule_summary.get("excess_return")),
                "ML Filter": _format_pct(ml_summary.get("excess_return")),
            },
            {
                "Metric": "Win Rate",
                "Rule Only": _format_pct(rule_summary.get("win_rate")),
                "ML Filter": _format_pct(ml_summary.get("win_rate")),
            },
            {
                "Metric": "Avg Closed Return",
                "Rule Only": _format_pct(rule_summary.get("avg_closed_return")),
                "ML Filter": _format_pct(ml_summary.get("avg_closed_return")),
            },
            {
                "Metric": "Median Return",
                "Rule Only": _format_pct(rule_summary.get("median_closed_return")),
                "ML Filter": _format_pct(ml_summary.get("median_closed_return")),
            },
            {
                "Metric": "Open Trade",
                "Rule Only": _format_pct(rule_summary.get("open_trade_return")),
                "ML Filter": _format_pct(ml_summary.get("open_trade_return")),
            },
        ]
    )
    st.dataframe(comparison_frame, width="stretch", hide_index=True)
    st.caption(
        "OOS threshold selection: "
        f"buy >= {comparison.get('buy_threshold', 0.0):.2f}, "
        f"sell >= {comparison.get('sell_threshold', 0.0):.2f}, "
        f"sell veto = {comparison.get('sell_veto_profile', ML_DEFAULT_SELL_VETO_PROFILE)}"
    )
    if not selection_frame.empty:
        with st.expander("Walk-forward fold selections", expanded=False):
            display = selection_frame.copy()
            display["WinRate"] = display["WinRate"].map(_format_pct)
            display["StrategyTotalReturn"] = display["StrategyTotalReturn"].map(_format_pct)
            display["BuyHoldReturn"] = display["BuyHoldReturn"].map(_format_pct)
            display["ExcessReturn"] = display["ExcessReturn"].map(_format_pct)
            display["AvgClosedReturn"] = display["AvgClosedReturn"].map(_format_pct)
            st.dataframe(display, width="stretch", hide_index=True)


def render_ticker_panel(
    selected_ticker: str,
    snapshot: TickerSnapshot,
    state_frame: pd.DataFrame,
    chart_signal_frame: pd.DataFrame,
    event_frame: pd.DataFrame,
    trade_frame: pd.DataFrame,
    replay_summary: dict[str, float | int | None],
    preset: SignalPreset,
    comparison: dict[str, Any] | None = None,
    selection_frame: pd.DataFrame | None = None,
) -> None:
    st.markdown('<div class="section-label">Ticker Drill-down</div>', unsafe_allow_html=True)
    latest = state_frame.dropna(subset=["Close"]).iloc[-1]
    display_label = format_snapshot_label(snapshot)
    left, right = st.columns([1.65, 1.0])
    with left:
        st.plotly_chart(build_price_context_figure(display_label, state_frame, chart_signal_frame, preset), width="stretch", config={"displaylogo": False})
    with right:
        score_cards_top = st.columns(2)
        with score_cards_top[0]:
            render_metric_card("Current State", snapshot.state, f"Resolved {snapshot.resolved_symbol}")
        with score_cards_top[1]:
            render_metric_card("Score Spread", str(snapshot.buy_score - snapshot.sell_score), "BuyScore minus SellScore")

        score_cards_bottom = st.columns(2)
        with score_cards_bottom[0]:
            render_metric_card("BuyScore", str(snapshot.buy_score), "Composite buy score")
        with score_cards_bottom[1]:
            render_metric_card("SellScore", str(snapshot.sell_score), "Composite sell score")

        detail = pd.DataFrame(
            [
                {"Field": "Close", "Value": _format_float(snapshot.last_close)},
                {"Field": "1D Return", "Value": _format_pct(snapshot.daily_return)},
                {"Field": "CycleScore", "Value": _format_float(snapshot.cycle_score, 1)},
                {"Field": "TDLabel", "Value": snapshot.td_label},
                {"Field": "ATRStretch", "Value": _format_float(snapshot.atr_stretch, 2)},
                {"Field": "RSI", "Value": _format_float(snapshot.rsi, 1)},
                {"Field": "MFI", "Value": _format_float(_coerce_float(latest.get("MFI")), 1)},
                {"Field": "ADX", "Value": _format_float(_coerce_float(latest.get("ADX")), 1)},
                {"Field": "MACDHist", "Value": _format_float(_coerce_float(latest.get("MACDHist")), 3)},
                {"Field": "BBPos20", "Value": _format_float(_coerce_float(latest.get("BBPos20")), 2)},
                {"Field": "RSPercentile", "Value": _format_float(_coerce_float(latest.get("RSPercentile")), 1)},
                {"Field": "FearGreed", "Value": _format_float(snapshot.fear_greed, 0)},
                {"Field": "StopDistance", "Value": _format_pct(snapshot.stop_distance)},
                {"Field": "Regime", "Value": str(latest.get("Regime", "n/a"))},
                {"Field": "MLBuyProb", "Value": _format_float(_coerce_float(latest.get("MLBuyProb")), 3)},
                {"Field": "MLSellProb", "Value": _format_float(_coerce_float(latest.get("MLSellProb")), 3)},
                {"Field": "MLBuyApproved", "Value": str(bool(_coerce_bool(latest.get("MLBuyApproved"))))},
                {"Field": "MLSellApproved", "Value": str(bool(_coerce_bool(latest.get("MLSellApproved"))))},
                {"Field": "ModelMode", "Value": str(latest.get("ModelMode", "rules_only"))},
                {"Field": "SellVetoProfile", "Value": str(latest.get("SellVetoProfile", ML_DEFAULT_SELL_VETO_PROFILE))},
                {"Field": "StrategySource", "Value": str(latest.get("StrategySource", "RuleOnly"))},
                {"Field": "As Of", "Value": snapshot.as_of.date().isoformat()},
            ]
        )
        st.dataframe(detail, width="stretch", hide_index=True)

    render_strategy_comparison(comparison or {}, selection_frame if selection_frame is not None else pd.DataFrame())
    render_signal_review(event_frame, trade_frame, replay_summary)


def render_diagnostics(diagnostics: pd.DataFrame, performance_frame: pd.DataFrame) -> None:
    with st.expander("Data diagnostics", expanded=False):
        st.dataframe(diagnostics, width="stretch", hide_index=True)
        if not performance_frame.empty:
            st.markdown("**Performance timings (seconds)**")
            display = performance_frame.copy()
            display["Seconds"] = display["Seconds"].map(lambda value: f"{value:.4f}")
            st.dataframe(display, width="stretch", hide_index=True)


def build_performance_frame(timing_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not timing_rows:
        return pd.DataFrame(columns=["Stage", "Target", "Seconds", "Rows"])
    frame = pd.DataFrame(timing_rows)
    frame["Rows"] = pd.to_numeric(frame["Rows"], errors="coerce").astype("Int64")
    return frame.sort_values(["Seconds", "Stage", "Target"], ascending=[False, True, True]).reset_index(drop=True)


def build_diagnostics_frame(
    watchlist: list[str],
    price_payloads: dict[str, dict[str, Any]],
    macro_data: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in watchlist:
        payload = price_payloads.get(ticker, {})
        frame = payload.get("frame", pd.DataFrame())
        rows.append(
            {
                "Ticker": ticker,
                "Resolved": payload.get("resolved_symbol", ticker),
                "Rows": int(len(frame)),
                "FirstDate": frame.index.min().date().isoformat() if isinstance(frame, pd.DataFrame) and not frame.empty else "n/a",
                "LastDate": frame.index.max().date().isoformat() if isinstance(frame, pd.DataFrame) and not frame.empty else "n/a",
                "Warning": payload.get("warning") or "",
            }
        )
    rows.append(
        {
            "Ticker": "Macro",
            "Resolved": "FearGreed",
            "Rows": int(len(macro_data.get("score_series", pd.Series(dtype=float)))),
            "FirstDate": macro_data["score_series"].index.min().date().isoformat() if not macro_data["score_series"].empty else "n/a",
            "LastDate": macro_data["score_series"].index.max().date().isoformat() if not macro_data["score_series"].empty else "n/a",
            "Warning": macro_data.get("warning") or "",
        }
    )
    return pd.DataFrame(rows)


def synthetic_ohlcv_from_close(close: pd.Series) -> pd.DataFrame:
    base_open = close.shift(1).fillna(close.iloc[0]) * 1.001
    high = pd.concat([base_open, close], axis=1).max(axis=1) + 0.8
    low = pd.concat([base_open, close], axis=1).min(axis=1) - 0.8
    volume = pd.Series(np.linspace(1_000_000, 1_500_000, len(close)), index=close.index)
    return pd.DataFrame({"Open": base_open, "High": high, "Low": low, "Close": close, "Volume": volume})


def run_benchmark() -> dict[str, Any]:
    rows = 5_000
    rng = np.random.default_rng(ML_RANDOM_STATE)
    index = pd.date_range("2010-01-04", periods=rows, freq="B")
    close = pd.Series(300 + rng.normal(0.04, 1.9, rows).cumsum(), index=index)
    frame = synthetic_ohlcv_from_close(close)
    macro = pd.Series(50 + rng.normal(0, 10, rows).cumsum() / 25.0, index=index, name="FearGreed")
    benchmark_close = close * 0.95

    started_at = time.perf_counter()
    indicators = compute_indicators(frame, macro, benchmark_close, macro / 100.0, ML_TARGET_SYMBOL)
    indicators_seconds = time.perf_counter() - started_at

    started_at = time.perf_counter()
    state_frame = build_state_frame(indicators, DEFAULT_SIGNAL_PRESET)
    state_seconds = time.perf_counter() - started_at

    started_at = time.perf_counter()
    _, _, replay_summary = build_trade_replay_from_state_frame(ML_TARGET_SYMBOL, attach_rule_only_columns(state_frame), DEFAULT_SIGNAL_PRESET)
    replay_seconds = time.perf_counter() - started_at

    download_symbols = list(ML_FEATURE_TICKERS[:10])
    sequential_started_at = time.perf_counter()
    for symbol in download_symbols:
        quiet_yfinance_download(
            symbol,
            start="2023-01-01",
            end=date.today().isoformat(),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    sequential_seconds = time.perf_counter() - sequential_started_at

    batch_started_at = time.perf_counter()
    quiet_yfinance_download(
        download_symbols,
        start="2023-01-01",
        end=date.today().isoformat(),
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    batch_seconds = time.perf_counter() - batch_started_at

    baseline_state_seconds = 2.05
    return {
        "rows": rows,
        "compute_indicators_seconds": round(indicators_seconds, 4),
        "build_state_frame_seconds": round(state_seconds, 4),
        "build_state_improvement_vs_baseline_pct": round((baseline_state_seconds - state_seconds) / baseline_state_seconds * 100.0, 2),
        "replay_seconds": round(replay_seconds, 4),
        "download_symbols": download_symbols,
        "sequential_download_seconds": round(sequential_seconds, 4),
        "batch_download_seconds": round(batch_seconds, 4),
        "batch_speedup_x": round(sequential_seconds / batch_seconds, 2) if batch_seconds > 0 else None,
        "replay_summary": replay_summary,
    }


def run_benchmark_cli(json_output: bool = False) -> int:
    benchmark = run_benchmark()
    if json_output:
        print(json.dumps(benchmark, ensure_ascii=False, default=str, indent=2))
        return 0
    print("BENCHMARK")
    print(f"rows={benchmark['rows']}")
    print(f"compute_indicators_seconds={benchmark['compute_indicators_seconds']}")
    print(f"build_state_frame_seconds={benchmark['build_state_frame_seconds']}")
    print(f"build_state_improvement_vs_baseline_pct={benchmark['build_state_improvement_vs_baseline_pct']}")
    print(f"replay_seconds={benchmark['replay_seconds']}")
    print(f"sequential_download_seconds={benchmark['sequential_download_seconds']}")
    print(f"batch_download_seconds={benchmark['batch_download_seconds']}")
    print(f"batch_speedup_x={benchmark['batch_speedup_x']}")
    print(f"replay_summary={benchmark['replay_summary']}")
    return 0


def run_ml_backtest(symbol: str, start_iso: str, end_iso: str) -> dict[str, Any]:
    symbol = symbol.strip().upper()
    preset = DEFAULT_SIGNAL_PRESET
    macro_data = load_macro_fear_greed_cached(end_iso)
    requested_tickers = [symbol]
    if symbol == ML_TARGET_SYMBOL:
        requested_tickers = dedupe_preserve_order(requested_tickers + list(ML_BATCH_TICKERS))
    price_payloads = load_price_payloads(requested_tickers, start_iso, end_iso)
    payload = price_payloads.get(symbol, {"frame": pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS)), "resolved_symbol": symbol, "warning": f"{symbol} payload unavailable"})
    analysis = analyze_symbol_state(
        symbol,
        payload,
        macro_data,
        preset,
        feature_payloads=price_payloads if symbol == ML_TARGET_SYMBOL else None,
    )
    final_state_frame = attach_rule_only_columns(analysis["state_frame"])
    baseline_state_frame = attach_rule_only_columns(analysis["baseline_state_frame"])
    ml_bundle = analysis["ml_bundle"]
    oos_start = ml_bundle.get("oos_start") or final_state_frame.index.min()
    final_slice = final_state_frame.loc[oos_start:].copy() if oos_start is not None else final_state_frame.copy()
    baseline_slice = baseline_state_frame.loc[oos_start:].copy() if oos_start is not None else baseline_state_frame.copy()
    _, _, rule_summary = build_trade_replay_from_state_frame(symbol, baseline_slice, preset)
    _, _, ml_summary = build_trade_replay_from_state_frame(symbol, final_slice, preset)
    return {
        "symbol": symbol,
        "start": start_iso,
        "end": end_iso,
        "oos_start": None if oos_start is None else pd.Timestamp(oos_start).date().isoformat(),
        "model_mode": ml_bundle.get("model_mode", "rules_only"),
        "sell_veto_profile": ml_bundle.get("comparison", {}).get("sell_veto_profile", ML_DEFAULT_SELL_VETO_PROFILE),
        "ml_enabled": bool(ml_bundle.get("enabled")),
        "reason": ml_bundle.get("reason"),
        "rule_summary": rule_summary,
        "ml_summary": ml_summary,
        "selection_frame": ml_bundle.get("selection_frame", pd.DataFrame()).to_dict("records"),
    }


def run_ml_backtest_cli(symbol: str, start_iso: str, end_iso: str, json_output: bool = False) -> int:
    result = run_ml_backtest(symbol, start_iso, end_iso)
    if json_output:
        print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
        return 0
    print("ML_BACKTEST")
    print(f"symbol={result['symbol']}")
    print(f"start={result['start']} end={result['end']} oos_start={result['oos_start']}")
    print(f"model_mode={result['model_mode']} sell_veto_profile={result['sell_veto_profile']} ml_enabled={result['ml_enabled']}")
    print(f"reason={result['reason']}")
    print(f"rule_summary={result['rule_summary']}")
    print(f"ml_summary={result['ml_summary']}")
    return 0


def run_self_tests() -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        results.append((name, passed, detail))

    try:
        down_close = pd.Series(np.linspace(200, 100, 80), index=pd.date_range("2023-01-02", periods=80, freq="B"))
        down_frame = synthetic_ohlcv_from_close(down_close)
        down_td = compute_td_sequential(down_frame)
        passed = int(down_td["BuySetup"].max()) >= 9 and int(down_td["BuyCountdown"].max()) >= 13
        record("td_buy_exhaustion", passed, f"max BuySetup={int(down_td['BuySetup'].max())}, max BuyCountdown={int(down_td['BuyCountdown'].max())}")
    except Exception as exc:
        record("td_buy_exhaustion", False, str(exc))

    try:
        up_close = pd.Series(np.linspace(100, 200, 80), index=pd.date_range("2023-01-02", periods=80, freq="B"))
        up_frame = synthetic_ohlcv_from_close(up_close)
        up_td = compute_td_sequential(up_frame)
        passed = int(up_td["SellSetup"].max()) >= 9 and int(up_td["SellCountdown"].max()) >= 13
        record("td_sell_exhaustion", passed, f"max SellSetup={int(up_td['SellSetup'].max())}, max SellCountdown={int(up_td['SellCountdown'].max())}")
    except Exception as exc:
        record("td_sell_exhaustion", False, str(exc))

    try:
        x_axis = np.linspace(0, 10 * np.pi, 320)
        cycle_close = pd.Series(
            120 + np.linspace(0, 12, 320) + (9 * np.sin(x_axis)),
            index=pd.date_range("2021-01-04", periods=320, freq="B"),
        )
        cycle_frame = synthetic_ohlcv_from_close(cycle_close)
        cycle_result = compute_stl_cycle(cycle_frame)
        valid = cycle_result["CycleScore"].notna()
        trough_mask = pd.Series(np.sin(x_axis), index=cycle_result.index) < -0.90
        peak_mask = pd.Series(np.sin(x_axis), index=cycle_result.index) > 0.90
        trough_score = cycle_result.loc[valid & trough_mask, "CycleScore"].mean()
        peak_score = cycle_result.loc[valid & peak_mask, "CycleScore"].mean()
        passed = bool(pd.notna(trough_score) and pd.notna(peak_score) and trough_score + 15 < peak_score)
        record("stl_cycle_extremes", passed, f"trough={trough_score:.2f}, peak={peak_score:.2f}")
    except Exception as exc:
        record("stl_cycle_extremes", False, str(exc))

    try:
        trend_close = pd.Series(np.linspace(100, 160, 120), index=pd.date_range("2022-01-03", periods=120, freq="B"))
        trend_frame = synthetic_ohlcv_from_close(trend_close)
        indicators = compute_indicators(trend_frame, None)
        atr_ready = indicators["ATR"].dropna()
        stop_distance = indicators["StopDistance"].dropna()
        stretch = indicators["ATRStretch"].dropna()
        passed = bool((atr_ready > 0).all() and stop_distance.median() > 0 and stretch.iloc[-1] > 0)
        record("atr_and_stop_direction", passed, f"atr_count={len(atr_ready)}, median_stop_distance={stop_distance.median():.4f}, last_stretch={stretch.iloc[-1]:.4f}")
    except Exception as exc:
        record("atr_and_stop_direction", False, str(exc))

    try:
        momentum_index = pd.date_range("2023-01-02", periods=100, freq="B")
        up_momentum = calc_13612w_momentum(pd.Series(np.linspace(100, 130, 100), index=momentum_index)).dropna()
        down_momentum = calc_13612w_momentum(pd.Series(np.linspace(130, 100, 100), index=momentum_index)).dropna()
        passed = bool(not up_momentum.empty and not down_momentum.empty and up_momentum.iloc[-1] > 0 and down_momentum.iloc[-1] < 0)
        record("tips_13612w_momentum_sign", passed, f"up_last={up_momentum.iloc[-1]:.4f}, down_last={down_momentum.iloc[-1]:.4f}")
    except Exception as exc:
        record("tips_13612w_momentum_sign", False, str(exc))

    try:
        preset = PROFILE_PRESETS["Balanced"]
        replay_index = pd.date_range("2024-01-02", periods=6, freq="B")
        replay_state = pd.DataFrame(
            {
                "Close": [100.0, 101.0, 103.0, 110.0, 107.0, 98.0],
                "ChandelierStop": [90.0, 91.0, 92.0, 94.0, 95.0, 100.0],
                "State": ["Hold / Neutral", "Strong Buy", "Buy", "Weak Sell", "Hold / Neutral", "Strong Sell"],
                "BuyScore": [15, 81, 78, 34, 30, 20],
                "SellScore": [18, 20, 25, 62, 40, 82],
                "BuyTrigger": [False, True, False, False, False, False],
                "SellTrigger": [False, False, False, False, False, True],
                "SellSetupActive": [False, False, False, True, True, True],
                "SignalNote": ["n"] * 6,
            },
            index=replay_index,
        )
        events, trades, _ = build_trade_replay_from_state_frame("TEST", replay_state, preset)
        expected_signals = ["BUY", "PARTIAL SELL", "SELL"]
        passed = (events["Signal"].tolist() == expected_signals) and (len(trades) == 1) and (trades.iloc[0]["Status"] == "Closed")
        record("replay_sequence", passed, f"signals={events['Signal'].tolist()}")
    except Exception as exc:
        record("replay_sequence", False, str(exc))

    try:
        preset = PROFILE_PRESETS["Aggressive"]
        cooldown_index = pd.date_range("2024-03-01", periods=6, freq="B")
        cooldown_state = pd.DataFrame(
            {
                "Close": [100.0, 98.0, 99.0, 101.0, 102.0, 103.0],
                "ChandelierStop": [90.0, 99.0, 90.0, 90.0, 90.0, 90.0],
                "State": ["Strong Buy", "Strong Sell", "Buy", "Hold / Neutral", "Strong Buy", "Hold / Neutral"],
                "BuyScore": [80, 20, 78, 30, 82, 25],
                "SellScore": [20, 84, 25, 25, 20, 20],
                "BuyTrigger": [True, False, True, False, True, False],
                "SellTrigger": [False, True, False, False, False, False],
                "SellSetupActive": [False, True, False, False, False, False],
                "SignalNote": ["n"] * 6,
            },
            index=cooldown_index,
        )
        events, _, _ = build_trade_replay_from_state_frame("TEST2", cooldown_state, preset)
        signals = events["Signal"].tolist()
        passed = signals == ["BUY", "SELL", "BUY"] and signals.count("SELL") == 1
        record("replay_cooldown_and_orphan_guard", passed, f"signals={signals}")
    except Exception as exc:
        record("replay_cooldown_and_orphan_guard", False, str(exc))

    try:
        preset = PROFILE_PRESETS["Balanced"]
        whipsaw_index = pd.date_range("2024-05-01", periods=6, freq="B")
        whipsaw_state = pd.DataFrame(
            {
                "Close": [100.0, 102.0, 101.0, 100.5, 99.0, 98.0],
                "Low": [99.0, 101.0, 100.8, 100.0, 98.8, 97.5],
                "ChandelierStop": [92.0, 94.0, 95.0, 95.0, 95.0, 95.0],
                "State": ["Hold / Neutral", "Buy", "Sell", "Weak Sell", "Sell", "Strong Sell"],
                "BuyScore": [25, 74, 45, 38, 22, 10],
                "SellScore": [20, 18, 72, 60, 82, 92],
                "BuyTrigger": [False, True, False, False, False, False],
                "SellTrigger": [False, False, True, False, True, True],
                "SellSetupActive": [False, False, True, True, True, True],
                "StopTouched": [False, False, False, False, False, True],
                "SignalNote": ["n"] * 6,
            },
            index=whipsaw_index,
        )
        events, _, _ = build_trade_replay_from_state_frame("WHIP", whipsaw_state, preset)
        signals = events["Signal"].tolist()
        first_sell_date = events.loc[events["Signal"] == "SELL", "Date"].iloc[0] if (events["Signal"] == "SELL").any() else None
        passed = bool(signals and signals[0] == "BUY" and first_sell_date is not None and pd.Timestamp(first_sell_date) >= whipsaw_index[4])
        record("whipsaw_guard", passed, f"signals={signals}, first_sell={first_sell_date}")
    except Exception as exc:
        record("whipsaw_guard", False, str(exc))

    try:
        preset = PROFILE_PRESETS["Balanced"]
        marker_index = pd.date_range("2024-04-01", periods=7, freq="B")
        marker_state = pd.DataFrame(
            {
                "Close": [100.0, 101.0, 103.0, 104.0, 102.0, 97.0, 96.0],
                "ChandelierStop": [90.0, 90.0, 91.0, 92.0, 95.0, 98.0, 99.0],
                "State": ["Hold / Neutral", "Strong Buy", "Buy", "Weak Buy", "Hold / Neutral", "Sell", "Strong Sell"],
                "BuyScore": [20, 80, 79, 60, 44, 18, 15],
                "SellScore": [15, 20, 21, 30, 45, 82, 84],
                "BuyTrigger": [False, True, False, False, False, False, False],
                "SellTrigger": [False, False, False, False, False, True, False],
            },
            index=marker_index,
        )
        chart_signals = build_chart_signal_frame(marker_state, preset)
        signals = chart_signals["Signal"].tolist()
        passed = signals == ["BUY", "SELL"]
        record("chart_signal_visibility", passed, f"signals={signals}")
    except Exception as exc:
        record("chart_signal_visibility", False, str(exc))

    try:
        plain = TickerSnapshot(
            ticker="SPY",
            resolved_symbol="SPY",
            as_of=pd.Timestamp("2024-01-02"),
            last_close=100.0,
            daily_return=0.01,
            state="Buy",
            buy_score=70,
            sell_score=20,
            cycle_score=10.0,
            td_label="Buy setup 9",
            atr_stretch=-1.2,
            rsi=35.0,
            fear_greed=42.0,
            stop_distance=0.05,
            note="n",
            warning=None,
        )
        resolved = TickerSnapshot(
            ticker="005930",
            resolved_symbol="005930.KS",
            as_of=pd.Timestamp("2024-01-02"),
            last_close=100.0,
            daily_return=0.01,
            state="Buy",
            buy_score=70,
            sell_score=20,
            cycle_score=10.0,
            td_label="Buy setup 9",
            atr_stretch=-1.2,
            rsi=35.0,
            fear_greed=42.0,
            stop_distance=0.05,
            note="n",
            warning=None,
        )
        passed = format_snapshot_label(plain) == "SPY" and format_snapshot_label(resolved) == "005930 (005930.KS)"
        record("drilldown_label_alignment", passed, f"plain={format_snapshot_label(plain)}, resolved={format_snapshot_label(resolved)}")
    except Exception as exc:
        record("drilldown_label_alignment", False, str(exc))

    try:
        splitter = TimeSeriesSplit(n_splits=ML_CV_SPLITS, test_size=ML_CV_TEST_SIZE, gap=ML_CV_GAP)
        leakage_free = True
        for train_idx, test_idx in splitter.split(np.arange(600)):
            if int(train_idx[-1]) + ML_CV_GAP >= int(test_idx[0]):
                leakage_free = False
                break
        record("timeseries_gap_no_leakage", leakage_free, f"gap={ML_CV_GAP}, leakage_free={leakage_free}")
    except Exception as exc:
        record("timeseries_gap_no_leakage", False, str(exc))

    try:
        label_close = pd.Series(100 + np.sin(np.linspace(0, 14, 260)).cumsum(), index=pd.date_range("2023-01-02", periods=260, freq="B"))
        label_frame = synthetic_ohlcv_from_close(label_close)
        macro_series = pd.Series(50.0, index=label_frame.index)
        indicators = compute_indicators(label_frame, macro_series, label_close * 0.98, macro_series / 100.0, ML_TARGET_SYMBOL)
        state_frame = attach_rule_only_columns(build_state_frame(indicators, DEFAULT_SIGNAL_PRESET))
        labels = build_ml_label_frame(state_frame, DEFAULT_SIGNAL_PRESET)
        passed = (
            not labels.empty
            and {"BuyCandidate", "SellCandidate", "BuyLabel", "SellLabel", "SellEdge"}.issubset(labels.columns)
            and labels["BuyCandidate"].dtype == bool
            and labels["SellCandidate"].dtype == bool
        )
        record("label_generation_stability", passed, f"buy_candidates={int(labels['BuyCandidate'].sum())}, sell_candidates={int(labels['SellCandidate'].sum())}")
    except Exception as exc:
        record("label_generation_stability", False, str(exc))

    try:
        descriptor = resolve_boosting_descriptor(prefer_lightgbm=False)
        passed = descriptor.name == "hist_gradient_boosting"
        record("fallback_model_selection", passed, f"descriptor={descriptor.name}")
    except Exception as exc:
        record("fallback_model_selection", False, str(exc))

    try:
        stop_index = pd.date_range("2024-06-03", periods=3, freq="B")
        stop_state = pd.DataFrame(
            {
                "Close": [100.0, 98.0, 95.0],
                "Open": [100.0, 98.5, 95.5],
                "High": [101.0, 99.0, 96.0],
                "Low": [99.0, 97.0, 94.0],
                "State": ["Buy", "Strong Sell", "Strong Sell"],
                "BuyScore": [75, 25, 20],
                "SellScore": [20, 90, 92],
                "BuyTrigger": [True, False, False],
                "SellTrigger": [False, True, True],
                "SellSetupActive": [False, True, True],
                "StopExitConfirmed": [False, True, True],
                "StopEmergencyExit": [False, True, True],
                "SignalNote": ["", "", ""],
            },
            index=stop_index,
        )
        filtered = apply_ml_thresholds_to_state_frame(
            attach_rule_only_columns(stop_state),
            pd.Series([0.9, 0.1, 0.1], index=stop_index),
            pd.Series([0.1, 0.1, 0.1], index=stop_index),
            0.8,
            0.8,
            model_mode="random_forest",
        )
        passed = bool(filtered["SellTrigger"].iloc[1] and filtered["State"].iloc[1] == "Strong Sell")
        record("hard_stop_non_rejection", passed, f"sell_trigger={filtered['SellTrigger'].iloc[1]}, state={filtered['State'].iloc[1]}")
    except Exception as exc:
        record("hard_stop_non_rejection", False, str(exc))

    try:
        veto_index = pd.date_range("2024-07-01", periods=3, freq="B")
        veto_state = pd.DataFrame(
            {
                "Close": [100.0, 103.0, 104.5],
                "Open": [99.8, 102.5, 104.0],
                "High": [100.5, 103.5, 105.0],
                "Low": [99.5, 102.0, 103.5],
                "State": ["Buy", "Sell", "Strong Sell"],
                "BuyScore": [74, 42, 20],
                "SellScore": [20, 64, 86],
                "BuyTrigger": [True, False, False],
                "SellTrigger": [False, True, True],
                "SellSetupActive": [False, True, True],
                "StopExitConfirmed": [False, False, False],
                "StopEmergencyExit": [False, False, True],
                "Regime": ["Bull Trend", "Bull Trend", "Bull Trend"],
                "CycleScore": [18.0, 42.0, 72.0],
                "FearGreed": [58.0, 61.0, 61.0],
                "RSPercentile": [88.0, 91.0, 91.0],
                "SignalNote": ["", "", ""],
            },
            index=veto_index,
        )
        filtered = apply_ml_thresholds_to_state_frame(
            attach_rule_only_columns(veto_state),
            pd.Series([0.9, 0.9, 0.1], index=veto_index),
            pd.Series([0.1, 0.9, 0.1], index=veto_index),
            0.8,
            0.8,
            model_mode="random_forest",
            sell_veto_profile="bull_cycle55",
        )
        passed = (
            bool(not filtered["SellTrigger"].iloc[1])
            and str(filtered["StrategySource"].iloc[1]) == "MLSellVeto"
            and bool(filtered["SellTrigger"].iloc[2])
        )
        record(
            "bull_sell_veto_only_blocks_discretionary_exits",
            passed,
            (
                f"blocked={filtered['SellTrigger'].iloc[1]}, "
                f"source={filtered['StrategySource'].iloc[1]}, "
                f"hard_stop={filtered['SellTrigger'].iloc[2]}"
            ),
        )
    except Exception as exc:
        record("bull_sell_veto_only_blocks_discretionary_exits", False, str(exc))

    try:
        rows = 180
        feature_index = pd.date_range("2023-01-02", periods=rows, freq="B")
        feature_frame = pd.DataFrame(
            {
                "F1": np.linspace(-2, 2, rows),
                "F2": np.sin(np.linspace(0, 6 * np.pi, rows)),
                "F3": np.cos(np.linspace(0, 5 * np.pi, rows)),
            },
            index=feature_index,
        )
        candidate_mask = pd.Series(True, index=feature_index)
        label_series = pd.Series(([0] * 90) + ([1] * 90), index=feature_index)
        first_train, first_test, _ = build_probability_views_for_target(
            feature_frame,
            candidate_mask,
            label_series,
            feature_frame.iloc[0:0],
            pd.Series(dtype=bool),
            ["F1", "F2", "F3"],
            prefer_lightgbm=False,
        )
        second_train, second_test, _ = build_probability_views_for_target(
            feature_frame,
            candidate_mask,
            label_series,
            feature_frame.iloc[0:0],
            pd.Series(dtype=bool),
            ["F1", "F2", "F3"],
            prefer_lightgbm=False,
        )
        passed = all(
            np.allclose(first_train[name], second_train[name], equal_nan=True)
            for name in first_train
        ) and all(
            np.allclose(first_test[name], second_test[name], equal_nan=True)
            for name in first_test
        )
        record("fixed_random_state_reproducibility", passed, "probability views stable across repeated fits")
    except Exception as exc:
        record("fixed_random_state_reproducibility", False, str(exc))

    return results


def run_self_tests_cli() -> int:
    results = run_self_tests()
    passed_count = 0
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        if passed:
            passed_count += 1
        print(f"{status} {name}: {detail}")
    total = len(results)
    failed_count = total - passed_count
    print(f"SUMMARY passed={passed_count} failed={failed_count} total={total}")
    return 0 if failed_count == 0 else 1


def main() -> None:
    run_started_at = time.perf_counter()
    configure_page()
    apply_style()
    watchlist, history_years, preset, refresh = build_controls()
    progress_total_steps = 2 + len(watchlist) + 2 + (1 if ML_TARGET_SYMBOL in watchlist else 0)
    progress_tracker = StreamlitProgressTracker(progress_total_steps)
    timing_rows: list[dict[str, Any]] = []

    def add_timing(stage: str, target: str, started_at: float, rows: int | None = None) -> None:
        timing_rows.append(
            {
                "Stage": stage,
                "Target": target,
                "Seconds": time.perf_counter() - started_at,
                "Rows": rows if rows is not None else pd.NA,
            }
        )

    if refresh:
        st.cache_data.clear()
        st.sidebar.success("Cached data cleared. Rerun to fetch fresh prices.")

    end_date = date.today()
    start_date = end_date - timedelta(days=(history_years * 365) + 30)
    price_payloads: dict[str, dict[str, Any]] = {}
    full_price_payloads: dict[str, dict[str, Any]] = {}

    try:
        progress_tracker.update(
            "Loading macro context",
            f"Fetching Fear & Greed and macro basket through {end_date.isoformat()}.",
            advance=0,
        )
        macro_started_at = time.perf_counter()
        macro_data = load_macro_fear_greed_cached(end_date.isoformat())
        add_timing("load", "macro_fear_greed", macro_started_at, len(macro_data.get("score_series", pd.Series(dtype=float))))
        progress_tracker.update(
            "Macro context ready",
            f"Loaded {len(macro_data.get('score_series', pd.Series(dtype=float)))} macro rows.",
            advance=1,
        )

        download_started_at = time.perf_counter()
        requested_tickers = watchlist.copy()
        if ML_TARGET_SYMBOL in watchlist:
            requested_tickers = dedupe_preserve_order(requested_tickers + list(ML_BATCH_TICKERS))
        progress_tracker.update(
            "Downloading price history",
            f"Requesting {len(requested_tickers)} ticker payloads: {', '.join(requested_tickers[:8])}{' ...' if len(requested_tickers) > 8 else ''}",
            advance=0,
        )
        full_price_payloads = load_price_payloads(requested_tickers, start_date.isoformat(), end_date.isoformat())
        total_rows_downloaded = sum(len(payload.get("frame", pd.DataFrame())) for payload in full_price_payloads.values())
        add_timing("download_batch", "watchlist_and_features", download_started_at, total_rows_downloaded)
        price_payloads = {ticker: full_price_payloads.get(ticker, download_single_history_cached(ticker, start_date.isoformat(), end_date.isoformat())) for ticker in watchlist}
        progress_tracker.update(
            "Price download ready",
            f"Received {len(full_price_payloads)} payloads and {total_rows_downloaded} total rows.",
            advance=1,
        )
    except Exception as exc:
        progress_tracker.fail(f"Dashboard data could not be loaded: {exc}")
        st.error("Dashboard data could not be loaded.")
        st.info(str(exc))
        st.stop()

    state_frames: dict[str, pd.DataFrame] = {}
    baseline_state_frames: dict[str, pd.DataFrame] = {}
    ml_bundles: dict[str, dict[str, Any]] = {}
    snapshots: list[TickerSnapshot] = []
    warnings: list[str] = []
    macro_warning = macro_data.get("warning")
    if macro_warning:
        warnings.append(str(macro_warning))

    cached_qqq_analysis: dict[str, Any] | None = None
    if ML_TARGET_SYMBOL in watchlist:
        progress_tracker.update(
            "Building QQQ ML bundle",
            "Training and selecting the walk-forward ML filter for QQQ.",
            advance=0,
        )
        qqq_ml_started_at = time.perf_counter()
        cached_qqq_analysis = load_cached_qqq_analysis(start_date.isoformat(), end_date.isoformat(), preset.name)
        add_timing("ml_bundle", ML_TARGET_SYMBOL, qqq_ml_started_at, len(cached_qqq_analysis.get("state_frame", pd.DataFrame())))
        progress_tracker.update(
            "QQQ ML bundle ready",
            f"Cached ML bundle mode: {cached_qqq_analysis.get('ml_bundle', {}).get('model_mode', 'rules_only')}.",
            advance=1,
        )

    for position, ticker in enumerate(watchlist, start=1):
        progress_tracker.update(
            "Computing ticker state",
            f"[{position}/{len(watchlist)}] Processing {ticker}: indicators, rules, and replay state.",
            advance=0,
        )
        compute_started_at = time.perf_counter()
        payload = price_payloads[ticker]
        price_frame = payload["frame"]
        warning = payload.get("warning")
        if warning:
            warnings.append(f"{ticker}: {warning}")
        if price_frame.empty:
            progress_tracker.update(
                "Ticker skipped",
                f"{ticker}: no usable price history was returned.",
                advance=1,
            )
            continue
        if ticker == ML_TARGET_SYMBOL and cached_qqq_analysis is not None:
            analysis = cached_qqq_analysis
        else:
            analysis = analyze_symbol_state(ticker, payload, macro_data, preset)
        state_frame = analysis["state_frame"]
        if state_frame.empty:
            warnings.append(f"{ticker}: indicator stack could not build a usable state frame")
            progress_tracker.update(
                "Ticker computation failed",
                f"{ticker}: indicator stack could not build a usable state frame.",
                advance=1,
            )
            continue
        baseline_state_frames[ticker] = analysis["baseline_state_frame"]
        state_frames[ticker] = state_frame
        ml_bundles[ticker] = analysis["ml_bundle"]
        snapshots.append(analysis["snapshot"])
        add_timing("compute", ticker, compute_started_at, len(state_frame))
        progress_tracker.update(
            "Ticker computed",
            f"{ticker}: {len(state_frame)} rows, latest state {analysis['snapshot'].state}, resolved {analysis['snapshot'].resolved_symbol}.",
            advance=1,
        )

    watchlist_frame = build_watchlist_frame(snapshots)
    render_header(macro_data, snapshots)
    render_top_metrics(watchlist_frame, macro_data, preset)
    render_warning_panel(warnings)
    render_watchlist_summary(watchlist_frame)

    diagnostics = build_diagnostics_frame(watchlist, price_payloads, macro_data)
    performance_frame = build_performance_frame(timing_rows)
    if watchlist_frame.empty:
        add_timing("total", "app_run", run_started_at)
        performance_frame = build_performance_frame(timing_rows)
        render_diagnostics(diagnostics, performance_frame)
        st.stop()

    snapshot_by_ticker = {snapshot.ticker: snapshot for snapshot in snapshots}
    ticker_options = [ticker for ticker in watchlist_frame["ticker"].tolist() if ticker in snapshot_by_ticker]
    selected_ticker = st.selectbox(
        "Drill-down ticker",
        options=ticker_options,
        index=0,
        format_func=lambda ticker: format_snapshot_label(snapshot_by_ticker[ticker]),
    )
    selected_snapshot = snapshot_by_ticker[selected_ticker]
    selected_state_frame = state_frames[selected_ticker]
    selected_baseline_state_frame = baseline_state_frames.get(selected_ticker, selected_state_frame)
    selected_ml_bundle = ml_bundles.get(selected_ticker, {})
    progress_tracker.update(
        "Building replay summary",
        f"Preparing trade replay and summary cards for {selected_ticker}.",
        advance=0,
    )
    replay_started_at = time.perf_counter()
    event_frame, trade_frame, replay_summary = build_trade_replay_from_state_frame(selected_ticker, selected_state_frame, preset)
    add_timing("replay", selected_ticker, replay_started_at, len(event_frame))
    progress_tracker.update(
        "Replay summary ready",
        f"{selected_ticker}: {len(event_frame)} events, {int(replay_summary.get('closed_trades') or 0)} closed trades.",
        advance=1,
    )
    progress_tracker.update(
        "Preparing chart markers",
        f"Building buy/sell markers for {selected_ticker} price context chart.",
        advance=0,
    )
    chart_started_at = time.perf_counter()
    chart_signal_frame = event_frame.loc[event_frame["Signal"].isin(["BUY", "SELL"])].copy() if not event_frame.empty else build_chart_signal_frame(selected_state_frame, preset)
    add_timing("chart_signals", selected_ticker, chart_started_at, len(chart_signal_frame))
    baseline_summary = build_trade_replay_from_state_frame(selected_ticker, selected_baseline_state_frame, preset)[2] if not selected_baseline_state_frame.empty else {}
    add_timing("total", "app_run", run_started_at)
    performance_frame = build_performance_frame(timing_rows)
    progress_tracker.update(
        "Chart markers ready",
        f"{selected_ticker}: {len(chart_signal_frame)} markers prepared for display.",
        advance=1,
    )
    progress_tracker.finish(
        f"Dashboard run completed in {time.perf_counter() - run_started_at:.2f}s. Selected ticker: {selected_ticker}."
    )
    comparison = selected_ml_bundle.get("comparison", {}).copy()
    if comparison and "rule_summary" not in comparison:
        comparison["rule_summary"] = baseline_summary
    render_ticker_panel(
        selected_ticker=selected_ticker,
        snapshot=selected_snapshot,
        state_frame=selected_state_frame,
        chart_signal_frame=chart_signal_frame,
        event_frame=event_frame,
        trade_frame=trade_frame,
        replay_summary=replay_summary,
        preset=preset,
        comparison=comparison,
        selection_frame=selected_ml_bundle.get("selection_frame", pd.DataFrame()),
    )
    render_diagnostics(diagnostics, performance_frame)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--self-test", action="store_true", help="Run built-in synthetic validation tests.")
    parser.add_argument("--benchmark", action="store_true", help="Run local performance benchmark.")
    parser.add_argument("--ml-backtest", action="store_true", help="Run strict out-of-sample ML backtest.")
    parser.add_argument("--symbol", default=ML_TARGET_SYMBOL, help="Ticker symbol for --ml-backtest.")
    parser.add_argument("--start", default="2016-01-01", help="Start date for --ml-backtest in YYYY-MM-DD format.")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date for --ml-backtest in YYYY-MM-DD format.")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit machine-readable JSON output for CLI commands.")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.self_test:
        raise SystemExit(run_self_tests_cli())
    if args.benchmark:
        raise SystemExit(run_benchmark_cli(json_output=args.json_output))
    if args.ml_backtest:
        raise SystemExit(run_ml_backtest_cli(args.symbol, args.start, args.end, json_output=args.json_output))
    main()
