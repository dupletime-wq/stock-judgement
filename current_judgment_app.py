from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import hashlib
import io
import json
import logging
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from statsmodels.tsa.seasonal import STL
import urllib3
import yfinance as yf


if any(flag in sys.argv for flag in ("--self-test", "--benchmark", "--rule-backtest", "--optimize-report")):
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)


APP_TITLE = "Rule-Only Hold-Beat Dashboard"
CACHE_TTL_SECONDS = 3600
MACRO_HISTORY_YEARS = 6
CORE_TICKERS = ("SPY", "QQQ", "069500.KS")
CORE_TICKER_SET = set(CORE_TICKERS)
DEFAULT_WATCHLIST = CORE_TICKERS
REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
MACRO_TICKERS = ("SPY", "^VIX", "HYG", "IEF", "RSP", "XLY", "XLP", "UUP", "TIPS")
VALIDATION_WINDOWS: tuple[tuple[str, int | None], ...] = (
    ("1Y", 365),
    ("3Y", 365 * 3),
    ("5Y", 365 * 5),
    ("Full", None),
)
OPTIMIZER_START_ISO = "2004-01-01"
TRANSACTION_COST_PER_SIDE = 0.001
RANDOM_STATE = 42
EMA_FAST_OPTIONS = (5, 8, 10, 12, 20)
EMA_SLOW_OPTIONS = (20, 30, 50, 100, 200)
EMA_ALL_SPANS = tuple(sorted(set(EMA_FAST_OPTIONS + EMA_SLOW_OPTIONS)))
OPTIMIZER_STATE_KEY = "rule_optimizer_state"
DEFAULT_HISTORY_YEARS = 5
DEFAULT_SEARCH_BUDGET = "Standard"
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
SEARCH_BUDGET_SPECS: dict[str, dict[str, Any]] = {
    "Quick": {
        "coarse_fast": (8,),
        "coarse_slow": (20, 50),
        "buy_thresholds": (58,),
        "sell_thresholds": (56,),
        "buy_votes": (2,),
        "sell_votes": (2,),
        "bottom_votes": (2,),
        "toggle_options": (
            (False, False, True),
            (True, False, True),
        ),
        "batch_size": 4,
        "local_top_n": 1,
        "local_limit": 12,
    },
    "Standard": {
        "coarse_fast": EMA_FAST_OPTIONS,
        "coarse_slow": EMA_SLOW_OPTIONS,
        "buy_thresholds": (52, 58, 64, 70),
        "sell_thresholds": (50, 56, 62, 68),
        "buy_votes": (1, 2, 3),
        "sell_votes": (1, 2, 3),
        "bottom_votes": (2, 3),
        "toggle_options": (
            (False, False, False),
            (False, False, True),
            (True, False, True),
            (False, True, True),
            (True, True, True),
            (True, True, False),
        ),
        "batch_size": 24,
        "local_top_n": 5,
        "local_limit": 360,
    },
    "Exhaustive": {
        "coarse_fast": EMA_FAST_OPTIONS,
        "coarse_slow": EMA_SLOW_OPTIONS,
        "buy_thresholds": (50, 54, 58, 62, 66, 70),
        "sell_thresholds": (48, 52, 56, 60, 64, 68),
        "buy_votes": (1, 2, 3),
        "sell_votes": (1, 2, 3),
        "bottom_votes": (2, 3),
        "toggle_options": (
            (False, False, False),
            (False, False, True),
            (True, False, False),
            (True, False, True),
            (False, True, False),
            (False, True, True),
            (True, True, False),
            (True, True, True),
        ),
        "batch_size": 36,
        "local_top_n": 8,
        "local_limit": None,
    },
}

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
class RuleStrategyConfig:
    fast_ema: int
    slow_ema: int
    bottom_required: int
    buy_confirmation_required: int
    sell_confirmation_required: int
    buy_score_threshold: int
    sell_score_threshold: int
    require_macd_for_buy: bool
    require_volume_for_buy: bool
    allow_reclaim_trigger: bool

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ValidationRow:
    ticker: str
    window: str
    start_date: str
    end_date: str
    strategy_total_return: float | None
    buy_hold_return: float | None
    excess_return: float | None
    max_drawdown: float | None
    buy_hold_max_drawdown: float | None
    cagr: float | None
    buy_hold_cagr: float | None
    win_rate: float | None
    closed_trades: int
    pass_case: bool
    fail_reason: str | None


@dataclass(frozen=True)
class OptimizerEvaluation:
    config: RuleStrategyConfig
    config_hash: str
    validation_rows: tuple[ValidationRow, ...]
    passed: bool
    best_so_far_summary: dict[str, Any]
    resume_cursor: int
    min_excess_return: float
    average_excess_return: float
    average_drawdown_advantage: float
    average_win_rate: float
    failing_cases: tuple[str, ...]


@dataclass
class RuleOptimizerState:
    budget_name: str
    stage: str
    resume_cursor: int
    coarse_candidates: list[RuleStrategyConfig]
    local_candidates: list[RuleStrategyConfig]
    top_results: list[OptimizerEvaluation]
    best_result: OptimizerEvaluation | None
    best_so_far_summary: dict[str, Any]
    evaluated_candidates: int
    total_candidates: int
    completed: bool
    last_message: str
    last_updated: str


PROFILE_PRESETS = {
    "Aggressive": SignalPreset(name="Aggressive", strong_threshold=66, watch_threshold=52, cooldown_bars=3),
    "Balanced": SignalPreset(name="Balanced", strong_threshold=70, watch_threshold=56, cooldown_bars=4),
    "Conservative": SignalPreset(name="Conservative", strong_threshold=74, watch_threshold=60, cooldown_bars=6),
}
DEFAULT_SIGNAL_PRESET = PROFILE_PRESETS["Balanced"]
DEFAULT_RULE_CONFIG = RuleStrategyConfig(
    fast_ema=8,
    slow_ema=50,
    bottom_required=2,
    buy_confirmation_required=2,
    sell_confirmation_required=2,
    buy_score_threshold=58,
    sell_score_threshold=56,
    require_macd_for_buy=False,
    require_volume_for_buy=False,
    allow_reclaim_trigger=True,
)


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
            background: linear-gradient(135deg, #0f172a 0%, #14532d 42%, #f5efe6 100%);
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
            color: rgba(248, 250, 252, 0.92);
        }
        .metric-card {
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 16px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            padding: 0.9rem 1rem;
            min-height: 118px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        }
        .metric-card .eyebrow {
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #475569;
            margin-bottom: 0.35rem;
        }
        .metric-card .value {
            font-size: 1.55rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.1;
            margin-bottom: 0.35rem;
        }
        .metric-card .caption {
            font-size: 0.84rem;
            color: #475569;
        }
        .section-label {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #475569;
            margin: 1.15rem 0 0.55rem;
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
        percent = min(100, int(round((self.completed_steps / self.total_steps) * 100.0)))
        self.progress.progress(percent, text=f"Failed ({self.completed_steps}/{self.total_steps})")
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
    if row is None:
        return default
    if hasattr(row, "get"):
        try:
            return row.get(key, default)
        except TypeError:
            pass
    return getattr(row, key, default)


class TupleRowAccessor:
    def __init__(self, row: Any) -> None:
        self.row = row

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self.row, key, default)


def is_us_batch_ticker(ticker: str) -> bool:
    normalized = ticker.strip().upper()
    return not bool(KOREAN_TICKER_PATTERN.fullmatch(normalized))


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.{digits}f}"


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:,.2f}%"


def _format_score(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.0f}"


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


def format_snapshot_label(snapshot: TickerSnapshot) -> str:
    resolved = snapshot.resolved_symbol.strip().upper()
    if resolved == snapshot.ticker.upper():
        return snapshot.ticker
    return f"{snapshot.ticker} ({resolved})"


def ema_column(span: int) -> str:
    return f"EMA{int(span)}"


def config_to_display(config: RuleStrategyConfig) -> dict[str, Any]:
    return {
        "Fast EMA": config.fast_ema,
        "Slow EMA": config.slow_ema,
        "Bottom Votes": config.bottom_required,
        "Buy Votes": config.buy_confirmation_required,
        "Sell Votes": config.sell_confirmation_required,
        "Buy Threshold": config.buy_score_threshold,
        "Sell Threshold": config.sell_score_threshold,
        "Need MACD": config.require_macd_for_buy,
        "Need Volume": config.require_volume_for_buy,
        "Allow Reclaim": config.allow_reclaim_trigger,
        "Config Hash": config.config_hash(),
    }


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


def build_controls() -> dict[str, Any]:
    st.sidebar.subheader("Analysis Controls")
    watchlist_raw = st.sidebar.text_area(
        "Watchlist tickers",
        value=", ".join(DEFAULT_WATCHLIST),
        help="Any ticker can be explored. Hold-beat completion is guaranteed only for SPY, QQQ, and 069500.KS.",
        height=96,
    )
    history_years = st.sidebar.slider("History lookback (years)", min_value=1, max_value=10, value=DEFAULT_HISTORY_YEARS, step=1)
    search_budget = st.sidebar.selectbox("Search Budget", options=list(SEARCH_BUDGET_SPECS), index=list(SEARCH_BUDGET_SPECS).index(DEFAULT_SEARCH_BUDGET))
    optimize = st.sidebar.button("Optimize", width="stretch")
    resume_search = st.sidebar.button("Resume Search", width="stretch")
    reset_search = st.sidebar.button("Reset Search", width="stretch")
    refresh = st.sidebar.button("Refresh cached data", width="stretch")
    st.sidebar.caption("Core guarantee matrix: SPY / QQQ / 069500.KS x 1Y / 3Y / 5Y / Full.")

    watchlist = dedupe_preserve_order(parse_tickers(watchlist_raw))
    if not watchlist:
        watchlist = list(DEFAULT_WATCHLIST)
    return {
        "watchlist": watchlist,
        "history_years": int(history_years),
        "search_budget": str(search_budget),
        "optimize": bool(optimize),
        "resume_search": bool(resume_search),
        "reset_search": bool(reset_search),
        "refresh": bool(refresh),
    }


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
    if sell_cd == 13:
        return "Sell countdown 13"
    if buy_cd == 13:
        return "Buy countdown 13"
    if sell_setup >= 7:
        return f"Sell setup {sell_setup}"
    if buy_setup >= 7:
        return f"Buy setup {buy_setup}"
    return "No active TD setup"


def build_td_label_series(frame: pd.DataFrame) -> pd.Series:
    labels = pd.Series("No active TD setup", index=frame.index, dtype=object)
    sell_setup = pd.to_numeric(frame.get("SellSetup"), errors="coerce").fillna(0).astype(int)
    buy_setup = pd.to_numeric(frame.get("BuySetup"), errors="coerce").fillna(0).astype(int)
    sell_countdown = pd.to_numeric(frame.get("SellCountdown"), errors="coerce").fillna(0).astype(int)
    buy_countdown = pd.to_numeric(frame.get("BuyCountdown"), errors="coerce").fillna(0).astype(int)

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
    work["ReferenceAsset"] = reference_symbol in CORE_TICKER_SET
    for span in EMA_ALL_SPANS:
        work[ema_column(span)] = work["Close"].ewm(span=span, adjust=False, min_periods=span).mean()
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


def classify_market_regime(row: Any) -> str:
    close_value = _coerce_float(row_get(row, "Close"))
    ema50 = _coerce_float(row_get(row, "EMA50"))
    ema200 = _coerce_float(row_get(row, "EMA200"))
    trend_delta = _coerce_float(row_get(row, "TrendDelta5"))
    adx = _coerce_float(row_get(row, "ADX"))
    plus_di = _coerce_float(row_get(row, "PlusDI"))
    minus_di = _coerce_float(row_get(row, "MinusDI"))

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


def _sum_bool_columns(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame.loc[:, columns].fillna(False).astype(bool).sum(axis=1).astype(int)


def _build_buy_reason_parts(row: Any) -> list[str]:
    reasons: list[str] = []
    if _coerce_bool(row_get(row, "BottomRSILow")):
        reasons.append("RSI low")
    if _coerce_bool(row_get(row, "BottomSTLLow")):
        reasons.append("RobustSTL low")
    if _coerce_bool(row_get(row, "BottomFearLow")):
        reasons.append("Fear & Greed low")
    if _coerce_bool(row_get(row, "BuyConfirmTD")):
        reasons.append("TD buy exhaustion")
    if _coerce_bool(row_get(row, "BuyConfirmMACD")):
        reasons.append("MACD turn")
    if _coerce_bool(row_get(row, "BuyConfirmAccumulation")):
        reasons.append("Accumulation flow")
    if _coerce_bool(row_get(row, "BuyConfirmVolumeReset")):
        reasons.append("Volume reset")
    if _coerce_bool(row_get(row, "BuyMACrossUp")):
        reasons.append("EMA bull cross")
    elif _coerce_bool(row_get(row, "BuyMAReclaim")):
        reasons.append("EMA reclaim")
    return reasons[:5]


def _build_sell_reason_parts(row: Any) -> list[str]:
    reasons: list[str] = []
    if _coerce_bool(row_get(row, "SellConfirmRSI")):
        reasons.append("RSI hot")
    if _coerce_bool(row_get(row, "SellConfirmFear")):
        reasons.append("Greed high")
    if _coerce_bool(row_get(row, "SellConfirmTD")):
        reasons.append("TD sell exhaustion")
    if _coerce_bool(row_get(row, "SellConfirmMACD")):
        reasons.append("MACD bear")
    if _coerce_bool(row_get(row, "SellConfirmDistribution")):
        reasons.append("Distribution flow")
    if _coerce_bool(row_get(row, "SellConfirmATR")):
        reasons.append("ATR hot")
    if _coerce_bool(row_get(row, "SellMACrossDown")):
        reasons.append("EMA bear cross")
    elif _coerce_bool(row_get(row, "SellMALoseFast")):
        reasons.append("Lost fast EMA")
    if _coerce_bool(row_get(row, "StopExitConfirmed")):
        reasons.append("Stop confirmed")
    return reasons[:5]


def _compose_signal_note(row: Any, state: str) -> str:
    buy_score = int(row_get(row, "BuyScore", 0) or 0)
    sell_score = int(row_get(row, "SellScore", 0) or 0)
    regime = str(row_get(row, "Regime", "n/a"))
    td_label = str(row_get(row, "TDLabel", "Unavailable"))
    cycle_score = _coerce_float(row_get(row, "CycleScore"))
    fear_greed = _coerce_float(row_get(row, "FearGreed"))
    reason_parts = _build_buy_reason_parts(row) if state in {"Strong Buy", "Buy", "Weak Buy"} else _build_sell_reason_parts(row)
    if state == "Hold / Neutral":
        reason_parts = (_build_buy_reason_parts(row)[:1] + _build_sell_reason_parts(row)[:1])[:2]
    cycle_text = f"Cycle {_format_float(cycle_score, 1)}" if cycle_score is not None else "Cycle n/a"
    fear_text = f"Fear & Greed {_format_float(fear_greed, 0)}" if fear_greed is not None else "Fear & Greed n/a"
    reason_text = ", ".join(reason_parts) if reason_parts else "No dominant edge"
    return (
        f"{state} | {regime} | B{buy_score}/S{sell_score} | "
        f"Setup {int(_coerce_bool(row_get(row, 'BuySetupActive')))}/{int(_coerce_bool(row_get(row, 'SellSetupActive')))} | "
        f"Trigger {int(_coerce_bool(row_get(row, 'BuyTrigger')))}/{int(_coerce_bool(row_get(row, 'SellTrigger')))} | "
        f"{cycle_text} | {td_label} | {fear_text} | {reason_text}"
    )


def build_state_frame(
    indicator_frame: pd.DataFrame,
    preset: SignalPreset,
    config: RuleStrategyConfig | None = None,
    *,
    include_notes: bool = True,
) -> pd.DataFrame:
    strategy = config or DEFAULT_RULE_CONFIG
    frame = indicator_frame.dropna(subset=["Close"]).copy()
    if frame.empty:
        return frame

    fast_col = ema_column(strategy.fast_ema)
    slow_col = ema_column(strategy.slow_ema)
    if fast_col not in frame.columns or slow_col not in frame.columns:
        return pd.DataFrame(index=frame.index)

    regime_series = frame.apply(classify_market_regime, axis=1)
    fast_ema = pd.to_numeric(frame[fast_col], errors="coerce")
    slow_ema = pd.to_numeric(frame[slow_col], errors="coerce")
    fast_slope = fast_ema - fast_ema.shift(3)
    slow_slope = slow_ema - slow_ema.shift(5)
    reference_threshold = np.where(frame["ReferenceAsset"].fillna(False), 40.0, 50.0)
    rs_support = pd.to_numeric(frame.get("RSPercentile"), errors="coerce").ge(reference_threshold) | pd.to_numeric(frame.get("RSMomentum63"), errors="coerce").ge(-0.02)
    accumulation = frame["VolumeClimaxUp"].fillna(False) | (
        pd.to_numeric(frame["CMF20"], errors="coerce").ge(0.03) & pd.to_numeric(frame["OBVSlope10"], errors="coerce").gt(0.0)
    )
    distribution = frame["VolumeClimaxDown"].fillna(False) | (
        pd.to_numeric(frame["CMF20"], errors="coerce").le(-0.03) & pd.to_numeric(frame["OBVSlope10"], errors="coerce").lt(0.0)
    )
    volume_reset = frame["VolumeDryUp"].fillna(False) | (
        pd.to_numeric(frame["VolumeRatio20"], errors="coerce").ge(1.05) & frame["Close"].ge(frame["Open"])
    )
    bottom_rsi = pd.to_numeric(frame["RSI"], errors="coerce").le(BUY_CUTOFFS["rsi_reset"])
    bottom_stl = pd.to_numeric(frame["CycleScore"], errors="coerce").le(25.0)
    bottom_fear = pd.to_numeric(frame["FearGreed"], errors="coerce").le(BUY_CUTOFFS["fear_reset"])
    buy_td = pd.to_numeric(frame["BuySetup"], errors="coerce").ge(8) | pd.to_numeric(frame["BuyCountdown"], errors="coerce").eq(13)
    sell_td = pd.to_numeric(frame["SellSetup"], errors="coerce").ge(8) | pd.to_numeric(frame["SellCountdown"], errors="coerce").eq(13)
    buy_macd = frame["MACDBullCross"].fillna(False) | (
        pd.to_numeric(frame["MACDHistDelta"], errors="coerce").gt(0.0) & pd.to_numeric(frame["MACDHist"], errors="coerce").le(0.12)
    )
    sell_macd = frame["MACDBearCross"].fillna(False) | (
        pd.to_numeric(frame["MACDHistDelta"], errors="coerce").lt(0.0) & pd.to_numeric(frame["MACDHist"], errors="coerce").ge(-0.12)
    )
    buy_ma_cross = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
    buy_ma_reclaim = frame["Close"].gt(fast_ema) & frame["Close"].shift(1).le(fast_ema.shift(1)) & fast_slope.gt(0)
    sell_ma_cross = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
    sell_ma_reclaim = frame["Close"].lt(fast_ema) & frame["Close"].shift(1).ge(fast_ema.shift(1)) & slow_slope.lt(0)

    frame["Regime"] = regime_series
    frame["ConfigHash"] = strategy.config_hash()
    frame["ConfigFastEMA"] = fast_ema
    frame["ConfigSlowEMA"] = slow_ema
    frame["ConfigFastSpan"] = strategy.fast_ema
    frame["ConfigSlowSpan"] = strategy.slow_ema
    frame["BottomRSILow"] = bottom_rsi.fillna(False)
    frame["BottomSTLLow"] = bottom_stl.fillna(False)
    frame["BottomFearLow"] = bottom_fear.fillna(False)
    frame["BottomVotes"] = _sum_bool_columns(frame, ["BottomRSILow", "BottomSTLLow", "BottomFearLow"])
    frame["BuyConfirmMFI"] = pd.to_numeric(frame["MFI"], errors="coerce").le(BUY_CUTOFFS["mfi_reset"]).fillna(False)
    frame["BuyConfirmATR"] = pd.to_numeric(frame["ATRStretch"], errors="coerce").le(-0.85).fillna(False)
    frame["BuyConfirmBB"] = pd.to_numeric(frame["BBPos20"], errors="coerce").le(BUY_CUTOFFS["bb_low"]).fillna(False)
    frame["BuyConfirmTD"] = buy_td.fillna(False)
    frame["BuyConfirmMACD"] = buy_macd.fillna(False)
    frame["BuyConfirmAccumulation"] = accumulation.fillna(False)
    frame["BuyConfirmRelative"] = rs_support.fillna(False)
    frame["BuyConfirmVolumeReset"] = volume_reset.fillna(False)
    frame["BuyMACrossUp"] = buy_ma_cross.fillna(False)
    frame["BuyMAReclaim"] = buy_ma_reclaim.fillna(False)
    frame["BuyConfirmationVotes"] = _sum_bool_columns(
        frame,
        [
            "BuyConfirmMFI",
            "BuyConfirmATR",
            "BuyConfirmBB",
            "BuyConfirmTD",
            "BuyConfirmMACD",
            "BuyConfirmAccumulation",
            "BuyConfirmRelative",
            "BuyConfirmVolumeReset",
        ],
    )
    frame["SellConfirmRSI"] = pd.to_numeric(frame["RSI"], errors="coerce").ge(SELL_CUTOFFS["rsi_hot"]).fillna(False)
    frame["SellConfirmFear"] = pd.to_numeric(frame["FearGreed"], errors="coerce").ge(SELL_CUTOFFS["greed_hot"]).fillna(False)
    frame["SellConfirmTD"] = sell_td.fillna(False)
    frame["SellConfirmMACD"] = sell_macd.fillna(False)
    frame["SellConfirmDistribution"] = distribution.fillna(False)
    frame["SellConfirmATR"] = pd.to_numeric(frame["ATRStretch"], errors="coerce").ge(0.95).fillna(False)
    frame["SellConfirmBB"] = pd.to_numeric(frame["BBPos20"], errors="coerce").ge(SELL_CUTOFFS["bb_high"]).fillna(False)
    frame["SellMACrossDown"] = sell_ma_cross.fillna(False)
    frame["SellMALoseFast"] = sell_ma_reclaim.fillna(False)
    frame["SellConfirmationVotes"] = _sum_bool_columns(
        frame,
        [
            "SellConfirmRSI",
            "SellConfirmFear",
            "SellConfirmTD",
            "SellConfirmMACD",
            "SellConfirmDistribution",
            "SellConfirmATR",
            "SellConfirmBB",
        ],
    )
    frame["SellCapitulationBlock"] = (
        frame["BottomVotes"].ge(2)
        & (pd.to_numeric(frame["CycleScore"], errors="coerce").le(30.0) | pd.to_numeric(frame["RSI"], errors="coerce").le(35.0))
    ).fillna(False)

    buy_score = (
        np.where(frame["BottomRSILow"], 16.0, 0.0)
        + np.where(pd.to_numeric(frame["RSI"], errors="coerce").le(BUY_CUTOFFS["rsi_extreme"]), 6.0, 0.0)
        + np.where(frame["BottomSTLLow"], 18.0, 0.0)
        + np.where(pd.to_numeric(frame["CycleScore"], errors="coerce").le(12.0), 6.0, 0.0)
        + np.where(frame["BottomFearLow"], 14.0, 0.0)
        + np.where(pd.to_numeric(frame["FearGreed"], errors="coerce").le(BUY_CUTOFFS["fear_extreme"]), 6.0, 0.0)
        + np.where(frame["BuyConfirmMFI"], 7.0, 0.0)
        + np.where(pd.to_numeric(frame["MFI"], errors="coerce").le(BUY_CUTOFFS["mfi_extreme"]), 3.0, 0.0)
        + np.where(frame["BuyConfirmATR"], 8.0, 0.0)
        + np.where(frame["BuyConfirmBB"], 7.0, 0.0)
        + np.where(frame["BuyConfirmTD"], 9.0, 0.0)
        + np.where(frame["BuyConfirmMACD"], 10.0, 0.0)
        + np.where(frame["BuyConfirmAccumulation"], 8.0, 0.0)
        + np.where(frame["BuyConfirmRelative"], 6.0, 0.0)
        + np.where(frame["BuyConfirmVolumeReset"], 6.0, 0.0)
        + np.where(frame["BuyMACrossUp"], 10.0, 0.0)
        + np.where(frame["BuyMAReclaim"], 6.0, 0.0)
        + np.where(frame["BottomVotes"].ge(2), 8.0, 0.0)
        + np.where(frame["BottomVotes"].eq(3), 6.0, 0.0)
        + np.where(regime_series.isin({"Bull Trend", "Bull Pullback"}), 5.0, 0.0)
    )
    sell_score = (
        np.where(frame["SellConfirmRSI"], 10.0, 0.0)
        + np.where(pd.to_numeric(frame["RSI"], errors="coerce").ge(SELL_CUTOFFS["rsi_extreme"]), 5.0, 0.0)
        + np.where(frame["SellConfirmFear"], 8.0, 0.0)
        + np.where(pd.to_numeric(frame["FearGreed"], errors="coerce").ge(SELL_CUTOFFS["greed_extreme"]), 4.0, 0.0)
        + np.where(frame["SellConfirmTD"], 8.0, 0.0)
        + np.where(frame["SellConfirmMACD"], 9.0, 0.0)
        + np.where(frame["SellConfirmDistribution"], 9.0, 0.0)
        + np.where(frame["SellConfirmATR"], 8.0, 0.0)
        + np.where(frame["SellConfirmBB"], 7.0, 0.0)
        + np.where(frame["SellMACrossDown"], 10.0, 0.0)
        + np.where(frame["SellMALoseFast"], 6.0, 0.0)
        + np.where(regime_series.isin({"Bear Trend", "Bear Rally"}), 5.0, 0.0)
    )
    frame["BuyScore"] = np.clip(np.rint(buy_score), 0, 100).astype(int)
    frame["SellScore"] = np.clip(np.rint(sell_score), 0, 100).astype(int)

    buy_setup_active = (
        frame["BottomVotes"].ge(strategy.bottom_required)
        & frame["BuyConfirmationVotes"].ge(max(1, strategy.buy_confirmation_required - 1))
        & pd.Series(frame["BuyScore"], index=frame.index).ge(max(strategy.buy_score_threshold - 8, 40))
    )
    sell_setup_active = (
        ~frame["SellCapitulationBlock"]
        & frame["SellConfirmationVotes"].ge(max(1, strategy.sell_confirmation_required - 1))
        & pd.Series(frame["SellScore"], index=frame.index).ge(max(strategy.sell_score_threshold - 8, 40))
    )
    buy_trigger_candidate = (
        buy_setup_active
        & frame["BuyConfirmationVotes"].ge(strategy.buy_confirmation_required)
        & (frame["BuyMACrossUp"] | (frame["BuyMAReclaim"] if strategy.allow_reclaim_trigger else False))
    )
    if strategy.require_macd_for_buy:
        buy_trigger_candidate &= frame["BuyConfirmMACD"]
    if strategy.require_volume_for_buy:
        buy_trigger_candidate &= (frame["BuyConfirmAccumulation"] | frame["BuyConfirmVolumeReset"])
    sell_trigger_candidate = (
        sell_setup_active
        & frame["SellConfirmationVotes"].ge(strategy.sell_confirmation_required)
        & (frame["SellMACrossDown"] | frame["SellMALoseFast"])
    )

    state_rows = frame.copy()
    state_rows["BuySetupActive"] = buy_setup_active.fillna(False)
    state_rows["SellSetupActive"] = sell_setup_active.fillna(False)
    state_rows["BuyTrigger"] = False
    state_rows["SellTrigger"] = False
    state_rows["State"] = "Hold / Neutral"
    state_rows["HoldLockActive"] = False
    state_rows["ReentryLockActive"] = False
    state_rows["StopExitConfirmed"] = False
    state_rows["StopEmergencyExit"] = False
    state_rows["StopGapATR"] = np.nan
    state_rows["BuyReasonText"] = ""
    state_rows["SellReasonText"] = ""
    state_rows["SignalNote"] = ""

    in_virtual_position = False
    entry_index = -10_000
    last_exit_index = -10_000
    min_hold_bars = preset_min_hold_bars(preset)
    reentry_lock_bars = preset_reentry_lock_bars(preset)

    for index, row_tuple in enumerate(state_rows.itertuples(index=False, name="StateRow")):
        row = TupleRowAccessor(row_tuple)
        regime = str(row.get("Regime", "Range / Transition"))
        bear_regime = _is_bear_regime(regime)
        hold_lock_active = in_virtual_position and (index - entry_index) < min_hold_bars
        reentry_lock_active = (not in_virtual_position) and (index - last_exit_index) < reentry_lock_bars
        stop_exit_confirmed, stop_emergency_exit, stop_gap_atr = evaluate_stop_exit(
            _coerce_float(row.get("Close")),
            _coerce_float(row.get("ChandelierStop")),
            _coerce_float(row.get("ATR")),
            sell_score=int(row.get("SellScore", 0) or 0),
            buy_score=int(row.get("BuyScore", 0) or 0),
            bearish_context=bear_regime,
            sell_context=bool(_coerce_bool(row.get("SellSetupActive")) or _coerce_bool(row.get("SellTriggerPrice"))),
            hold_lock_active=hold_lock_active,
            macro_risk_off=_coerce_bool(row.get("TIPSRiskOff")),
        )

        buy_trigger = bool(row.get("BuySetupActive")) and bool(buy_trigger_candidate.iloc[index]) and not reentry_lock_active and not stop_exit_confirmed
        sell_trigger = (
            in_virtual_position
            and (
                stop_emergency_exit
                or stop_exit_confirmed
                or (bool(row.get("SellSetupActive")) and bool(sell_trigger_candidate.iloc[index]) and not bool(row.get("SellCapitulationBlock")))
            )
        )
        if hold_lock_active and not stop_emergency_exit:
            sell_trigger = bool(stop_exit_confirmed and _coerce_bool(row.get("StopCloseBreach")))
        if stop_emergency_exit:
            buy_trigger = False
            sell_trigger = True

        buy_reason_parts = _build_buy_reason_parts(row)
        sell_reason_parts = _build_sell_reason_parts(row)

        if stop_emergency_exit:
            state = "Strong Sell"
        elif sell_trigger and int(row.get("SellScore", 0) or 0) >= max(strategy.sell_score_threshold + 10, int(row.get("BuyScore", 0) or 0) + 8):
            state = "Strong Sell"
        elif sell_trigger:
            state = "Sell"
        elif buy_trigger and int(row.get("BuyScore", 0) or 0) >= max(strategy.buy_score_threshold + 10, preset.strong_threshold):
            state = "Strong Buy"
        elif buy_trigger:
            state = "Buy"
        elif bool(row.get("BuySetupActive")) and int(row.get("BuyScore", 0) or 0) >= max(strategy.buy_score_threshold - 10, preset.watch_threshold - 4):
            state = "Weak Buy"
        elif bool(row.get("SellSetupActive")) and not bool(row.get("SellCapitulationBlock")):
            state = "Weak Sell"
        else:
            state = "Hold / Neutral"

        state_rows.iat[index, state_rows.columns.get_loc("BuyTrigger")] = buy_trigger
        state_rows.iat[index, state_rows.columns.get_loc("SellTrigger")] = sell_trigger
        state_rows.iat[index, state_rows.columns.get_loc("State")] = state
        state_rows.iat[index, state_rows.columns.get_loc("HoldLockActive")] = hold_lock_active
        state_rows.iat[index, state_rows.columns.get_loc("ReentryLockActive")] = reentry_lock_active
        state_rows.iat[index, state_rows.columns.get_loc("StopExitConfirmed")] = stop_exit_confirmed
        state_rows.iat[index, state_rows.columns.get_loc("StopEmergencyExit")] = stop_emergency_exit
        state_rows.iat[index, state_rows.columns.get_loc("StopGapATR")] = stop_gap_atr if stop_gap_atr is not None else np.nan
        if include_notes:
            state_rows.iat[index, state_rows.columns.get_loc("BuyReasonText")] = " | ".join(buy_reason_parts)
            state_rows.iat[index, state_rows.columns.get_loc("SellReasonText")] = " | ".join(sell_reason_parts)
            state_rows.iat[index, state_rows.columns.get_loc("SignalNote")] = _compose_signal_note(TupleRowAccessor(state_rows.iloc[index]), state)

        if not in_virtual_position and buy_trigger:
            in_virtual_position = True
            entry_index = index
        elif in_virtual_position and sell_trigger:
            in_virtual_position = False
            last_exit_index = index
            entry_index = -10_000

    setup_window = max(3, min(6, preset.cooldown_bars + 1))
    state_rows["BuySetupWindow"] = state_rows["BuySetupActive"].rolling(setup_window, min_periods=1).max().fillna(0.0).astype(bool)
    state_rows["SellSetupWindow"] = state_rows["SellSetupActive"].rolling(setup_window, min_periods=1).max().fillna(0.0).astype(bool)
    return state_rows


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


def compute_max_drawdown(equity_curve: pd.Series) -> float | None:
    series = pd.to_numeric(equity_curve, errors="coerce").dropna()
    if series.empty:
        return None
    running_max = series.cummax()
    drawdown = series / running_max - 1.0
    return abs(float(drawdown.min()))


def compute_cagr(equity_curve: pd.Series) -> float | None:
    series = pd.to_numeric(equity_curve, errors="coerce").dropna()
    if len(series) < 2:
        return None
    start_value = float(series.iloc[0])
    end_value = float(series.iloc[-1])
    if start_value <= 0 or end_value <= 0:
        return None
    start_date = pd.Timestamp(series.index[0])
    end_date = pd.Timestamp(series.index[-1])
    years = max((end_date - start_date).days / 365.25, 1 / 365.25)
    return float((end_value / start_value) ** (1.0 / years) - 1.0)


def build_buy_hold_metrics(state_frame: pd.DataFrame, transaction_cost: float = TRANSACTION_COST_PER_SIDE) -> tuple[dict[str, float | None], pd.Series]:
    if state_frame.empty or "Close" not in state_frame.columns:
        return {
            "buy_hold_return": None,
            "buy_hold_start_price": None,
            "buy_hold_end_price": None,
            "buy_hold_max_drawdown": None,
            "buy_hold_cagr": None,
            "buy_hold_cost_drag": None,
        }, pd.Series(dtype=float)
    close = pd.to_numeric(state_frame["Close"], errors="coerce").dropna()
    if close.empty:
        return {
            "buy_hold_return": None,
            "buy_hold_start_price": None,
            "buy_hold_end_price": None,
            "buy_hold_max_drawdown": None,
            "buy_hold_cagr": None,
            "buy_hold_cost_drag": None,
        }, pd.Series(dtype=float)
    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])
    if start_price <= 0:
        return {
            "buy_hold_return": None,
            "buy_hold_start_price": start_price,
            "buy_hold_end_price": end_price,
            "buy_hold_max_drawdown": None,
            "buy_hold_cagr": None,
            "buy_hold_cost_drag": None,
        }, pd.Series(dtype=float)

    shares = (1.0 - transaction_cost) / start_price
    equity_curve = shares * close
    equity_curve.iloc[-1] = equity_curve.iloc[-1] * (1.0 - transaction_cost)
    raw_return = float(end_price / start_price - 1.0)
    net_return = float(equity_curve.iloc[-1] - 1.0)
    return {
        "buy_hold_return": net_return,
        "buy_hold_start_price": start_price,
        "buy_hold_end_price": end_price,
        "buy_hold_max_drawdown": compute_max_drawdown(equity_curve),
        "buy_hold_cagr": compute_cagr(equity_curve),
        "buy_hold_cost_drag": raw_return - net_return,
    }, equity_curve


def build_trade_replay_from_state_frame(
    ticker: str,
    state_frame: pd.DataFrame,
    preset: SignalPreset,
    *,
    transaction_cost: float = TRANSACTION_COST_PER_SIDE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | None]]:
    if state_frame.empty:
        empty_summary: dict[str, float | int | None] = {
            "closed_trades": 0,
            "trade_count": 0,
            "win_rate": None,
            "avg_closed_return": None,
            "median_closed_return": None,
            "avg_hold_days": None,
            "open_trade_return": None,
            "strategy_total_return": None,
            "max_drawdown": None,
            "cagr": None,
            "cost_drag": None,
            "buy_hold_return": None,
            "buy_hold_start_price": None,
            "buy_hold_end_price": None,
            "buy_hold_max_drawdown": None,
            "buy_hold_cagr": None,
            "buy_hold_cost_drag": None,
            "excess_return": None,
            "relative_alpha": None,
        }
        return pd.DataFrame(), pd.DataFrame(), empty_summary

    events: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    cash = 1.0
    shares = 0.0
    cumulative_cost_paid = 0.0
    in_position = False
    partial_taken = False
    trade_id = 0
    entry_date: pd.Timestamp | None = None
    entry_price: float | None = None
    entry_equity: float | None = None
    last_buy_index = -10_000
    last_sell_index = -10_000
    last_partial_index = -10_000
    entry_index = -10_000
    min_hold_bars = preset_min_hold_bars(preset)
    partial_hold_bars = max(1, min_hold_bars - 1)
    iter_frame = state_frame.reset_index().rename(columns={"index": "Date"})

    for position, row in enumerate(iter_frame.itertuples(index=False)):
        timestamp = pd.Timestamp(row.Date)
        state = str(row.State)
        price = float(row.Close)
        buy_score = int(row.BuyScore)
        sell_score = int(row.SellScore)
        buy_trigger = _coerce_bool(getattr(row, "BuyTrigger", state in {"Buy", "Strong Buy"}))
        sell_trigger = _coerce_bool(getattr(row, "SellTrigger", state in {"Sell", "Strong Sell"}))
        sell_setup_active = _coerce_bool(getattr(row, "SellSetupActive", state in {"Weak Sell", "Sell", "Strong Sell"}))
        stop_emergency_exit = _coerce_bool(getattr(row, "StopEmergencyExit", False))
        stop_exit_confirmed = _coerce_bool(getattr(row, "StopExitConfirmed", False))
        note = str(getattr(row, "SignalNote", "") or "")
        if not note:
            note = _compose_signal_note(TupleRowAccessor(row), state)

        hold_bars_elapsed = position - entry_index if in_position else 10_000
        force_exit = bool(stop_emergency_exit)
        buy_ready = (not in_position) and buy_trigger and (position - last_buy_index) >= preset.cooldown_bars
        partial_ready = (
            in_position
            and (not partial_taken)
            and state == "Weak Sell"
            and sell_setup_active
            and not stop_exit_confirmed
            and (hold_bars_elapsed >= partial_hold_bars)
            and (position - last_partial_index) >= preset.cooldown_bars
        )
        sell_ready = (
            in_position
            and sell_trigger
            and ((hold_bars_elapsed >= min_hold_bars) or force_exit)
            and (position - last_sell_index) >= preset.cooldown_bars
        )

        if buy_ready and price > 0 and cash > 0:
            trade_id += 1
            trade_cash = cash
            buy_cost = trade_cash * transaction_cost
            cumulative_cost_paid += buy_cost
            shares = (trade_cash - buy_cost) / price
            cash = 0.0
            in_position = True
            partial_taken = False
            entry_date = timestamp
            entry_price = price
            entry_equity = trade_cash
            entry_index = position
            last_buy_index = position
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
                    "PositionAfter": 1.0,
                    "Reason": note,
                }
            )
        elif partial_ready and shares > 0:
            shares_to_sell = shares * 0.5
            gross_proceeds = shares_to_sell * price
            sell_cost = gross_proceeds * transaction_cost
            cumulative_cost_paid += sell_cost
            cash += gross_proceeds - sell_cost
            shares -= shares_to_sell
            partial_taken = True
            last_partial_index = position
            position_after = 0.0 if (cash + (shares * price)) <= 0 else float((shares * price) / (cash + (shares * price)))
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
                    "PositionAfter": position_after,
                    "Reason": note,
                }
            )
        elif sell_ready and shares > 0 and entry_equity is not None and entry_date is not None:
            gross_proceeds = shares * price
            sell_cost = gross_proceeds * transaction_cost
            cumulative_cost_paid += sell_cost
            cash += gross_proceeds - sell_cost
            shares = 0.0
            last_sell_index = position
            trade_return = None if entry_equity <= 0 else float(cash / entry_equity - 1.0)
            events.append(
                {
                    "Ticker": ticker,
                    "Trade": trade_id,
                    "Date": timestamp,
                    "Signal": "SELL",
                    "Price": price,
                    "State": state,
                    "BuyScore": buy_score,
                    "SellScore": sell_score,
                    "PositionAfter": 0.0,
                    "Reason": note,
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
                    "ExitPrice": price,
                    "Return": trade_return,
                    "HoldDays": int((timestamp - entry_date).days),
                    "PartialScaledOut": "Yes" if partial_taken else "No",
                }
            )
            in_position = False
            partial_taken = False
            entry_date = None
            entry_price = None
            entry_equity = None
            entry_index = -10_000

        equity_rows.append(
            {
                "Date": timestamp,
                "Equity": cash + (shares * price),
            }
        )

    if in_position and entry_equity is not None and entry_date is not None:
        last_timestamp = pd.Timestamp(state_frame.index[-1])
        last_price = float(state_frame["Close"].iloc[-1])
        mark_to_market = cash + (shares * last_price)
        trades.append(
            {
                "Ticker": ticker,
                "Trade": trade_id,
                "Status": "Open",
                "EntryDate": entry_date,
                "ExitDate": last_timestamp,
                "EntryPrice": entry_price,
                "ExitPrice": last_price,
                "Return": float(mark_to_market / entry_equity - 1.0),
                "HoldDays": int((last_timestamp - entry_date).days),
                "PartialScaledOut": "Yes" if partial_taken else "No",
            }
        )

    event_frame = pd.DataFrame(events)
    trade_frame = pd.DataFrame(trades)
    equity_frame = pd.DataFrame(equity_rows)
    equity_curve = pd.Series(equity_frame["Equity"].to_numpy(dtype=float), index=pd.to_datetime(equity_frame["Date"])) if not equity_frame.empty else pd.Series(dtype=float)
    closed = trade_frame.loc[trade_frame["Status"] == "Closed"].copy() if not trade_frame.empty else pd.DataFrame()
    open_trade = trade_frame.loc[trade_frame["Status"] == "Open"].copy() if not trade_frame.empty else pd.DataFrame()

    summary: dict[str, float | int | None] = {
        "closed_trades": int(len(closed)),
        "trade_count": int(len(trade_frame)),
        "win_rate": float(closed["Return"].gt(0.0).mean()) if not closed.empty else None,
        "avg_closed_return": float(closed["Return"].mean()) if not closed.empty else None,
        "median_closed_return": float(closed["Return"].median()) if not closed.empty else None,
        "avg_hold_days": float(closed["HoldDays"].mean()) if not closed.empty else None,
        "open_trade_return": float(open_trade["Return"].iloc[-1]) if not open_trade.empty else None,
        "strategy_total_return": float(equity_curve.iloc[-1] - 1.0) if not equity_curve.empty else None,
        "max_drawdown": compute_max_drawdown(equity_curve),
        "cagr": compute_cagr(equity_curve),
        "cost_drag": cumulative_cost_paid,
    }
    buy_hold_metrics, _ = build_buy_hold_metrics(state_frame, transaction_cost=transaction_cost)
    summary.update(buy_hold_metrics)
    strategy_total_return = summary.get("strategy_total_return")
    buy_hold_return = summary.get("buy_hold_return")
    summary["excess_return"] = (
        None if strategy_total_return is None or buy_hold_return is None else float(strategy_total_return - buy_hold_return)
    )
    summary["relative_alpha"] = (
        None
        if strategy_total_return is None or buy_hold_return is None
        else float((1.0 + strategy_total_return) / (1.0 + buy_hold_return) - 1.0)
    )
    return event_frame, trade_frame, summary


def build_chart_signal_frame(state_frame: pd.DataFrame, preset: SignalPreset) -> pd.DataFrame:
    if state_frame.empty:
        return pd.DataFrame(columns=["Date", "Signal", "Price", "State", "BuyScore", "SellScore"])

    markers: list[dict[str, Any]] = []
    last_buy_index = -10_000
    last_sell_index = -10_000
    iter_frame = state_frame.reset_index().rename(columns={"index": "Date"})

    for position, row in enumerate(iter_frame.itertuples(index=False)):
        price = float(row.Close)
        buy_trigger = _coerce_bool(getattr(row, "BuyTrigger", str(row.State) in {"Buy", "Strong Buy"}))
        sell_trigger = _coerce_bool(getattr(row, "SellTrigger", str(row.State) in {"Sell", "Strong Sell"}))
        if buy_trigger and (position - last_buy_index) >= preset.cooldown_bars:
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
        if sell_trigger and (position - last_sell_index) >= preset.cooldown_bars:
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
    plot_frame = state_frame.loc[:, ["Close", "BuyScore", "SellScore", "State", "ConfigFastEMA", "ConfigSlowEMA", "ConfigFastSpan", "ConfigSlowSpan"]].copy().reset_index().rename(columns={"index": "Date"})
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

    fast_span = int(plot_frame["ConfigFastSpan"].dropna().iloc[-1]) if not plot_frame["ConfigFastSpan"].dropna().empty else DEFAULT_RULE_CONFIG.fast_ema
    slow_span = int(plot_frame["ConfigSlowSpan"].dropna().iloc[-1]) if not plot_frame["ConfigSlowSpan"].dropna().empty else DEFAULT_RULE_CONFIG.slow_ema
    fig.add_trace(go.Scatter(x=plot_frame["Date"], y=plot_frame["ConfigFastEMA"], mode="lines", name=f"EMA{fast_span}", line={"color": "#16a34a", "width": 1.5}))
    fig.add_trace(go.Scatter(x=plot_frame["Date"], y=plot_frame["ConfigSlowEMA"], mode="lines", name=f"EMA{slow_span}", line={"color": "#ea580c", "width": 1.5}))

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
    fig.update_layout(height=640, margin={"l": 10, "r": 10, "t": 70, "b": 20}, legend={"orientation": "h"})
    return fig


def analyze_symbol_state(
    ticker: str,
    payload: dict[str, Any],
    macro_data: dict[str, Any],
    preset: SignalPreset,
    config: RuleStrategyConfig,
) -> dict[str, Any]:
    price_frame = payload.get("frame", pd.DataFrame())
    indicators = compute_indicators(
        price_frame,
        macro_data["score_series"],
        macro_data.get("benchmark_close"),
        macro_data.get("tips_13612w_momentum"),
        payload.get("resolved_symbol"),
    )
    state_frame = build_state_frame(indicators, preset, config, include_notes=True)
    snapshot = build_snapshot(ticker, payload.get("resolved_symbol", ticker), state_frame, payload.get("warning")) if not state_frame.empty else None
    return {
        "indicators": indicators,
        "state_frame": state_frame,
        "snapshot": snapshot,
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_core_indicator_inputs(end_iso: str) -> dict[str, Any]:
    macro_data = load_macro_fear_greed_cached(end_iso)
    payloads = load_price_payloads(list(CORE_TICKERS), OPTIMIZER_START_ISO, end_iso)
    indicator_frames: dict[str, pd.DataFrame] = {}
    for ticker in CORE_TICKERS:
        payload = payloads.get(ticker, {"frame": pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS)), "resolved_symbol": ticker, "warning": f"{ticker} payload unavailable"})
        indicator_frames[ticker] = compute_indicators(
            payload.get("frame", pd.DataFrame()),
            macro_data["score_series"],
            macro_data.get("benchmark_close"),
            macro_data.get("tips_13612w_momentum"),
            payload.get("resolved_symbol"),
        )
    return {
        "macro_data": macro_data,
        "payloads": payloads,
        "indicator_frames": indicator_frames,
    }


def _validation_summary(rows: list[ValidationRow]) -> tuple[float, float, float, float, int]:
    if not rows:
        return -10_000.0, -10_000.0, -10_000.0, 0.0, 99
    excess_values = [row.excess_return if row.excess_return is not None else -10_000.0 for row in rows]
    drawdown_adv = [
        ((row.buy_hold_max_drawdown or 0.0) - (row.max_drawdown or 0.0))
        for row in rows
    ]
    win_rates = [row.win_rate if row.win_rate is not None else 0.0 for row in rows]
    failing = sum(not row.pass_case for row in rows)
    return (
        float(min(excess_values)),
        float(np.mean(excess_values)),
        float(np.mean(drawdown_adv)),
        float(np.mean(win_rates)),
        failing,
    )


def _assess_validation_case(window_label: str, summary: dict[str, float | int | None]) -> tuple[bool, str | None]:
    excess_return = summary.get("excess_return")
    if excess_return is None or float(excess_return) < 0.0:
        return False, "Underperformed buy and hold"

    win_rate = summary.get("win_rate")
    if win_rate is None or float(win_rate) < 0.35:
        return False, "Win rate below 35%"

    min_trades = 2 if window_label == "1Y" else 4
    closed_trades = int(summary.get("closed_trades") or 0)
    if closed_trades < min_trades:
        return False, f"Closed trades below {min_trades}"

    max_drawdown = summary.get("max_drawdown")
    buy_hold_max_drawdown = summary.get("buy_hold_max_drawdown")
    if max_drawdown is not None and buy_hold_max_drawdown is not None:
        if float(max_drawdown) > float(buy_hold_max_drawdown) + 0.10:
            return False, "Drawdown guardrail breached"

    return True, None


def validate_rule_strategy_config(
    core_indicator_frames: dict[str, pd.DataFrame],
    config: RuleStrategyConfig,
    preset: SignalPreset,
    *,
    resume_cursor: int = 0,
) -> OptimizerEvaluation:
    validation_rows: list[ValidationRow] = []

    for ticker, indicator_frame in core_indicator_frames.items():
        if indicator_frame.empty:
            continue
        full_state_frame = build_state_frame(indicator_frame, preset, config, include_notes=False)
        if full_state_frame.empty:
            continue
        max_date = pd.Timestamp(full_state_frame.index.max())
        for window_label, days in VALIDATION_WINDOWS:
            if days is None:
                sliced = full_state_frame.copy()
            else:
                start_ts = max_date - pd.Timedelta(days=days)
                sliced = full_state_frame.loc[full_state_frame.index >= start_ts].copy()
            _, _, summary = build_trade_replay_from_state_frame(
                ticker,
                sliced,
                preset,
                transaction_cost=TRANSACTION_COST_PER_SIDE,
            )
            pass_case, fail_reason = _assess_validation_case(window_label, summary)
            validation_rows.append(
                ValidationRow(
                    ticker=ticker,
                    window=window_label,
                    start_date=pd.Timestamp(sliced.index.min()).date().isoformat() if not sliced.empty else "n/a",
                    end_date=pd.Timestamp(sliced.index.max()).date().isoformat() if not sliced.empty else "n/a",
                    strategy_total_return=_coerce_float(summary.get("strategy_total_return")),
                    buy_hold_return=_coerce_float(summary.get("buy_hold_return")),
                    excess_return=_coerce_float(summary.get("excess_return")),
                    max_drawdown=_coerce_float(summary.get("max_drawdown")),
                    buy_hold_max_drawdown=_coerce_float(summary.get("buy_hold_max_drawdown")),
                    cagr=_coerce_float(summary.get("cagr")),
                    buy_hold_cagr=_coerce_float(summary.get("buy_hold_cagr")),
                    win_rate=_coerce_float(summary.get("win_rate")),
                    closed_trades=int(summary.get("closed_trades") or 0),
                    pass_case=pass_case,
                    fail_reason=fail_reason,
                )
            )

    min_excess, average_excess, drawdown_advantage, average_win_rate, failing_count = _validation_summary(validation_rows)
    failing_cases = tuple(
        f"{row.ticker} {row.window}: {row.fail_reason}"
        for row in validation_rows
        if not row.pass_case and row.fail_reason
    )
    summary = {
        "min_excess_return": min_excess,
        "average_excess_return": average_excess,
        "average_drawdown_advantage": drawdown_advantage,
        "average_win_rate": average_win_rate,
        "passed_cases": len(validation_rows) - failing_count,
        "total_cases": len(validation_rows),
        "failing_cases": failing_count,
    }
    return OptimizerEvaluation(
        config=config,
        config_hash=config.config_hash(),
        validation_rows=tuple(validation_rows),
        passed=all(row.pass_case for row in validation_rows) if validation_rows else False,
        best_so_far_summary=summary,
        resume_cursor=resume_cursor,
        min_excess_return=min_excess,
        average_excess_return=average_excess,
        average_drawdown_advantage=drawdown_advantage,
        average_win_rate=average_win_rate,
        failing_cases=failing_cases,
    )


def _evaluation_sort_key(result: OptimizerEvaluation) -> tuple[float, float, float, float, int]:
    fail_count = sum(not row.pass_case for row in result.validation_rows)
    return (
        result.min_excess_return,
        result.average_excess_return,
        result.average_drawdown_advantage,
        result.average_win_rate,
        -fail_count,
    )


def generate_coarse_candidates(budget_name: str) -> list[RuleStrategyConfig]:
    spec = SEARCH_BUDGET_SPECS[budget_name]
    candidates: list[RuleStrategyConfig] = []
    for fast in spec["coarse_fast"]:
        for slow in spec["coarse_slow"]:
            if fast >= slow:
                continue
            for bottom_required in spec["bottom_votes"]:
                for buy_votes in spec["buy_votes"]:
                    for sell_votes in spec["sell_votes"]:
                        for buy_threshold in spec["buy_thresholds"]:
                            for sell_threshold in spec["sell_thresholds"]:
                                for need_macd, need_volume, allow_reclaim in spec["toggle_options"]:
                                    candidates.append(
                                        RuleStrategyConfig(
                                            fast_ema=int(fast),
                                            slow_ema=int(slow),
                                            bottom_required=int(bottom_required),
                                            buy_confirmation_required=int(buy_votes),
                                            sell_confirmation_required=int(sell_votes),
                                            buy_score_threshold=int(buy_threshold),
                                            sell_score_threshold=int(sell_threshold),
                                            require_macd_for_buy=bool(need_macd),
                                            require_volume_for_buy=bool(need_volume),
                                            allow_reclaim_trigger=bool(allow_reclaim),
                                        )
                                    )
    unique: dict[str, RuleStrategyConfig] = {candidate.config_hash(): candidate for candidate in candidates}
    return list(unique.values())


def _neighboring_values(options: tuple[int, ...], value: int) -> list[int]:
    index = options.index(value)
    candidates = {value}
    if index > 0:
        candidates.add(int(options[index - 1]))
    if index + 1 < len(options):
        candidates.add(int(options[index + 1]))
    return sorted(candidates)


def generate_local_candidates(top_results: list[OptimizerEvaluation], budget_name: str) -> list[RuleStrategyConfig]:
    if not top_results:
        return []
    spec = SEARCH_BUDGET_SPECS[budget_name]
    top_n = int(spec["local_top_n"])
    configs = [result.config for result in sorted(top_results, key=_evaluation_sort_key, reverse=True)[:top_n]]
    buy_threshold_neighbors = lambda value: sorted({max(44, value - 6), max(44, value - 3), value, min(78, value + 3), min(78, value + 6)})
    sell_threshold_neighbors = lambda value: sorted({max(42, value - 6), max(42, value - 3), value, min(76, value + 3), min(76, value + 6)})

    local_candidates: dict[str, RuleStrategyConfig] = {}
    for config in configs:
        for fast in _neighboring_values(EMA_FAST_OPTIONS, config.fast_ema):
            for slow in _neighboring_values(EMA_SLOW_OPTIONS, config.slow_ema):
                if fast >= slow:
                    continue
                for bottom_required in sorted({2, 3, config.bottom_required}):
                    for buy_votes in sorted({1, 2, 3, config.buy_confirmation_required}):
                        for sell_votes in sorted({1, 2, 3, config.sell_confirmation_required}):
                            for buy_threshold in buy_threshold_neighbors(config.buy_score_threshold):
                                for sell_threshold in sell_threshold_neighbors(config.sell_score_threshold):
                                    for need_macd in {config.require_macd_for_buy, not config.require_macd_for_buy}:
                                        for need_volume in {config.require_volume_for_buy, not config.require_volume_for_buy}:
                                            for allow_reclaim in {config.allow_reclaim_trigger, not config.allow_reclaim_trigger}:
                                                candidate = RuleStrategyConfig(
                                                    fast_ema=int(fast),
                                                    slow_ema=int(slow),
                                                    bottom_required=int(bottom_required),
                                                    buy_confirmation_required=int(min(3, max(1, buy_votes))),
                                                    sell_confirmation_required=int(min(3, max(1, sell_votes))),
                                                    buy_score_threshold=int(buy_threshold),
                                                    sell_score_threshold=int(sell_threshold),
                                                    require_macd_for_buy=bool(need_macd),
                                                    require_volume_for_buy=bool(need_volume),
                                                    allow_reclaim_trigger=bool(allow_reclaim),
                                                )
                                                local_candidates[candidate.config_hash()] = candidate
    values = list(local_candidates.values())
    local_limit = spec.get("local_limit")
    if local_limit is not None:
        return values[: int(local_limit)]
    return values


def create_optimizer_state(budget_name: str) -> RuleOptimizerState:
    coarse_candidates = generate_coarse_candidates(budget_name)
    return RuleOptimizerState(
        budget_name=budget_name,
        stage="coarse",
        resume_cursor=0,
        coarse_candidates=coarse_candidates,
        local_candidates=[],
        top_results=[],
        best_result=None,
        best_so_far_summary={},
        evaluated_candidates=0,
        total_candidates=len(coarse_candidates),
        completed=False,
        last_message=f"Initialized {len(coarse_candidates)} coarse candidates.",
        last_updated=time.strftime("%H:%M:%S"),
    )


def _update_optimizer_best(state: RuleOptimizerState, evaluation: OptimizerEvaluation) -> None:
    state.top_results.append(evaluation)
    state.top_results = sorted(state.top_results, key=_evaluation_sort_key, reverse=True)[:12]
    if state.best_result is None or _evaluation_sort_key(evaluation) > _evaluation_sort_key(state.best_result):
        state.best_result = evaluation
        state.best_so_far_summary = evaluation.best_so_far_summary


def run_optimizer_batch(
    state: RuleOptimizerState,
    core_indicator_frames: dict[str, pd.DataFrame],
    preset: SignalPreset,
) -> RuleOptimizerState:
    if state.completed:
        return state

    batch_size = int(SEARCH_BUDGET_SPECS[state.budget_name]["batch_size"])
    if state.stage == "coarse":
        candidates = state.coarse_candidates
    elif state.stage == "local":
        candidates = state.local_candidates
    else:
        candidates = []

    if state.stage in {"coarse", "local"}:
        end_cursor = min(len(candidates), state.resume_cursor + batch_size)
        for offset, config in enumerate(candidates[state.resume_cursor:end_cursor], start=1):
            evaluation = validate_rule_strategy_config(
                core_indicator_frames,
                config,
                preset,
                resume_cursor=state.resume_cursor + offset,
            )
            state.evaluated_candidates += 1
            _update_optimizer_best(state, evaluation)
        state.resume_cursor = end_cursor
        state.last_updated = time.strftime("%H:%M:%S")
        if state.resume_cursor >= len(candidates):
            if state.stage == "coarse":
                state.local_candidates = generate_local_candidates(state.top_results, state.budget_name)
                state.total_candidates += len(state.local_candidates)
                state.stage = "local" if state.local_candidates else "retest"
                state.resume_cursor = 0
                state.last_message = f"Coarse search complete. Generated {len(state.local_candidates)} local candidates."
            else:
                state.stage = "retest"
                state.resume_cursor = 0
                state.last_message = "Local refinement complete. Running robustness retest on best configuration."
        else:
            state.last_message = f"Processed {state.resume_cursor}/{len(candidates)} {state.stage} candidates."
        return state

    if state.stage == "retest":
        if state.best_result is not None:
            retested = validate_rule_strategy_config(
                core_indicator_frames,
                state.best_result.config,
                preset,
                resume_cursor=state.resume_cursor,
            )
            state.best_result = retested
            state.best_so_far_summary = retested.best_so_far_summary
            state.last_message = "Robustness retest completed for current best configuration."
        else:
            state.last_message = "No valid optimizer result was available for retest."
        state.completed = True
        state.last_updated = time.strftime("%H:%M:%S")
        return state

    return state


def render_header(macro_data: dict[str, Any], snapshots: list[TickerSnapshot], optimizer_state: RuleOptimizerState | None) -> None:
    latest_as_of = max(snapshot.as_of for snapshot in snapshots).date().isoformat() if snapshots else "n/a"
    optimizer_status = "UNSOLVED / keep searching"
    if optimizer_state and optimizer_state.best_result and optimizer_state.best_result.passed:
        optimizer_status = "SOLVED"
    warning_text = ""
    if macro_data.get("warning"):
        warning_text = " Macro model is using a neutral fallback for unavailable fields."
    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_TITLE}</h1>
            <p>Latest completed daily bar: {latest_as_of} | Macro Fear & Greed: {macro_data['latest_label']} ({macro_data['latest_score']:.0f}) | Core status: {optimizer_status}.{warning_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_metrics(
    watchlist_frame: pd.DataFrame,
    macro_data: dict[str, Any],
    optimizer_state: RuleOptimizerState | None,
    active_config: RuleStrategyConfig,
) -> None:
    strong_buy = int((watchlist_frame["state"] == "Strong Buy").sum()) if not watchlist_frame.empty else 0
    buy_count = int((watchlist_frame["state"] == "Buy").sum()) if not watchlist_frame.empty else 0
    strong_sell = int((watchlist_frame["state"] == "Strong Sell").sum()) if not watchlist_frame.empty else 0
    sell_count = int((watchlist_frame["state"] == "Sell").sum()) if not watchlist_frame.empty else 0
    solved = bool(optimizer_state and optimizer_state.best_result and optimizer_state.best_result.passed)
    passed_cases = 0
    total_cases = 12
    if optimizer_state and optimizer_state.best_result:
        passed_cases = int(sum(row.pass_case for row in optimizer_state.best_result.validation_rows))
        total_cases = int(len(optimizer_state.best_result.validation_rows))
    cards = st.columns(4)
    with cards[0]:
        render_metric_card("Macro Fear & Greed", f"{macro_data['latest_score']:.0f}", macro_data["latest_label"])
    with cards[1]:
        render_metric_card("Core Status", "SOLVED" if solved else "UNSOLVED", f"{passed_cases}/{total_cases} validation cases passed")
    with cards[2]:
        render_metric_card("Watchlist Bias", str(strong_buy + buy_count), f"Buy {strong_buy + buy_count} / Sell {strong_sell + sell_count}")
    with cards[3]:
        render_metric_card("Active EMA Pair", f"{active_config.fast_ema}/{active_config.slow_ema}", f"Config {active_config.config_hash()}")


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
    display["scope"] = display["ticker"].map(lambda value: "Guaranteed core" if str(value).upper() in CORE_TICKER_SET else "Exploratory only")
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
        "Scope",
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


def render_optimizer_status(optimizer_state: RuleOptimizerState | None) -> None:
    st.markdown('<div class="section-label">Optimizer Status</div>', unsafe_allow_html=True)
    if optimizer_state is None:
        st.info("Optimizer has not run yet. Use `Optimize` or `Resume Search` from the sidebar.")
        return
    best_result = optimizer_state.best_result
    status = "SOLVED" if best_result and best_result.passed else "UNSOLVED / keep searching"
    min_excess = _format_pct(best_result.min_excess_return if best_result else None)
    avg_excess = _format_pct(best_result.average_excess_return if best_result else None)
    cards = st.columns(4)
    with cards[0]:
        render_metric_card("Status", status, f"Stage {optimizer_state.stage} | Budget {optimizer_state.budget_name}")
    with cards[1]:
        render_metric_card("Evaluated", str(optimizer_state.evaluated_candidates), f"Candidate cursor {optimizer_state.resume_cursor}/{optimizer_state.total_candidates}")
    with cards[2]:
        render_metric_card("Best Min Excess", min_excess, "Primary optimization objective")
    with cards[3]:
        render_metric_card("Best Avg Excess", avg_excess, f"Updated {optimizer_state.last_updated}")
    st.caption(optimizer_state.last_message)


def render_best_configuration(best_result: OptimizerEvaluation | None) -> None:
    st.markdown('<div class="section-label">Best Configuration</div>', unsafe_allow_html=True)
    if best_result is None:
        st.info("No configuration has been evaluated yet.")
        return
    display = pd.DataFrame(
        [
            {"Field": key, "Value": str(value)}
            for key, value in config_to_display(best_result.config).items()
        ]
        + [
            {"Field": "Passed All Cases", "Value": str(best_result.passed)},
            {"Field": "Min Excess Return", "Value": _format_pct(best_result.min_excess_return)},
            {"Field": "Average Excess Return", "Value": _format_pct(best_result.average_excess_return)},
            {"Field": "Average Drawdown Advantage", "Value": _format_pct(best_result.average_drawdown_advantage)},
            {"Field": "Average Win Rate", "Value": _format_pct(best_result.average_win_rate)},
        ]
    )
    st.dataframe(display, width="stretch", hide_index=True)


def render_validation_matrix(best_result: OptimizerEvaluation | None) -> None:
    st.markdown('<div class="section-label">Validation Matrix</div>', unsafe_allow_html=True)
    if best_result is None:
        st.info("Validation matrix will appear after the first optimizer batch.")
        return
    rows = [asdict(row) for row in best_result.validation_rows]
    frame = pd.DataFrame(rows)
    if frame.empty:
        st.info("No validation rows were produced.")
        return
    frame["strategy_total_return"] = frame["strategy_total_return"].map(_format_pct)
    frame["buy_hold_return"] = frame["buy_hold_return"].map(_format_pct)
    frame["excess_return"] = frame["excess_return"].map(_format_pct)
    frame["max_drawdown"] = frame["max_drawdown"].map(_format_pct)
    frame["buy_hold_max_drawdown"] = frame["buy_hold_max_drawdown"].map(_format_pct)
    frame["cagr"] = frame["cagr"].map(_format_pct)
    frame["buy_hold_cagr"] = frame["buy_hold_cagr"].map(_format_pct)
    frame["win_rate"] = frame["win_rate"].map(_format_pct)
    frame = frame.rename(
        columns={
            "ticker": "Ticker",
            "window": "Window",
            "start_date": "Start",
            "end_date": "End",
            "strategy_total_return": "Strategy Total",
            "buy_hold_return": "Buy & Hold",
            "excess_return": "Excess",
            "max_drawdown": "Strategy MDD",
            "buy_hold_max_drawdown": "Hold MDD",
            "cagr": "Strategy CAGR",
            "buy_hold_cagr": "Hold CAGR",
            "win_rate": "Win Rate",
            "closed_trades": "Closed Trades",
            "pass_case": "Pass",
            "fail_reason": "Fail Reason",
        }
    )
    st.dataframe(frame, width="stretch", hide_index=True)


def render_failing_cases(best_result: OptimizerEvaluation | None) -> None:
    st.markdown('<div class="section-label">Failing Cases</div>', unsafe_allow_html=True)
    if best_result is None:
        st.info("No failing-case diagnostics yet.")
        return
    failing = [item for item in best_result.failing_cases if item]
    if not failing:
        st.success("All validation cases passed.")
        return
    for item in failing:
        st.warning(item)


def render_signal_review(event_frame: pd.DataFrame, trade_frame: pd.DataFrame, summary: dict[str, float | int | None]) -> None:
    st.markdown('<div class="section-label">Historical Signal Replay</div>', unsafe_allow_html=True)
    st.caption("Signals below replay the same rule set through history with fixed transaction costs.")

    avg_hold_days = "n/a" if summary["avg_hold_days"] is None else f"{summary['avg_hold_days']:.0f}d"
    cards_top = st.columns(6)
    with cards_top[0]:
        render_metric_card("Closed Trades", str(summary["closed_trades"]), "Completed round trips")
    with cards_top[1]:
        render_metric_card("Win Rate", _format_pct(summary["win_rate"]), "Closed trades only")
    with cards_top[2]:
        render_metric_card("Avg Closed Return", _format_pct(summary["avg_closed_return"]), "Closed trades only")
    with cards_top[3]:
        render_metric_card("Median Return", _format_pct(summary["median_closed_return"]), "Closed trades only")
    with cards_top[4]:
        render_metric_card("Avg Hold", avg_hold_days, "Closed trades only")
    with cards_top[5]:
        render_metric_card("Cost Drag", _format_pct(summary.get("cost_drag")), "Cumulative strategy costs")

    cards_bottom = st.columns(5)
    with cards_bottom[0]:
        render_metric_card("Strategy Total", _format_pct(summary.get("strategy_total_return")), "Net of costs")
    with cards_bottom[1]:
        render_metric_card("Buy & Hold", _format_pct(summary.get("buy_hold_return")), "Net of costs")
    with cards_bottom[2]:
        render_metric_card("Alpha vs Hold", _format_pct(summary.get("excess_return")), "Strategy minus hold")
    with cards_bottom[3]:
        render_metric_card("Max Drawdown", _format_pct(summary.get("max_drawdown")), "Strategy equity curve")
    with cards_bottom[4]:
        render_metric_card("CAGR", _format_pct(summary.get("cagr")), "Strategy equity curve")

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


def render_ticker_panel(
    selected_ticker: str,
    snapshot: TickerSnapshot,
    state_frame: pd.DataFrame,
    chart_signal_frame: pd.DataFrame,
    event_frame: pd.DataFrame,
    trade_frame: pd.DataFrame,
    replay_summary: dict[str, float | int | None],
    active_config: RuleStrategyConfig,
) -> None:
    st.markdown('<div class="section-label">Ticker Drill-down</div>', unsafe_allow_html=True)
    latest = state_frame.dropna(subset=["Close"]).iloc[-1]
    display_label = format_snapshot_label(snapshot)
    if selected_ticker.upper() not in CORE_TICKER_SET:
        st.info(f"{selected_ticker} is analyzed with the same rule engine, but hold-beat completion is guaranteed only for {', '.join(CORE_TICKERS)}.")
    left, right = st.columns([1.65, 1.0])
    with left:
        st.plotly_chart(build_price_context_figure(display_label, state_frame, chart_signal_frame, DEFAULT_SIGNAL_PRESET), width="stretch", config={"displaylogo": False})
    with right:
        score_cards_top = st.columns(2)
        with score_cards_top[0]:
            render_metric_card("Current State", snapshot.state, f"Resolved {snapshot.resolved_symbol}")
        with score_cards_top[1]:
            render_metric_card("Score Spread", str(snapshot.buy_score - snapshot.sell_score), "BuyScore minus SellScore")

        score_cards_bottom = st.columns(2)
        with score_cards_bottom[0]:
            render_metric_card("BuyScore", str(snapshot.buy_score), "Bottom-fishing + confirmation")
        with score_cards_bottom[1]:
            render_metric_card("SellScore", str(snapshot.sell_score), "MA exit + confirmation")

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
                {"Field": "EMA Pair", "Value": f"{active_config.fast_ema}/{active_config.slow_ema}"},
                {"Field": "Config Hash", "Value": active_config.config_hash()},
                {"Field": "Scope", "Value": "Guaranteed core" if selected_ticker.upper() in CORE_TICKER_SET else "Exploratory only"},
                {"Field": "As Of", "Value": snapshot.as_of.date().isoformat()},
            ]
        )
        st.dataframe(detail, width="stretch", hide_index=True)

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


def build_test_indicator_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    close = pd.Series(np.linspace(100, 120, len(index)), index=index)
    frame = synthetic_ohlcv_from_close(close)
    macro = pd.Series(50.0, index=index, name="FearGreed")
    indicators = compute_indicators(frame, macro, close * 0.99, macro / 100.0, "SPY")
    indicators["FearGreed"] = 50.0
    return indicators


def run_benchmark() -> dict[str, Any]:
    rows = 5_000
    rng = np.random.default_rng(RANDOM_STATE)
    index = pd.date_range("2010-01-04", periods=rows, freq="B")
    close = pd.Series(300 + rng.normal(0.04, 1.9, rows).cumsum(), index=index)
    frame = synthetic_ohlcv_from_close(close)
    macro = pd.Series(50 + rng.normal(0, 10, rows).cumsum() / 25.0, index=index, name="FearGreed")
    benchmark_close = close * 0.95

    started_at = time.perf_counter()
    indicators = compute_indicators(frame, macro, benchmark_close, macro / 100.0, "QQQ")
    indicators_seconds = time.perf_counter() - started_at

    started_at = time.perf_counter()
    state_frame = build_state_frame(indicators, DEFAULT_SIGNAL_PRESET, DEFAULT_RULE_CONFIG, include_notes=False)
    state_seconds = time.perf_counter() - started_at

    started_at = time.perf_counter()
    _, _, replay_summary = build_trade_replay_from_state_frame("QQQ", state_frame, DEFAULT_SIGNAL_PRESET)
    replay_seconds = time.perf_counter() - started_at

    download_symbols = ["QQQ", "SPY", "AAPL", "MSFT"]
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

    return {
        "rows": rows,
        "compute_indicators_seconds": round(indicators_seconds, 4),
        "build_state_frame_seconds": round(state_seconds, 4),
        "replay_seconds": round(replay_seconds, 4),
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
    print(f"replay_seconds={benchmark['replay_seconds']}")
    print(f"sequential_download_seconds={benchmark['sequential_download_seconds']}")
    print(f"batch_download_seconds={benchmark['batch_download_seconds']}")
    print(f"batch_speedup_x={benchmark['batch_speedup_x']}")
    print(f"replay_summary={benchmark['replay_summary']}")
    return 0


def run_rule_backtest(symbol: str, start_iso: str, end_iso: str) -> dict[str, Any]:
    symbol = symbol.strip().upper()
    preset = DEFAULT_SIGNAL_PRESET
    macro_data = load_macro_fear_greed_cached(end_iso)
    price_payloads = load_price_payloads([symbol], start_iso, end_iso)
    payload = price_payloads.get(symbol, {"frame": pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS)), "resolved_symbol": symbol, "warning": f"{symbol} payload unavailable"})
    analysis = analyze_symbol_state(symbol, payload, macro_data, preset, DEFAULT_RULE_CONFIG)
    state_frame = analysis["state_frame"]
    _, _, summary = build_trade_replay_from_state_frame(symbol, state_frame, preset)
    return {
        "symbol": symbol,
        "start": start_iso,
        "end": end_iso,
        "config": config_to_display(DEFAULT_RULE_CONFIG),
        "summary": summary,
    }


def run_rule_backtest_cli(symbol: str, start_iso: str, end_iso: str, json_output: bool = False) -> int:
    result = run_rule_backtest(symbol, start_iso, end_iso)
    if json_output:
        print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
        return 0
    print("RULE_BACKTEST")
    print(f"symbol={result['symbol']} start={result['start']} end={result['end']}")
    print(f"config={result['config']}")
    print(f"summary={result['summary']}")
    return 0


def run_optimize_report(end_iso: str, budget_name: str) -> dict[str, Any]:
    inputs = load_core_indicator_inputs(end_iso)
    state = create_optimizer_state(budget_name)
    while not state.completed:
        state = run_optimizer_batch(state, inputs["indicator_frames"], DEFAULT_SIGNAL_PRESET)
    best_result = state.best_result
    return {
        "budget": budget_name,
        "completed": state.completed,
        "evaluated_candidates": state.evaluated_candidates,
        "best_config": config_to_display(best_result.config) if best_result else None,
        "best_summary": best_result.best_so_far_summary if best_result else {},
        "passed": bool(best_result and best_result.passed),
        "validation_rows": [asdict(row) for row in best_result.validation_rows] if best_result else [],
        "failing_cases": list(best_result.failing_cases) if best_result else [],
    }


def run_optimize_report_cli(end_iso: str, budget_name: str, json_output: bool = False) -> int:
    result = run_optimize_report(end_iso, budget_name)
    if json_output:
        print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
        return 0
    print("OPTIMIZE_REPORT")
    print(f"budget={result['budget']} completed={result['completed']} evaluated_candidates={result['evaluated_candidates']}")
    print(f"best_config={result['best_config']}")
    print(f"best_summary={result['best_summary']}")
    print(f"passed={result['passed']}")
    print(f"failing_cases={result['failing_cases']}")
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
        idx = pd.date_range("2024-01-02", periods=40, freq="B")
        indicators = build_test_indicator_frame(idx)
        indicators.loc[idx[-2:], "RSI"] = [34.0, 28.0]
        indicators.loc[idx[-2:], "CycleScore"] = [18.0, 8.0]
        indicators.loc[idx[-2:], "FearGreed"] = [30.0, 18.0]
        indicators.loc[idx[-2:], "MFI"] = [28.0, 20.0]
        indicators.loc[idx[-2:], "ATRStretch"] = [-0.9, -1.4]
        indicators.loc[idx[-2:], "BBPos20"] = [0.18, 0.08]
        indicators.loc[idx[-2:], "BuySetup"] = [8, 9]
        indicators.loc[idx[-2:], "BuyCountdown"] = [0, 13]
        indicators.loc[idx[-2:], "MACDBullCross"] = [False, True]
        indicators.loc[idx[-2:], "MACDHist"] = [-0.2, -0.05]
        indicators.loc[idx[-2:], "MACDHistDelta"] = [0.05, 0.08]
        indicators.loc[idx[-2:], "CMF20"] = [0.01, 0.06]
        indicators.loc[idx[-2:], "OBVSlope10"] = [10.0, 20.0]
        indicators.loc[idx[-2:], "VolumeDryUp"] = [True, True]
        indicators.loc[idx[-2:], "RSPercentile"] = [55.0, 65.0]
        indicators.loc[idx[-2:], "RSMomentum63"] = [0.01, 0.03]
        indicators.loc[idx[-2:], "EMA8"] = [100.0, 103.0]
        indicators.loc[idx[-2:], "EMA50"] = [101.0, 102.0]
        indicators.loc[idx[-2:], "Close"] = [101.5, 104.0]
        indicators.loc[idx[-2:], "ChandelierStop"] = [95.0, 96.0]
        indicators.loc[idx[-2:], "StopCloseBreach"] = [False, False]
        indicators.loc[idx[-2:], "StopTouched"] = [False, False]
        state = build_state_frame(indicators, DEFAULT_SIGNAL_PRESET, DEFAULT_RULE_CONFIG, include_notes=False)
        latest = state.iloc[-1]
        passed = bool(int(latest["BottomVotes"]) == 3 and bool(latest["BuySetupActive"]) and bool(latest["BuyTrigger"]))
        record("bottom_setup_detection", passed, f"bottom_votes={int(latest['BottomVotes'])}, buy_trigger={bool(latest['BuyTrigger'])}")
    except Exception as exc:
        record("bottom_setup_detection", False, str(exc))

    try:
        idx = pd.date_range("2024-02-01", periods=30, freq="B")
        indicators = build_test_indicator_frame(idx)
        indicators.loc[idx[-5:-3], "RSI"] = [34.0, 29.0]
        indicators.loc[idx[-5:-3], "CycleScore"] = [16.0, 9.0]
        indicators.loc[idx[-5:-3], "FearGreed"] = [32.0, 20.0]
        indicators.loc[idx[-5:-3], "MFI"] = [30.0, 22.0]
        indicators.loc[idx[-5:-3], "ATRStretch"] = [-0.8, -1.1]
        indicators.loc[idx[-5:-3], "BBPos20"] = [0.15, 0.10]
        indicators.loc[idx[-5:-3], "BuySetup"] = [8, 9]
        indicators.loc[idx[-5:-3], "BuyCountdown"] = [0, 13]
        indicators.loc[idx[-5:-3], "MACDBullCross"] = [False, True]
        indicators.loc[idx[-5:-3], "MACDHist"] = [-0.18, -0.04]
        indicators.loc[idx[-5:-3], "MACDHistDelta"] = [0.04, 0.07]
        indicators.loc[idx[-5:-3], "CMF20"] = [0.02, 0.06]
        indicators.loc[idx[-5:-3], "OBVSlope10"] = [8.0, 18.0]
        indicators.loc[idx[-5:-3], "VolumeDryUp"] = [True, True]
        indicators.loc[idx[-5:-3], "RSPercentile"] = [54.0, 63.0]
        indicators.loc[idx[-5:-3], "RSMomentum63"] = [0.01, 0.03]
        indicators.loc[idx[-5:-3], "EMA8"] = [100.0, 103.0]
        indicators.loc[idx[-5:-3], "EMA50"] = [101.0, 102.0]
        indicators.loc[idx[-5:-3], "Close"] = [101.5, 104.0]
        indicators.loc[idx[-5:-3], "ChandelierStop"] = [95.0, 96.0]
        indicators.loc[idx[-5:-3], "StopCloseBreach"] = [False, False]
        indicators.loc[idx[-5:-3], "StopTouched"] = [False, False]
        indicators.loc[idx[-2:], "EMA8"] = [104.0, 101.0]
        indicators.loc[idx[-2:], "EMA50"] = [103.0, 102.0]
        indicators.loc[idx[-2:], "Close"] = [103.0, 99.0]
        indicators.loc[idx[-2:], "RSI"] = [68.0, 72.0]
        indicators.loc[idx[-2:], "FearGreed"] = [62.0, 68.0]
        indicators.loc[idx[-2:], "SellSetup"] = [8, 9]
        indicators.loc[idx[-2:], "SellCountdown"] = [0, 13]
        indicators.loc[idx[-2:], "MACDBearCross"] = [False, True]
        indicators.loc[idx[-2:], "MACDHist"] = [0.15, 0.02]
        indicators.loc[idx[-2:], "MACDHistDelta"] = [-0.04, -0.06]
        indicators.loc[idx[-2:], "CMF20"] = [-0.01, -0.06]
        indicators.loc[idx[-2:], "OBVSlope10"] = [-4.0, -12.0]
        indicators.loc[idx[-2:], "ATRStretch"] = [0.8, 1.1]
        indicators.loc[idx[-2:], "BBPos20"] = [0.8, 0.95]
        indicators.loc[idx[-2:], "ChandelierStop"] = [97.0, 96.0]
        indicators.loc[idx[-2:], "StopCloseBreach"] = [False, False]
        indicators.loc[idx[-2:], "StopTouched"] = [False, False]
        state = build_state_frame(indicators, DEFAULT_SIGNAL_PRESET, DEFAULT_RULE_CONFIG, include_notes=False)
        latest = state.iloc[-1]
        passed = bool(latest["SellTrigger"] and latest["State"] in {"Sell", "Strong Sell"})
        record("ema_bearish_trigger", passed, f"sell_trigger={bool(latest['SellTrigger'])}, state={latest['State']}")
    except Exception as exc:
        record("ema_bearish_trigger", False, str(exc))

    try:
        preset = DEFAULT_SIGNAL_PRESET
        replay_index = pd.date_range("2024-03-01", periods=6, freq="B")
        replay_state = pd.DataFrame(
            {
                "Close": [100.0, 101.0, 104.0, 110.0, 107.0, 99.0],
                "ChandelierStop": [90.0, 91.0, 92.0, 94.0, 95.0, 100.0],
                "State": ["Hold / Neutral", "Strong Buy", "Buy", "Weak Sell", "Hold / Neutral", "Strong Sell"],
                "BuyScore": [15, 80, 78, 34, 30, 20],
                "SellScore": [18, 20, 25, 62, 40, 82],
                "BuyTrigger": [False, True, False, False, False, False],
                "SellTrigger": [False, False, False, False, False, True],
                "SellSetupActive": [False, False, False, True, True, True],
                "SignalNote": ["n"] * 6,
            },
            index=replay_index,
        )
        _, _, net_summary = build_trade_replay_from_state_frame("TEST", replay_state, preset, transaction_cost=TRANSACTION_COST_PER_SIDE)
        _, _, gross_summary = build_trade_replay_from_state_frame("TEST", replay_state, preset, transaction_cost=0.0)
        passed = bool(
            net_summary["strategy_total_return"] is not None
            and gross_summary["strategy_total_return"] is not None
            and float(net_summary["strategy_total_return"]) < float(gross_summary["strategy_total_return"])
            and float(net_summary["cost_drag"] or 0.0) > 0.0
        )
        record(
            "transaction_cost_application",
            passed,
            f"net={net_summary['strategy_total_return']}, gross={gross_summary['strategy_total_return']}, cost_drag={net_summary['cost_drag']}",
        )
    except Exception as exc:
        record("transaction_cost_application", False, str(exc))

    try:
        failed_summary = {
            "excess_return": 0.02,
            "win_rate": 0.34,
            "closed_trades": 4,
            "max_drawdown": 0.25,
            "buy_hold_max_drawdown": 0.10,
        }
        passed_flag, reason = _assess_validation_case("3Y", failed_summary)
        passed = (not passed_flag) and bool(reason)
        record("optimizer_guardrail_rejection", passed, f"passed={passed_flag}, reason={reason}")
    except Exception as exc:
        record("optimizer_guardrail_rejection", False, str(exc))

    try:
        idx = pd.date_range("2024-01-02", periods=260, freq="B")
        core_frames: dict[str, pd.DataFrame] = {}
        for ticker in CORE_TICKERS:
            close = pd.Series(100 + np.linspace(0, 50, len(idx)) + (3 * np.sin(np.linspace(0, 12, len(idx)))), index=idx)
            frame = synthetic_ohlcv_from_close(close)
            macro = pd.Series(50.0, index=idx)
            indicators = compute_indicators(frame, macro, close * 0.99, macro / 100.0, ticker)
            core_frames[ticker] = indicators
        state = create_optimizer_state("Quick")
        first_cursor = state.resume_cursor
        updated = run_optimizer_batch(state, core_frames, DEFAULT_SIGNAL_PRESET)
        passed = updated.evaluated_candidates > 0 and (updated.resume_cursor != first_cursor or updated.stage != "coarse")
        record("resumable_search_state", passed, f"evaluated={updated.evaluated_candidates}, stage={updated.stage}, cursor={updated.resume_cursor}")
    except Exception as exc:
        record("resumable_search_state", False, str(exc))

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
    controls = build_controls()
    watchlist = controls["watchlist"]
    history_years = controls["history_years"]
    preset = DEFAULT_SIGNAL_PRESET
    should_run_optimizer = bool(controls["optimize"] or controls["resume_search"])
    progress_total_steps = 2 + len(watchlist) + 1 + (1 if should_run_optimizer else 0)
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

    if controls["refresh"]:
        st.cache_data.clear()
        st.sidebar.success("Cached data cleared. Rerun to fetch fresh prices.")
    if controls["reset_search"]:
        st.session_state.pop(OPTIMIZER_STATE_KEY, None)
        st.sidebar.success("Optimizer search state cleared.")

    end_date = date.today()
    start_date = end_date - timedelta(days=(history_years * 365) + 30)
    optimizer_state = st.session_state.get(OPTIMIZER_STATE_KEY)

    if should_run_optimizer:
        progress_tracker.update(
            "Running optimizer batch",
            f"Loading core matrix inputs through {end_date.isoformat()} with budget {controls['search_budget']}.",
            advance=0,
        )
        optimizer_started_at = time.perf_counter()
        core_inputs = load_core_indicator_inputs(end_date.isoformat())
        if (
            optimizer_state is None
            or controls["optimize"]
            or optimizer_state.budget_name != controls["search_budget"]
        ):
            optimizer_state = create_optimizer_state(controls["search_budget"])
        optimizer_state = run_optimizer_batch(optimizer_state, core_inputs["indicator_frames"], preset)
        st.session_state[OPTIMIZER_STATE_KEY] = optimizer_state
        add_timing("optimizer_batch", "core_matrix", optimizer_started_at, optimizer_state.evaluated_candidates)
        progress_tracker.update(
            "Optimizer batch complete",
            optimizer_state.last_message,
            advance=1,
        )

    active_config = optimizer_state.best_result.config if optimizer_state and optimizer_state.best_result else DEFAULT_RULE_CONFIG

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
        progress_tracker.update(
            "Downloading price history",
            f"Requesting {len(watchlist)} ticker payloads: {', '.join(watchlist[:8])}{' ...' if len(watchlist) > 8 else ''}",
            advance=0,
        )
        price_payloads = load_price_payloads(watchlist, start_date.isoformat(), end_date.isoformat())
        total_rows_downloaded = sum(len(payload.get("frame", pd.DataFrame())) for payload in price_payloads.values())
        add_timing("download_batch", "watchlist", download_started_at, total_rows_downloaded)
        progress_tracker.update(
            "Price download ready",
            f"Received {len(price_payloads)} payloads and {total_rows_downloaded} total rows.",
            advance=1,
        )
    except Exception as exc:
        progress_tracker.fail(f"Dashboard data could not be loaded: {exc}")
        st.error("Dashboard data could not be loaded.")
        st.info(str(exc))
        st.stop()

    state_frames: dict[str, pd.DataFrame] = {}
    snapshots: list[TickerSnapshot] = []
    warnings: list[str] = []
    if macro_data.get("warning"):
        warnings.append(str(macro_data["warning"]))

    for position, ticker in enumerate(watchlist, start=1):
        progress_tracker.update(
            "Computing ticker state",
            f"[{position}/{len(watchlist)}] Processing {ticker}: indicators, rules, and replay state.",
            advance=0,
        )
        compute_started_at = time.perf_counter()
        payload = price_payloads.get(ticker, {})
        warning = payload.get("warning")
        if warning:
            warnings.append(f"{ticker}: {warning}")
        analysis = analyze_symbol_state(ticker, payload, macro_data, preset, active_config)
        state_frame = analysis["state_frame"]
        snapshot = analysis["snapshot"]
        if state_frame.empty or snapshot is None:
            warnings.append(f"{ticker}: indicator stack could not build a usable state frame")
            progress_tracker.update(
                "Ticker computation failed",
                f"{ticker}: indicator stack could not build a usable state frame.",
                advance=1,
            )
            continue
        state_frames[ticker] = state_frame
        snapshots.append(snapshot)
        add_timing("compute", ticker, compute_started_at, len(state_frame))
        progress_tracker.update(
            "Ticker computed",
            f"{ticker}: {len(state_frame)} rows, latest state {snapshot.state}, config {active_config.config_hash()}.",
            advance=1,
        )

    watchlist_frame = build_watchlist_frame(snapshots)
    render_header(macro_data, snapshots, optimizer_state)
    render_top_metrics(watchlist_frame, macro_data, optimizer_state, active_config)
    render_warning_panel(warnings)
    render_optimizer_status(optimizer_state)
    render_best_configuration(optimizer_state.best_result if optimizer_state else None)
    render_validation_matrix(optimizer_state.best_result if optimizer_state else None)
    render_failing_cases(optimizer_state.best_result if optimizer_state else None)
    render_watchlist_summary(watchlist_frame)

    with st.expander("Macro Context", expanded=False):
        st.plotly_chart(build_macro_figure(macro_data, history_years), width="stretch", config={"displaylogo": False})

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
    progress_tracker.update(
        "Building replay summary",
        f"Preparing trade replay and summary cards for {selected_ticker}.",
        advance=0,
    )
    replay_started_at = time.perf_counter()
    event_frame, trade_frame, replay_summary = build_trade_replay_from_state_frame(selected_ticker, selected_state_frame, preset)
    add_timing("replay", selected_ticker, replay_started_at, len(event_frame))
    chart_signal_frame = event_frame.loc[event_frame["Signal"].isin(["BUY", "SELL"])].copy() if not event_frame.empty else build_chart_signal_frame(selected_state_frame, preset)
    progress_tracker.update(
        "Replay summary ready",
        f"{selected_ticker}: {len(event_frame)} events, {int(replay_summary.get('closed_trades') or 0)} closed trades.",
        advance=1,
    )
    add_timing("total", "app_run", run_started_at)
    performance_frame = build_performance_frame(timing_rows)
    progress_tracker.finish(
        f"Dashboard run completed in {time.perf_counter() - run_started_at:.2f}s. Selected ticker: {selected_ticker}."
    )

    render_ticker_panel(
        selected_ticker=selected_ticker,
        snapshot=selected_snapshot,
        state_frame=selected_state_frame,
        chart_signal_frame=chart_signal_frame,
        event_frame=event_frame,
        trade_frame=trade_frame,
        replay_summary=replay_summary,
        active_config=active_config,
    )
    render_diagnostics(diagnostics, performance_frame)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--self-test", action="store_true", help="Run built-in synthetic validation tests.")
    parser.add_argument("--benchmark", action="store_true", help="Run local performance benchmark.")
    parser.add_argument("--rule-backtest", action="store_true", help="Run cost-aware rule backtest for one symbol.")
    parser.add_argument("--optimize-report", action="store_true", help="Run optimizer search on the core validation matrix.")
    parser.add_argument("--symbol", default="QQQ", help="Ticker symbol for --rule-backtest.")
    parser.add_argument("--start", default="2016-01-01", help="Start date for --rule-backtest in YYYY-MM-DD format.")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date in YYYY-MM-DD format.")
    parser.add_argument("--budget", choices=list(SEARCH_BUDGET_SPECS), default=DEFAULT_SEARCH_BUDGET, help="Search budget for --optimize-report.")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit machine-readable JSON output for CLI commands.")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.self_test:
        raise SystemExit(run_self_tests_cli())
    if args.benchmark:
        raise SystemExit(run_benchmark_cli(json_output=args.json_output))
    if args.rule_backtest:
        raise SystemExit(run_rule_backtest_cli(args.symbol, args.start, args.end, json_output=args.json_output))
    if args.optimize_report:
        raise SystemExit(run_optimize_report_cli(args.end, args.budget, json_output=args.json_output))
    main()
