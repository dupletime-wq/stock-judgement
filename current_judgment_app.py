from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import io
import logging
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from statsmodels.tsa.seasonal import STL
import yfinance as yf


if "--self-test" in sys.argv:
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)


APP_TITLE = "Single-File Swing Trading Dashboard"
CACHE_TTL_SECONDS = 3600
MACRO_HISTORY_YEARS = 6
DEFAULT_WATCHLIST = ("SPY", "QQQ", "NVDA", "AAPL", "MSFT", "005930.KS")
REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
MACRO_TICKERS = ("SPY", "^VIX", "HYG", "IEF", "RSP", "XLY", "XLP", "UUP")
KOREAN_TICKER_PATTERN = re.compile(r"^(?P<code>\d{6})(?:\.(?P<suffix>KS|KQ))?$", re.IGNORECASE)
STATE_COLORS = {
    "Strong Buy": "#15803d",
    "Buy Watch": "#2563eb",
    "Hold / Neutral": "#475569",
    "Trim": "#d97706",
    "Strong Sell": "#dc2626",
}
STATE_PRIORITY = {
    "Strong Buy": 0,
    "Buy Watch": 1,
    "Hold / Neutral": 2,
    "Trim": 3,
    "Strong Sell": 4,
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


PROFILE_PRESETS = {
    "Aggressive": SignalPreset(name="Aggressive", strong_threshold=65, watch_threshold=50, cooldown_bars=3),
    "Balanced": SignalPreset(name="Balanced", strong_threshold=70, watch_threshold=55, cooldown_bars=5),
    "Conservative": SignalPreset(name="Conservative", strong_threshold=75, watch_threshold=60, cooldown_bars=8),
}


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


def build_controls() -> tuple[list[str], int, SignalPreset, bool]:
    st.sidebar.subheader("Dashboard Controls")
    watchlist_raw = st.sidebar.text_area(
        "Watchlist tickers",
        value=", ".join(DEFAULT_WATCHLIST),
        help="US tickers and Korean six-digit codes are both supported.",
        height=96,
    )
    history_years = st.sidebar.slider("History lookback (years)", min_value=1, max_value=5, value=3, step=1)
    profile_name = st.sidebar.selectbox("Signal profile", options=list(PROFILE_PRESETS), index=1)
    refresh = st.sidebar.button("Refresh cached data", width="stretch")
    st.sidebar.caption("Daily-bar swing dashboard built for Streamlit Community Cloud.")

    watchlist = parse_tickers(watchlist_raw) or list(DEFAULT_WATCHLIST)
    return watchlist, int(history_years), PROFILE_PRESETS[profile_name], refresh


def normalize_datetime_index(index: Any) -> pd.DatetimeIndex:
    normalized = pd.to_datetime(index)
    if isinstance(normalized, pd.DatetimeIndex):
        if normalized.tz is not None:
            normalized = normalized.tz_localize(None)
        return normalized
    return pd.DatetimeIndex(normalized)


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
        last_error = f"{candidate}: no usable OHLCV rows returned"

    return {
        "requested_ticker": requested_ticker,
        "resolved_symbol": resolve_yfinance_candidates(requested_ticker)[-1],
        "frame": pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS)),
        "warning": last_error or "No usable OHLCV data returned",
    }


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


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_macro_fear_greed_cached(end_iso: str) -> dict[str, Any]:
    end_ts = pd.Timestamp(end_iso)
    start_ts = end_ts - pd.DateOffset(years=MACRO_HISTORY_YEARS)
    try:
        raw = quiet_yfinance_download(
            list(MACRO_TICKERS),
            start=start_ts,
            end=end_ts + pd.Timedelta(days=1),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as exc:
        score_index = pd.date_range(end=end_ts, periods=260, freq="B")
        neutral_series = pd.Series(50.0, index=score_index, name="FearGreed")
        return {
            "score_series": neutral_series,
            "plot_df": pd.DataFrame({"Score": neutral_series}),
            "latest_score": 50.0,
            "latest_label": "Neutral",
            "latest_factors": pd.Series(dtype=float),
            "warning": f"Fear & Greed fallback to neutral: {exc}",
        }

    close_panel = _extract_close_panel(raw)
    if close_panel.dropna(how="all").empty:
        score_index = pd.date_range(end=end_ts, periods=260, freq="B")
        neutral_series = pd.Series(50.0, index=score_index, name="FearGreed")
        return {
            "score_series": neutral_series,
            "plot_df": pd.DataFrame({"Score": neutral_series}),
            "latest_score": 50.0,
            "latest_label": "Neutral",
            "latest_factors": pd.Series(dtype=float),
            "warning": "Fear & Greed fallback to neutral: macro basket returned no usable close data",
        }

    score_series, latest_factors, plot_df = compute_macro_fear_greed(close_panel)
    latest_score = float(score_series.dropna().iloc[-1]) if not score_series.dropna().empty else 50.0
    latest_label = classify_fear_greed(latest_score)
    return {
        "score_series": score_series,
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


def compute_indicators(price_df: pd.DataFrame, macro_score_series: pd.Series | None) -> pd.DataFrame:
    frame = normalize_ohlcv_frame(price_df)
    if frame.empty:
        return frame

    work = frame.copy()
    work["EMA20"] = work["Close"].ewm(span=20, adjust=False, min_periods=20).mean()
    work["EMA50"] = work["Close"].ewm(span=50, adjust=False, min_periods=50).mean()
    work["EMA200"] = work["Close"].ewm(span=200, adjust=False, min_periods=200).mean()
    work["RSI"] = calc_rsi(work["Close"], period=14)
    work["ATR"] = calc_atr(work, period=14)
    work["ChandelierStop"] = work["High"].rolling(22, min_periods=22).max() - (3.0 * work["ATR"])
    work["ATRStretch"] = (work["Close"] - work["EMA20"]) / work["ATR"].replace(0, np.nan)
    work["StopDistance"] = (work["Close"] / work["ChandelierStop"]) - 1.0
    work["EMA50Slope20"] = work["EMA50"] - work["EMA50"].shift(20)

    td = compute_td_sequential(work)
    work = work.join(td, how="left")
    stl = compute_stl_cycle(work)
    work = work.join(stl, how="left")
    work["TrendDelta5"] = work["Trend"] - work["Trend"].shift(5)
    work["TDLabel"] = build_td_label_series(work)

    if macro_score_series is None or macro_score_series.empty:
        work["FearGreed"] = 50.0
    else:
        aligned = pd.to_numeric(macro_score_series, errors="coerce").reindex(work.index).ffill().bfill()
        work["FearGreed"] = aligned.fillna(50.0)
    return work


def _coerce_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def score_buy_signal(row: pd.Series) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    cycle_score = _coerce_float(row.get("CycleScore"))
    buy_setup = int(row.get("BuySetup", 0) or 0)
    buy_countdown = int(row.get("BuyCountdown", 0) or 0)
    atr_stretch = _coerce_float(row.get("ATRStretch"))
    rsi = _coerce_float(row.get("RSI"))
    fear_greed = _coerce_float(row.get("FearGreed"))
    close_value = _coerce_float(row.get("Close"))
    ema20 = _coerce_float(row.get("EMA20"))
    ema50 = _coerce_float(row.get("EMA50"))
    ema200 = _coerce_float(row.get("EMA200"))
    ema50_slope = _coerce_float(row.get("EMA50Slope20"))
    trend_delta = _coerce_float(row.get("TrendDelta5"))

    if cycle_score is not None:
        if cycle_score <= 20:
            score += 30
            reasons.append("STL deep cycle")
        elif cycle_score <= 35:
            score += 20
            reasons.append("STL accumulation zone")
        elif cycle_score <= 45:
            score += 10
            reasons.append("STL early pullback")

    if buy_countdown == 13:
        score += 25
        reasons.append("TD buy countdown 13")
    elif buy_setup >= 8:
        score += 18
        reasons.append(f"TD buy setup {buy_setup}")
    elif buy_setup >= 7:
        score += 10
        reasons.append(f"TD buy setup {buy_setup}")

    if atr_stretch is not None:
        if atr_stretch <= -1.8:
            score += 20
            reasons.append("ATR deep pullback")
        elif atr_stretch <= -1.0:
            score += 12
            reasons.append("ATR pullback")
        elif atr_stretch <= -0.5:
            score += 6
            reasons.append("ATR light pullback")

    if rsi is not None:
        if rsi <= 35:
            score += 10
            reasons.append("RSI oversold")
        elif rsi <= 42:
            score += 6
            reasons.append("RSI weak")

    if fear_greed is not None:
        if fear_greed <= 40:
            score += 15
            reasons.append("Macro fear supportive")
        elif fear_greed <= 50:
            score += 8
            reasons.append("Macro neutral-to-fear")

    trend_supportive = all(
        value is not None
        for value in [close_value, ema20, ema50, ema200, trend_delta]
    ) and bool(close_value >= ema200 and ema50 >= ema200 and trend_delta >= 0)

    if trend_supportive and atr_stretch is not None:
        if atr_stretch <= -0.4:
            score += 12
            reasons.append("Trend pullback booster")
        elif atr_stretch <= 0.0:
            score += 6
            reasons.append("Shallow dip in uptrend")

    if trend_supportive and cycle_score is not None and cycle_score <= 55:
        score += 8
        reasons.append("Trend + cycle support")

    if trend_supportive and rsi is not None and rsi <= 52:
        score += 6
        reasons.append("RSI reset in uptrend")

    downtrend_cap = False
    if close_value is not None and ema200 is not None and close_value < ema200:
        downtrend_cap = True
    if ema50 is not None and ema200 is not None and ema50 < ema200:
        downtrend_cap = True
    if ema50_slope is not None and ema50_slope < 0:
        downtrend_cap = True
    if trend_delta is not None and trend_delta < 0:
        downtrend_cap = True

    exception_enabled = cycle_score is not None and cycle_score <= 15 and buy_countdown == 13
    if downtrend_cap and not exception_enabled:
        score = min(score, 55)
        reasons.append("Downtrend cap active")

    return max(0, min(100, int(round(score)))), reasons[:5]


def score_sell_signal(row: pd.Series) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    cycle_score = _coerce_float(row.get("CycleScore"))
    sell_setup = int(row.get("SellSetup", 0) or 0)
    sell_countdown = int(row.get("SellCountdown", 0) or 0)
    atr_stretch = _coerce_float(row.get("ATRStretch"))
    rsi = _coerce_float(row.get("RSI"))
    fear_greed = _coerce_float(row.get("FearGreed"))

    if cycle_score is not None:
        if cycle_score >= 80:
            score += 30
            reasons.append("STL overheated")
        elif cycle_score >= 65:
            score += 20
            reasons.append("STL extended")
        elif cycle_score >= 55:
            score += 10
            reasons.append("STL rich")

    if sell_countdown == 13:
        score += 25
        reasons.append("TD sell countdown 13")
    elif sell_setup >= 8:
        score += 18
        reasons.append(f"TD sell setup {sell_setup}")
    elif sell_setup >= 7:
        score += 10
        reasons.append(f"TD sell setup {sell_setup}")

    if atr_stretch is not None:
        if atr_stretch >= 1.8:
            score += 20
            reasons.append("ATR overextension")
        elif atr_stretch >= 1.0:
            score += 12
            reasons.append("ATR stretched")
        elif atr_stretch >= 0.5:
            score += 4
            reasons.append("ATR mildly extended")

    if rsi is not None:
        if rsi >= 72:
            score += 10
            reasons.append("RSI overbought")
        elif rsi >= 64:
            score += 6
            reasons.append("RSI hot")

    if fear_greed is not None:
        if fear_greed >= 60:
            score += 10
            reasons.append("Macro greed warning")
        elif fear_greed >= 50:
            score += 4
            reasons.append("Macro risk appetite")

    return max(0, min(100, int(round(score)))), reasons[:5]


def build_signal_state(row: Any, preset: SignalPreset) -> dict[str, Any]:
    buy_score, buy_reasons = score_buy_signal(row)
    sell_score, sell_reasons = score_sell_signal(row)
    close_value = _coerce_float(row.get("Close"))
    ema50 = _coerce_float(row.get("EMA50"))
    ema200 = _coerce_float(row.get("EMA200"))
    trend_delta = _coerce_float(row.get("TrendDelta5"))
    cycle_score = _coerce_float(row.get("CycleScore"))
    atr_stretch = _coerce_float(row.get("ATRStretch"))
    rsi = _coerce_float(row.get("RSI"))
    buy_countdown = int(row.get("BuyCountdown", 0) or 0)
    stop_breached = bool(pd.notna(row.get("ChandelierStop")) and row["Close"] <= row["ChandelierStop"])
    trend_supportive = all(value is not None for value in [close_value, ema50, ema200, trend_delta]) and bool(close_value >= ema200 and ema50 >= ema200 and trend_delta >= 0)
    trend_weak = (
        (close_value is not None and ema50 is not None and close_value < ema50)
        or (ema50 is not None and ema200 is not None and ema50 < ema200)
        or (trend_delta is not None and trend_delta < 0)
    )
    deep_reversal = bool(
        (cycle_score is not None and cycle_score <= 15 and buy_countdown == 13)
        or (
            atr_stretch is not None
            and rsi is not None
            and atr_stretch <= -1.5
            and rsi <= 32
            and sell_score <= max(20, preset.watch_threshold - 10)
        )
    )

    if buy_score >= preset.strong_threshold and ((buy_score - sell_score) >= 5 or (deep_reversal and buy_score >= sell_score)):
        state = "Strong Buy"
        key_reasons = buy_reasons
    elif buy_score >= preset.watch_threshold and ((trend_supportive and buy_score >= sell_score) or deep_reversal):
        state = "Buy Watch"
        key_reasons = buy_reasons
    elif sell_score >= preset.strong_threshold and (sell_score - buy_score) >= 8 and trend_weak:
        state = "Strong Sell"
        key_reasons = sell_reasons
    elif stop_breached and trend_weak and sell_score >= max(45, preset.watch_threshold - 5):
        state = "Strong Sell"
        key_reasons = sell_reasons if sell_reasons else ["Protective stop breach"]
    elif sell_score >= preset.watch_threshold and (sell_score - buy_score) >= 5:
        state = "Trim"
        key_reasons = sell_reasons
    else:
        state = "Hold / Neutral"
        key_reasons = buy_reasons[:2] + sell_reasons[:2]

    td_label = str(row.get("TDLabel", "Unavailable"))
    cycle_score = _coerce_float(row.get("CycleScore"))
    fear_greed = _coerce_float(row.get("FearGreed"))
    cycle_text = f"Cycle {_format_float(cycle_score, 1)}" if cycle_score is not None else "Cycle n/a"
    fear_text = f"Fear & Greed {_format_float(fear_greed, 0)}" if fear_greed is not None else "Fear & Greed n/a"
    reason_text = ", ".join(key_reasons[:3]) if key_reasons else "No dominant edge"
    note = f"{state} | B{buy_score}/S{sell_score} | {cycle_text} | {td_label} | {fear_text} | {reason_text}"

    return {
        "State": state,
        "BuyScore": buy_score,
        "SellScore": sell_score,
        "SignalNote": note,
        "BuyReasonText": " | ".join(buy_reasons),
        "SellReasonText": " | ".join(sell_reasons),
    }


def build_state_frame(indicator_frame: pd.DataFrame, preset: SignalPreset) -> pd.DataFrame:
    frame = indicator_frame.dropna(subset=["Close"]).copy()
    if frame.empty:
        return frame

    state_rows = [build_signal_state(row, preset) for row in frame.to_dict("records")]
    state_frame = pd.DataFrame(state_rows, index=frame.index)
    return frame.join(state_frame, how="left")


def build_snapshot(
    ticker: str,
    resolved_symbol: str,
    state_frame: pd.DataFrame,
    warning: str | None,
) -> TickerSnapshot:
    latest = state_frame.dropna(subset=["Close"]).iloc[-1]
    previous_close = state_frame["Close"].iloc[-2] if len(state_frame) > 1 else latest["Close"]
    daily_return = None if previous_close in {None, 0} else float(latest["Close"] / previous_close - 1.0)

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
        note=str(latest.get("SignalNote", "")),
        warning=warning,
    )


def build_watchlist_frame(snapshots: list[TickerSnapshot]) -> pd.DataFrame:
    if not snapshots:
        return pd.DataFrame()
    frame = pd.DataFrame([asdict(snapshot) for snapshot in snapshots])
    frame["state_priority"] = frame["state"].map(STATE_PRIORITY)
    frame = frame.sort_values(["state_priority", "sell_score", "buy_score", "ticker"], ascending=[True, False, False, True])
    return frame.drop(columns=["state_priority"])


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
    }


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
    iter_frame = state_frame.reset_index().rename(columns={"index": "Date"})

    for position, row in enumerate(iter_frame.itertuples(index=False)):
        timestamp = pd.Timestamp(row.Date)
        state = str(row.State)
        price = float(row.Close)
        stop_breached = bool(pd.notna(row.ChandelierStop) and price <= float(row.ChandelierStop))
        note = str(row.SignalNote)
        buy_score = int(row.BuyScore)
        sell_score = int(row.SellScore)

        ema50_value = getattr(row, "EMA50", np.nan)
        ema200_value = getattr(row, "EMA200", np.nan)
        trend_delta_value = getattr(row, "TrendDelta5", np.nan)
        trend_supportive = bool(
            pd.notna(ema50_value)
            and pd.notna(ema200_value)
            and pd.notna(trend_delta_value)
            and price >= float(ema200_value)
            and float(ema50_value) >= float(ema200_value)
            and float(trend_delta_value) >= 0
        )
        trend_weak = bool(
            (pd.notna(ema50_value) and price < float(ema50_value))
            or (pd.notna(ema50_value) and pd.notna(ema200_value) and float(ema50_value) < float(ema200_value))
            or (pd.notna(trend_delta_value) and float(trend_delta_value) < 0)
        )
        stop_exit_ready = stop_breached and (
            trend_weak
            or sell_score >= max(45, preset.watch_threshold - 5)
        )

        entry_signal = state == "Strong Buy" or (state == "Buy Watch" and trend_supportive and buy_score >= sell_score)
        buy_ready = (not in_position) and entry_signal and (position - last_buy_index) >= preset.cooldown_bars
        partial_ready = (
            in_position
            and (not partial_taken)
            and state == "Trim"
            and (position - last_partial_index) >= preset.cooldown_bars
        )
        sell_ready = (
            in_position
            and (state == "Strong Sell" or stop_exit_ready)
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

        if partial_ready and entry_price is not None and position_remaining > 0:
            sell_weight = min(0.5, position_remaining)
            realized_return += sell_weight * (price / entry_price - 1.0)
            position_remaining -= sell_weight
            partial_taken = position_remaining > 0
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
                    "Reason": note,
                }
            )
            continue

        if sell_ready and entry_price is not None and entry_date is not None and position_remaining > 0:
            realized_return += position_remaining * (price / entry_price - 1.0)
            last_sell_index = position
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
    return event_frame, trade_frame, build_replay_summary(trade_frame)


def build_chart_signal_frame(state_frame: pd.DataFrame, preset: SignalPreset) -> pd.DataFrame:
    if state_frame.empty:
        return pd.DataFrame(columns=["Date", "Signal", "Price", "State", "BuyScore", "SellScore"])

    markers: list[dict[str, Any]] = []
    last_buy_index = -10_000
    last_sell_index = -10_000
    previous_buy_active = False
    previous_sell_active = False
    iter_frame = state_frame.reset_index().rename(columns={"index": "Date"})

    for position, row in enumerate(iter_frame.itertuples(index=False)):
        ema50_value = getattr(row, "EMA50", np.nan)
        ema200_value = getattr(row, "EMA200", np.nan)
        trend_delta_value = getattr(row, "TrendDelta5", np.nan)
        trend_supportive = bool(
            pd.notna(ema50_value)
            and pd.notna(ema200_value)
            and pd.notna(trend_delta_value)
            and float(row.Close) >= float(ema200_value)
            and float(ema50_value) >= float(ema200_value)
            and float(trend_delta_value) >= 0
        )
        buy_active = str(row.State) == "Strong Buy" or (str(row.State) == "Buy Watch" and trend_supportive and int(row.BuyScore) >= int(row.SellScore))
        stop_breached = bool(pd.notna(row.ChandelierStop) and float(row.Close) <= float(row.ChandelierStop))
        sell_active = str(row.State) == "Strong Sell" or stop_breached

        buy_trigger = buy_active and not previous_buy_active and (position - last_buy_index) >= preset.cooldown_bars
        sell_trigger = sell_active and not previous_sell_active and (position - last_sell_index) >= preset.cooldown_bars

        if buy_trigger:
            last_buy_index = position
            markers.append(
                {
                    "Date": pd.Timestamp(row.Date),
                    "Signal": "BUY",
                    "Price": float(row.Close),
                    "State": str(row.State),
                    "BuyScore": int(row.BuyScore),
                    "SellScore": int(row.SellScore),
                }
            )
        if sell_trigger:
            last_sell_index = position
            markers.append(
                {
                    "Date": pd.Timestamp(row.Date),
                    "Signal": "SELL",
                    "Price": float(row.Close),
                    "State": str(row.State),
                    "BuyScore": int(row.BuyScore),
                    "SellScore": int(row.SellScore),
                }
            )

        previous_buy_active = buy_active
        previous_sell_active = sell_active

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
    strong_sell = int((watchlist_frame["state"] == "Strong Sell").sum()) if not watchlist_frame.empty else 0
    trim_count = int((watchlist_frame["state"] == "Trim").sum()) if not watchlist_frame.empty else 0
    avg_buy = float(watchlist_frame["buy_score"].mean()) if not watchlist_frame.empty else 0.0
    cards = st.columns(4)
    with cards[0]:
        render_metric_card("Macro Fear & Greed", f"{macro_data['latest_score']:.0f}", macro_data["latest_label"])
    with cards[1]:
        render_metric_card("Strong Buy", str(strong_buy), f"Threshold {preset.strong_threshold}")
    with cards[2]:
        render_metric_card("Strong Sell", str(strong_sell), f"Trim states {trim_count}")
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
    preset: SignalPreset,
) -> None:
    st.markdown('<div class="section-label">Ticker Drill-down</div>', unsafe_allow_html=True)
    left, right = st.columns([1.65, 1.0])
    with left:
        st.plotly_chart(build_price_context_figure(selected_ticker, state_frame, chart_signal_frame, preset), width="stretch", config={"displaylogo": False})
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
                {"Field": "FearGreed", "Value": _format_float(snapshot.fear_greed, 0)},
                {"Field": "StopDistance", "Value": _format_pct(snapshot.stop_distance)},
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
        preset = PROFILE_PRESETS["Balanced"]
        replay_index = pd.date_range("2024-01-02", periods=6, freq="B")
        replay_state = pd.DataFrame(
            {
                "Close": [100.0, 101.0, 103.0, 110.0, 107.0, 98.0],
                "ChandelierStop": [90.0, 91.0, 92.0, 94.0, 95.0, 100.0],
                "State": ["Hold / Neutral", "Strong Buy", "Strong Buy", "Trim", "Hold / Neutral", "Strong Sell"],
                "BuyScore": [15, 81, 78, 34, 30, 20],
                "SellScore": [18, 20, 25, 62, 40, 82],
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
                "State": ["Strong Buy", "Strong Sell", "Strong Buy", "Hold / Neutral", "Strong Buy", "Hold / Neutral"],
                "BuyScore": [80, 20, 78, 30, 82, 25],
                "SellScore": [20, 84, 25, 25, 20, 20],
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
        marker_index = pd.date_range("2024-04-01", periods=7, freq="B")
        marker_state = pd.DataFrame(
            {
                "Close": [100.0, 101.0, 103.0, 104.0, 102.0, 97.0, 96.0],
                "ChandelierStop": [90.0, 90.0, 91.0, 92.0, 95.0, 98.0, 99.0],
                "State": ["Hold / Neutral", "Strong Buy", "Strong Buy", "Buy Watch", "Hold / Neutral", "Strong Sell", "Strong Sell"],
                "BuyScore": [20, 80, 79, 60, 44, 18, 15],
                "SellScore": [15, 20, 21, 30, 45, 82, 84],
            },
            index=marker_index,
        )
        chart_signals = build_chart_signal_frame(marker_state, preset)
        signals = chart_signals["Signal"].tolist()
        passed = signals == ["BUY", "SELL"]
        record("chart_signal_visibility", passed, f"signals={signals}")
    except Exception as exc:
        record("chart_signal_visibility", False, str(exc))

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

    try:
        with st.spinner("Loading watchlist prices and macro context..."):
            macro_started_at = time.perf_counter()
            macro_data = load_macro_fear_greed_cached(end_date.isoformat())
            add_timing("load", "macro_fear_greed", macro_started_at, len(macro_data.get("score_series", pd.Series(dtype=float))))
            for ticker in watchlist:
                ticker_started_at = time.perf_counter()
                price_payloads[ticker] = download_single_history_cached(ticker, start_date.isoformat(), end_date.isoformat())
                add_timing("download", ticker, ticker_started_at, len(price_payloads[ticker]["frame"]))
    except Exception as exc:
        st.error("Dashboard data could not be loaded.")
        st.info(str(exc))
        st.stop()

    state_frames: dict[str, pd.DataFrame] = {}
    snapshots: list[TickerSnapshot] = []
    warnings: list[str] = []
    macro_warning = macro_data.get("warning")
    if macro_warning:
        warnings.append(str(macro_warning))

    with st.spinner("Computing indicators, scores, and replay states..."):
        for ticker in watchlist:
            compute_started_at = time.perf_counter()
            payload = price_payloads[ticker]
            price_frame = payload["frame"]
            warning = payload.get("warning")
            if warning:
                warnings.append(f"{ticker}: {warning}")
            if price_frame.empty:
                continue
            indicators = compute_indicators(price_frame, macro_data["score_series"])
            state_frame = build_state_frame(indicators, preset)
            if state_frame.empty:
                warnings.append(f"{ticker}: indicator stack could not build a usable state frame")
                continue
            state_frames[ticker] = state_frame
            snapshots.append(build_snapshot(ticker, payload["resolved_symbol"], state_frame, warning))
            add_timing("compute", ticker, compute_started_at, len(state_frame))

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

    ticker_options = watchlist_frame["ticker"].tolist()
    selected_ticker = st.selectbox("Drill-down ticker", options=ticker_options, index=0)
    selected_snapshot = next(snapshot for snapshot in snapshots if snapshot.ticker == selected_ticker)
    selected_state_frame = state_frames[selected_ticker]
    chart_started_at = time.perf_counter()
    chart_signal_frame = build_chart_signal_frame(selected_state_frame, preset)
    add_timing("chart_signals", selected_ticker, chart_started_at, len(chart_signal_frame))
    replay_started_at = time.perf_counter()
    event_frame, trade_frame, replay_summary = build_trade_replay_from_state_frame(selected_ticker, selected_state_frame, preset)
    add_timing("replay", selected_ticker, replay_started_at, len(event_frame))
    add_timing("total", "app_run", run_started_at)
    performance_frame = build_performance_frame(timing_rows)
    render_ticker_panel(
        selected_ticker=selected_ticker,
        snapshot=selected_snapshot,
        state_frame=selected_state_frame,
        chart_signal_frame=chart_signal_frame,
        event_frame=event_frame,
        trade_frame=trade_frame,
        replay_summary=replay_summary,
        preset=preset,
    )
    render_diagnostics(diagnostics, performance_frame)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(run_self_tests_cli())
    main()
