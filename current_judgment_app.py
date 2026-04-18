from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf


APP_TITLE = "Swing Trade Current Judgment Dashboard"
CACHE_TTL_SECONDS = 3600
DEFAULT_WATCHLIST = ("SPY", "QQQ", "NVDA", "MSFT", "AAPL", "SOXX")
DEFAULT_BREADTH_UNIVERSE = (
    "SPY",
    "QQQ",
    "IWM",
    "SMH",
    "SOXX",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "AVGO",
    "AMD",
    "NFLX",
    "TSLA",
    "CRM",
    "NOW",
    "PANW",
)
REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
REGIME_COLORS = {
    "Trend Healthy": "#15803d",
    "Overheated but Intact": "#d97706",
    "Scale Out / Warning": "#ea580c",
    "Neutral / Watch": "#2563eb",
    "Exit / Defensive": "#dc2626",
}


@dataclass(frozen=True)
class JudgmentThresholds:
    overheat_rsi: float = 70.0
    overheat_stretch_atr: float = 2.5
    trend_adx: float = 25.0
    scale_out_confirmation_count: int = 2
    exit_confirmation_count: int = 3


@dataclass(frozen=True)
class CurrentJudgment:
    ticker: str
    as_of: pd.Timestamp
    last_close: float
    daily_return: float
    regime: str
    action: str
    hold_score: int
    trend_score: int
    heat_score: int
    confirmation_count: int
    breadth_state: str
    supertrend_state: str
    rsi: float | None
    stretch_atr: float | None
    adx: float | None
    macd_hist: float | None
    squeeze_momentum: float | None
    wvf: float | None
    close_vs_stop_pct: float | None
    rationale: tuple[str, ...]


def configure_page() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="SW", layout="wide")


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .hero {
            padding: 1.35rem 1.5rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #0f172a 0%, #14532d 50%, #fef3c7 100%);
            color: #f8fafc;
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.08;
        }
        .hero p {
            margin-top: 0.55rem;
            margin-bottom: 0;
            color: rgba(248, 250, 252, 0.88);
        }
        .metric-card {
            padding: 1rem 1.05rem;
            border-radius: 16px;
            background: #f8fafc;
            border: 1px solid rgba(148, 163, 184, 0.28);
            min-height: 108px;
        }
        .metric-card .eyebrow {
            font-size: 0.82rem;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-card .value {
            font-size: 1.7rem;
            font-weight: 700;
            margin-top: 0.18rem;
            color: #0f172a;
        }
        .metric-card .caption {
            margin-top: 0.25rem;
            color: #475569;
            font-size: 0.92rem;
        }
        .section-label {
            margin-top: 0.35rem;
            color: #0f172a;
            font-weight: 700;
            font-size: 1.12rem;
        }
        .signal-pill {
            display: inline-block;
            padding: 0.36rem 0.65rem;
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


def parse_tickers(raw: str) -> list[str]:
    tokens = [item.strip().upper() for item in raw.replace("\n", ",").split(",")]
    return [item for item in tokens if item]


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100.0:.1f}%"


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.{digits}f}"


def build_controls() -> tuple[list[str], int, JudgmentThresholds, bool]:
    st.sidebar.subheader("Current Judgment Controls")
    watchlist_raw = st.sidebar.text_input(
        "Watchlist tickers",
        value=", ".join(DEFAULT_WATCHLIST),
        help="Comma-separated ticker list",
    )
    history_years = st.sidebar.slider("History lookback (years)", min_value=1, max_value=10, value=3, step=1)
    overheat_rsi = st.sidebar.slider("Overheat RSI", min_value=60.0, max_value=85.0, value=70.0, step=1.0)
    overheat_stretch = st.sidebar.slider("Overheat stretch (ATR)", min_value=1.5, max_value=4.0, value=2.5, step=0.1)
    trend_adx = st.sidebar.slider("Trend ADX floor", min_value=15.0, max_value=40.0, value=25.0, step=1.0)
    refresh = st.sidebar.button("Refresh cached data", use_container_width=True)
    st.sidebar.caption("Standalone app. No local package import is required.")

    watchlist = parse_tickers(watchlist_raw) or list(DEFAULT_WATCHLIST)
    thresholds = JudgmentThresholds(
        overheat_rsi=float(overheat_rsi),
        overheat_stretch_atr=float(overheat_stretch),
        trend_adx=float(trend_adx),
    )
    return watchlist, int(history_years), thresholds, refresh


def _normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS))
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized = normalized.sort_index()
    if "Adj Close" in normalized.columns and "Close" not in normalized.columns:
        normalized = normalized.rename(columns={"Adj Close": "Close"})
    for column in REQUIRED_OHLCV_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    normalized = normalized.loc[:, list(REQUIRED_OHLCV_COLUMNS)]
    normalized = normalized.dropna(subset=["Open", "High", "Low", "Close"])
    normalized["Volume"] = normalized["Volume"].fillna(0.0)
    return normalized


def _split_download_frame(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    if raw is None or raw.empty:
        return {ticker: pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS)) for ticker in tickers}

    frames: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        tickers_in_frame = set(raw.columns.get_level_values(-1))
        for ticker in tickers:
            if ticker not in tickers_in_frame:
                frames[ticker] = pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS))
                continue
            frames[ticker] = _normalize_ohlcv_frame(raw.xs(ticker, axis=1, level=-1, drop_level=True))
        return frames

    if len(tickers) == 1:
        return {tickers[0]: _normalize_ohlcv_frame(raw)}

    return {ticker: pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS)) for ticker in tickers}


def download_price_history(tickers: list[str], start_date: date, end_date: date) -> dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
        threads=False,
        group_by="column",
    )
    return _split_download_frame(raw, tickers)


def _prepare_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS))
    frame = price_df.copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        return pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS))
    return frame.loc[:, list(REQUIRED_OHLCV_COLUMNS)].dropna(subset=["Open", "High", "Low", "Close"])


def _true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["Close"].shift(1)
    return pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - prev_close).abs(),
            (frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _wilder_average(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def _compute_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = _wilder_average(gains, window)
    avg_loss = _wilder_average(losses, window)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(100.0).where(avg_gain.notna() | avg_loss.notna())


def _compute_adx(frame: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = frame["High"].diff()
    down_move = -frame["Low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0), index=frame.index)
    atr = _wilder_average(_true_range(frame), window)
    plus_di = 100.0 * _wilder_average(plus_dm, window) / atr.replace(0.0, np.nan)
    minus_di = 100.0 * _wilder_average(minus_dm, window) / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = _wilder_average(dx, window)
    return plus_di, minus_di, adx


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered**2))

    def _slope(values: np.ndarray) -> float:
        centered = values - values.mean()
        return float(np.dot(centered, x_centered) / denominator)

    return series.rolling(window).apply(_slope, raw=True)


def _compute_macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _compute_wvf(frame: pd.DataFrame, window: int = 22, band_window: int = 20, std_multiplier: float = 2.0) -> tuple[pd.Series, pd.Series]:
    highest_close = frame["Close"].rolling(window).max()
    wvf = (highest_close - frame["Low"]) / highest_close.replace(0.0, np.nan) * 100.0
    upper_band = wvf.rolling(band_window).mean() + (wvf.rolling(band_window).std() * std_multiplier)
    spike = wvf > upper_band
    return wvf, spike.fillna(False)


def _compute_squeeze(
    frame: pd.DataFrame,
    window: int = 20,
    bb_mult: float = 2.0,
    kc_mult: float = 1.5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = frame["Close"]
    basis = close.rolling(window).mean()
    deviation = close.rolling(window).std()
    bb_upper = basis + (deviation * bb_mult)
    bb_lower = basis - (deviation * bb_mult)

    tr = _true_range(frame)
    kc_basis = close.rolling(window).mean()
    kc_range = tr.rolling(window).mean()
    kc_upper = kc_basis + (kc_range * kc_mult)
    kc_lower = kc_basis - (kc_range * kc_mult)
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    highest_high = frame["High"].rolling(window).max()
    lowest_low = frame["Low"].rolling(window).min()
    midpoint = ((highest_high + lowest_low) / 2.0 + basis) / 2.0
    squeeze_base = close - midpoint
    momentum = _rolling_slope(squeeze_base, window) * window
    return squeeze_on.fillna(False), momentum, momentum.diff()


def _compute_supertrend(frame: pd.DataFrame, window: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    atr = _wilder_average(_true_range(frame), window)
    hl2 = (frame["High"] + frame["Low"]) / 2.0
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend = pd.Series(index=frame.index, dtype=float)
    direction = pd.Series(index=frame.index, dtype=float)

    for idx in range(len(frame)):
        if idx == 0:
            supertrend.iloc[idx] = np.nan
            direction.iloc[idx] = 1.0
            continue

        prev_idx = idx - 1
        if pd.notna(final_upper.iloc[prev_idx]):
            if basic_upper.iloc[idx] < final_upper.iloc[prev_idx] or frame["Close"].iloc[prev_idx] > final_upper.iloc[prev_idx]:
                final_upper.iloc[idx] = basic_upper.iloc[idx]
            else:
                final_upper.iloc[idx] = final_upper.iloc[prev_idx]
        if pd.notna(final_lower.iloc[prev_idx]):
            if basic_lower.iloc[idx] > final_lower.iloc[prev_idx] or frame["Close"].iloc[prev_idx] < final_lower.iloc[prev_idx]:
                final_lower.iloc[idx] = basic_lower.iloc[idx]
            else:
                final_lower.iloc[idx] = final_lower.iloc[prev_idx]

        previous_supertrend = supertrend.iloc[prev_idx]
        if pd.isna(previous_supertrend):
            if frame["Close"].iloc[idx] <= final_upper.iloc[idx]:
                supertrend.iloc[idx] = final_upper.iloc[idx]
                direction.iloc[idx] = -1.0
            else:
                supertrend.iloc[idx] = final_lower.iloc[idx]
                direction.iloc[idx] = 1.0
            continue

        if previous_supertrend == final_upper.iloc[prev_idx]:
            if frame["Close"].iloc[idx] <= final_upper.iloc[idx]:
                supertrend.iloc[idx] = final_upper.iloc[idx]
                direction.iloc[idx] = -1.0
            else:
                supertrend.iloc[idx] = final_lower.iloc[idx]
                direction.iloc[idx] = 1.0
        else:
            if frame["Close"].iloc[idx] >= final_lower.iloc[idx]:
                supertrend.iloc[idx] = final_lower.iloc[idx]
                direction.iloc[idx] = 1.0
            else:
                supertrend.iloc[idx] = final_upper.iloc[idx]
                direction.iloc[idx] = -1.0

    return supertrend, direction


def compute_market_breadth_context(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    usable_frames = {ticker: _prepare_frame(frame) for ticker, frame in price_frames.items() if not frame.empty}
    if not usable_frames:
        return pd.DataFrame(
            columns=[
                "AdvanceRatio",
                "BreadthEMA",
                "BreadthWeak",
                "BreadthThrust",
                "BreadthThrustActive",
                "PctAboveEMA20",
                "BreadthSupportive",
            ]
        )

    close_frame = pd.concat({ticker: frame["Close"] for ticker, frame in usable_frames.items()}, axis=1).sort_index()
    advance_ratio = close_frame.pct_change().gt(0.0).mean(axis=1)
    breadth_ema = advance_ratio.ewm(span=10, adjust=False, min_periods=10).mean()
    rolling_min = breadth_ema.rolling(10).min()
    breadth_thrust = breadth_ema.ge(0.615) & rolling_min.le(0.40)
    breadth_thrust_active = breadth_thrust.rolling(20).max().fillna(0.0).astype(bool)

    ema20_panel = close_frame.rolling(20).mean()
    pct_above_ema20 = close_frame.gt(ema20_panel).mean(axis=1)
    breadth_weak = breadth_ema.lt(0.50) | pct_above_ema20.lt(0.50) | breadth_ema.lt(breadth_ema.shift(3))
    breadth_supportive = breadth_thrust_active | breadth_ema.gt(0.52) | pct_above_ema20.gt(0.55)

    return pd.DataFrame(
        {
            "AdvanceRatio": advance_ratio,
            "BreadthEMA": breadth_ema,
            "BreadthWeak": breadth_weak.fillna(False),
            "BreadthThrust": breadth_thrust.fillna(False),
            "BreadthThrustActive": breadth_thrust_active.fillna(False),
            "PctAboveEMA20": pct_above_ema20,
            "BreadthSupportive": breadth_supportive.fillna(False),
        }
    )


def compute_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_frame(price_df)
    if frame.empty:
        return frame

    atr = _wilder_average(_true_range(frame), 14)
    _, _, adx = _compute_adx(frame, 14)
    ema = frame["Close"].ewm(span=20, adjust=False, min_periods=20).mean()
    rsi = _compute_rsi(frame["Close"], 14)
    chandelier_stop = frame["High"].rolling(22).max() - (atr * 3.0)
    stretch_atr = (frame["Close"] - ema) / atr.replace(0.0, np.nan)
    macd_line, macd_signal, macd_hist = _compute_macd(frame["Close"], 12, 26, 9)
    wvf, wvf_spike = _compute_wvf(frame)
    squeeze_on, squeeze_momentum, squeeze_momentum_delta = _compute_squeeze(frame)
    elder_ema = frame["Close"].ewm(span=13, adjust=False, min_periods=13).mean()
    elder_green = elder_ema.gt(elder_ema.shift(1)) & macd_hist.gt(macd_hist.shift(1))
    elder_red = elder_ema.lt(elder_ema.shift(1)) & macd_hist.lt(macd_hist.shift(1))
    supertrend, supertrend_direction = _compute_supertrend(frame)

    indicator_frame = frame.copy()
    indicator_frame["ATR"] = atr
    indicator_frame["ADX"] = adx
    indicator_frame["EMA"] = ema
    indicator_frame["RSI"] = rsi
    indicator_frame["ChandelierStop"] = chandelier_stop
    indicator_frame["StretchATR"] = stretch_atr
    indicator_frame["MACD"] = macd_line
    indicator_frame["MACDSignal"] = macd_signal
    indicator_frame["MACDHist"] = macd_hist
    indicator_frame["MACDHistDelta"] = macd_hist.diff()
    indicator_frame["WVF"] = wvf
    indicator_frame["WVFSpike"] = wvf_spike
    indicator_frame["SqueezeOn"] = squeeze_on
    indicator_frame["SqueezeMomentum"] = squeeze_momentum
    indicator_frame["SqueezeMomentumDelta"] = squeeze_momentum_delta
    indicator_frame["ElderImpulseGreen"] = elder_green.fillna(False)
    indicator_frame["ElderImpulseRed"] = elder_red.fillna(False)
    indicator_frame["SuperTrend"] = supertrend
    indicator_frame["SuperTrendDirection"] = supertrend_direction
    indicator_frame["SuperTrendBull"] = supertrend_direction.gt(0.0).fillna(False)
    indicator_frame["SuperTrendBear"] = supertrend_direction.lt(0.0).fillna(False)
    return indicator_frame


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_dashboard_data(
    *,
    watchlist: tuple[str, ...],
    breadth_universe: tuple[str, ...],
    history_years: int,
) -> dict[str, Any]:
    fetch_start = date.today() - timedelta(days=(history_years * 365) + 60)
    all_tickers = tuple(sorted(set(watchlist + breadth_universe)))
    price_frames = download_price_history(list(all_tickers), fetch_start, date.today())

    breadth_frames = {
        ticker: frame for ticker, frame in price_frames.items() if ticker in breadth_universe and not frame.empty
    }
    breadth_context = compute_market_breadth_context(breadth_frames)

    indicator_frames: dict[str, pd.DataFrame] = {}
    for ticker in watchlist:
        frame = price_frames.get(ticker, pd.DataFrame())
        if frame.empty:
            continue
        indicator = compute_indicators(frame).join(breadth_context, how="left")
        if "BreadthWeak" not in indicator.columns:
            indicator["BreadthWeak"] = False
        if "BreadthSupportive" not in indicator.columns:
            indicator["BreadthSupportive"] = False
        indicator["BreadthWeak"] = indicator["BreadthWeak"].fillna(False)
        indicator["BreadthSupportive"] = indicator["BreadthSupportive"].fillna(False)
        indicator_frames[ticker] = indicator

    diagnostics_frame = pd.DataFrame(
        [
            {
                "Provider": "yfinance",
                "Requested": len(all_tickers),
                "Returned": sum(1 for frame in price_frames.values() if not frame.empty),
                "Missing": sum(1 for frame in price_frames.values() if frame.empty),
                "Batches": 1,
                "Cache hits": "cache_data",
                "Start": fetch_start.isoformat(),
                "End": date.today().isoformat(),
            }
        ]
    )
    return {
        "price_frames": price_frames,
        "indicator_frames": indicator_frames,
        "breadth_context": breadth_context,
        "provider_diagnostics_frame": diagnostics_frame,
        "provider_name": "yfinance",
    }


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


def _signal_confirmation_count(row: pd.Series) -> int:
    confirmations = [
        bool(pd.notna(row.get("MACDHist")) and row["MACDHist"] < 0.0),
        bool(pd.notna(row.get("SqueezeMomentum")) and row["SqueezeMomentum"] < 0.0),
        bool(row.get("ElderImpulseRed", False)),
        bool(row.get("SuperTrendBear", False)),
        bool(row.get("BreadthWeak", False)),
        bool(row.get("WVFSpike", False)),
    ]
    return int(sum(confirmations))


def evaluate_current_judgment(
    ticker: str,
    indicator_frame: pd.DataFrame,
    thresholds: JudgmentThresholds,
) -> CurrentJudgment:
    frame = indicator_frame.dropna(subset=["Close"]).copy()
    latest = frame.iloc[-1]
    previous_close = frame["Close"].iloc[-2] if len(frame) > 1 else latest["Close"]
    daily_return = float(latest["Close"] / previous_close - 1.0) if previous_close else 0.0
    confirmation_count = _signal_confirmation_count(latest)

    heat_flags = [
        bool(pd.notna(latest.get("RSI")) and latest["RSI"] >= thresholds.overheat_rsi),
        bool(pd.notna(latest.get("StretchATR")) and latest["StretchATR"] >= thresholds.overheat_stretch_atr),
    ]
    heat_score = int(sum(heat_flags))
    trend_flags = [
        bool(pd.notna(latest.get("ADX")) and latest["ADX"] >= thresholds.trend_adx),
        bool(pd.notna(latest.get("MACDHist")) and latest["MACDHist"] > 0.0),
        bool(latest.get("SuperTrendBull", False)),
        bool(latest.get("ElderImpulseGreen", False)),
        bool(latest.get("BreadthSupportive", False)),
        bool(pd.notna(latest.get("EMA")) and latest["Close"] > latest["EMA"]),
    ]
    trend_score = int(sum(trend_flags))

    close_vs_stop_pct = None
    close_below_stop = False
    if pd.notna(latest.get("ChandelierStop")) and latest["ChandelierStop"] != 0:
        close_vs_stop_pct = float(latest["Close"] / latest["ChandelierStop"] - 1.0)
        close_below_stop = latest["Close"] <= latest["ChandelierStop"]

    breadth_supportive = bool(latest.get("BreadthSupportive", False))
    breadth_weak = bool(latest.get("BreadthWeak", False))
    supertrend_bear = bool(latest.get("SuperTrendBear", False))

    rationale: list[str] = []
    if pd.notna(latest.get("RSI")):
        rationale.append(f"RSI {_format_float(float(latest['RSI']), 1)}")
    if pd.notna(latest.get("StretchATR")):
        rationale.append(f"Stretch {_format_float(float(latest['StretchATR']), 2)} ATR")
    if pd.notna(latest.get("ADX")):
        rationale.append(f"ADX {_format_float(float(latest['ADX']), 1)}")
    if pd.notna(latest.get("MACDHist")):
        rationale.append(f"MACD hist {_format_float(float(latest['MACDHist']), 2)}")
    if bool(latest.get("WVFSpike", False)):
        rationale.append("Vix Fix stress spike")
    if breadth_supportive:
        rationale.append("Breadth supportive")
    if breadth_weak:
        rationale.append("Breadth weakening")
    if bool(latest.get("ElderImpulseRed", False)):
        rationale.append("Elder impulse red")
    if bool(latest.get("SuperTrendBear", False)):
        rationale.append("SuperTrend bearish")

    hold_score = 50 + (trend_score * 8) - (confirmation_count * 10) - (heat_score * 8)
    if breadth_supportive:
        hold_score += 6
    if close_below_stop:
        hold_score -= 18
    hold_score = max(0, min(100, int(round(hold_score))))

    if close_below_stop or (supertrend_bear and confirmation_count >= thresholds.exit_confirmation_count):
        regime = "Exit / Defensive"
        action = "Tighten aggressively or exit."
    elif heat_score == 2 and confirmation_count >= thresholds.scale_out_confirmation_count:
        if breadth_supportive and bool(latest.get("SuperTrendBull", False)):
            regime = "Overheated but Intact"
            action = "Trim partial only and keep the core trend."
        else:
            regime = "Scale Out / Warning"
            action = "Reduce 25-50% and trail the remainder."
    elif heat_score >= 1 and trend_score >= 4 and confirmation_count <= 1:
        regime = "Overheated but Intact"
        action = "Hold core, but do not chase. Trim only into strength."
    elif trend_score >= 4 and confirmation_count <= 1:
        regime = "Trend Healthy"
        action = "Hold core and wait for pullbacks instead of selling strength."
    else:
        regime = "Neutral / Watch"
        action = "No edge shift yet. Keep stops updated and wait."

    breadth_state = "Supportive" if breadth_supportive else "Weak" if breadth_weak else "Mixed"
    supertrend_state = "Bullish" if bool(latest.get("SuperTrendBull", False)) else "Bearish"

    return CurrentJudgment(
        ticker=ticker,
        as_of=pd.Timestamp(frame.index[-1]),
        last_close=float(latest["Close"]),
        daily_return=daily_return,
        regime=regime,
        action=action,
        hold_score=hold_score,
        trend_score=trend_score,
        heat_score=heat_score,
        confirmation_count=confirmation_count,
        breadth_state=breadth_state,
        supertrend_state=supertrend_state,
        rsi=float(latest["RSI"]) if pd.notna(latest.get("RSI")) else None,
        stretch_atr=float(latest["StretchATR"]) if pd.notna(latest.get("StretchATR")) else None,
        adx=float(latest["ADX"]) if pd.notna(latest.get("ADX")) else None,
        macd_hist=float(latest["MACDHist"]) if pd.notna(latest.get("MACDHist")) else None,
        squeeze_momentum=float(latest["SqueezeMomentum"]) if pd.notna(latest.get("SqueezeMomentum")) else None,
        wvf=float(latest["WVF"]) if pd.notna(latest.get("WVF")) else None,
        close_vs_stop_pct=close_vs_stop_pct,
        rationale=tuple(rationale[:8]),
    )


def build_judgment_frame(
    indicator_frames: dict[str, pd.DataFrame],
    thresholds: JudgmentThresholds,
) -> tuple[pd.DataFrame, dict[str, CurrentJudgment]]:
    judgments: list[CurrentJudgment] = []
    by_ticker: dict[str, CurrentJudgment] = {}
    for ticker, frame in indicator_frames.items():
        if frame.empty:
            continue
        judgment = evaluate_current_judgment(ticker, frame, thresholds)
        judgments.append(judgment)
        by_ticker[ticker] = judgment

    if not judgments:
        return pd.DataFrame(), {}

    table = pd.DataFrame([asdict(item) for item in judgments]).sort_values(["hold_score", "ticker"], ascending=[True, True])
    return table, by_ticker


def build_regime_gauge(judgment: CurrentJudgment) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=judgment.hold_score,
            title={"text": f"Hold Score<br><span style='font-size:0.9rem'>{judgment.regime}</span>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": REGIME_COLORS[judgment.regime]},
                "steps": [
                    {"range": [0, 25], "color": "#fee2e2"},
                    {"range": [25, 50], "color": "#ffedd5"},
                    {"range": [50, 75], "color": "#fef3c7"},
                    {"range": [75, 100], "color": "#dcfce7"},
                ],
            },
        )
    )
    fig.update_layout(margin={"l": 10, "r": 10, "t": 50, "b": 10}, height=280)
    return fig


def build_price_context_figure(ticker: str, indicator_frame: pd.DataFrame, thresholds: JudgmentThresholds) -> go.Figure:
    frame = indicator_frame.tail(120).copy().reset_index().rename(columns={"index": "Date"})
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.58, 0.2, 0.22],
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )
    fig.add_trace(
        go.Candlestick(
            x=frame["Date"],
            open=frame["Open"],
            high=frame["High"],
            low=frame["Low"],
            close=frame["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["EMA"], name="EMA", line={"color": "#2563eb", "width": 2}), row=1, col=1)
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["SuperTrend"], name="SuperTrend", line={"color": "#15803d", "width": 2}), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=frame["Date"], y=frame["ChandelierStop"], name="Chandelier Stop", line={"color": "#dc2626", "width": 2, "dash": "dash"}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=frame["Date"],
            y=frame["MACDHist"],
            name="MACD Hist",
            marker_color=["#16a34a" if value >= 0 else "#dc2626" for value in frame["MACDHist"].fillna(0.0)],
        ),
        row=2,
        col=1,
    )
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["SqueezeMomentum"], name="Squeeze Momentum", line={"color": "#7c3aed", "width": 2}), row=2, col=1)
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["RSI"], name="RSI", line={"color": "#0f766e", "width": 2}), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["StretchATR"], name="Stretch ATR", line={"color": "#ea580c", "width": 2}), row=3, col=1, secondary_y=True)
    fig.add_hline(y=thresholds.overheat_rsi, row=3, col=1, line_dash="dot", line_color="#0f766e")
    fig.add_hline(y=thresholds.overheat_stretch_atr, row=3, col=1, secondary_y=True, line_dash="dot", line_color="#ea580c")
    fig.update_layout(
        title=f"{ticker} Price and Signal Context",
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.04},
        height=760,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Momentum", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Stretch", row=3, col=1, secondary_y=True)
    return fig


def build_breadth_context_figure(breadth_context: pd.DataFrame) -> go.Figure:
    frame = breadth_context.tail(120).reset_index().rename(columns={"index": "Date"})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["BreadthEMA"], name="Breadth EMA", line={"color": "#2563eb", "width": 3}))
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["PctAboveEMA20"], name="% Above EMA20", line={"color": "#d97706", "width": 2}))
    fig.add_trace(
        go.Scatter(
            x=frame.loc[frame["BreadthThrustActive"], "Date"],
            y=frame.loc[frame["BreadthThrustActive"], "BreadthEMA"],
            mode="markers",
            marker={"color": "#15803d", "size": 8},
            name="Breadth Thrust Active",
        )
    )
    fig.add_hline(y=0.50, line_dash="dash", line_color="#94a3b8")
    fig.add_hline(y=0.615, line_dash="dot", line_color="#15803d")
    fig.update_layout(
        title="Market Breadth Context",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        yaxis_title="Ratio",
        height=320,
        legend={"orientation": "h", "y": 1.02},
    )
    return fig


def render_header(provider_name: str, judgments: dict[str, CurrentJudgment]) -> None:
    latest_as_of = max(item.as_of for item in judgments.values()).date().isoformat() if judgments else "n/a"
    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_TITLE}</h1>
            <p>Provider: {provider_name} | Last completed daily bar: {latest_as_of} | Standalone Streamlit app for swing-trade context and exit timing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(judgment_frame: pd.DataFrame) -> None:
    high_risk = int((judgment_frame["regime"] == "Exit / Defensive").sum()) if not judgment_frame.empty else 0
    extended = int(judgment_frame["regime"].isin(["Overheated but Intact", "Scale Out / Warning"]).sum()) if not judgment_frame.empty else 0
    avg_score = float(judgment_frame["hold_score"].mean()) if not judgment_frame.empty else 0.0
    avg_confirmations = float(judgment_frame["confirmation_count"].mean()) if not judgment_frame.empty else 0.0
    cards = st.columns(4)
    with cards[0]:
        render_metric_card("Watchlist", str(len(judgment_frame)), "Tickers scored")
    with cards[1]:
        render_metric_card("High Risk", str(high_risk), "Exit / defensive states")
    with cards[2]:
        render_metric_card("Extended", str(extended), "Overheated or scale-out states")
    with cards[3]:
        render_metric_card("Average Hold Score", f"{avg_score:.0f}", f"Avg confirmations {avg_confirmations:.1f}")


def render_summary_table(judgment_frame: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">Watchlist Summary</div>', unsafe_allow_html=True)
    if judgment_frame.empty:
        st.warning("No usable indicator frames were loaded for the watchlist.")
        return
    display = judgment_frame.loc[:, ["ticker", "as_of", "last_close", "daily_return", "regime", "action", "hold_score", "confirmation_count", "rsi", "stretch_atr", "adx", "breadth_state", "supertrend_state"]].copy()
    display.columns = ["Ticker", "As Of", "Close", "1D Return", "Regime", "Action", "Hold Score", "Confirmations", "RSI", "Stretch ATR", "ADX", "Breadth", "SuperTrend"]
    display["Close"] = display["Close"].map(_format_float)
    display["1D Return"] = display["1D Return"].map(_format_pct)
    display["RSI"] = display["RSI"].map(lambda value: _format_float(value, 1))
    display["Stretch ATR"] = display["Stretch ATR"].map(lambda value: _format_float(value, 2))
    display["ADX"] = display["ADX"].map(lambda value: _format_float(value, 1))
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_judgment_panel(selected_ticker: str, judgment: CurrentJudgment, indicator_frame: pd.DataFrame, breadth_context: pd.DataFrame, thresholds: JudgmentThresholds) -> None:
    st.markdown('<div class="section-label">Ticker Drill-down</div>', unsafe_allow_html=True)
    left, right = st.columns([1.6, 1.0])
    with left:
        st.plotly_chart(build_price_context_figure(selected_ticker, indicator_frame, thresholds), use_container_width=True, config={"displaylogo": False})
    with right:
        st.plotly_chart(build_regime_gauge(judgment), use_container_width=True, config={"displaylogo": False})
        st.plotly_chart(build_breadth_context_figure(breadth_context), use_container_width=True, config={"displaylogo": False})

    st.markdown(f"**Current Action**: {judgment.action}")
    pill_html = "".join(f'<span class="signal-pill">{item}</span>' for item in judgment.rationale)
    st.markdown(pill_html or '<span class="signal-pill">No extra signals</span>', unsafe_allow_html=True)

    detail = pd.DataFrame(
        [
            {"Field": "Regime", "Value": judgment.regime},
            {"Field": "Hold score", "Value": str(judgment.hold_score)},
            {"Field": "Trend score", "Value": str(judgment.trend_score)},
            {"Field": "Heat score", "Value": str(judgment.heat_score)},
            {"Field": "Confirmation count", "Value": str(judgment.confirmation_count)},
            {"Field": "Breadth state", "Value": judgment.breadth_state},
            {"Field": "SuperTrend state", "Value": judgment.supertrend_state},
            {"Field": "Close vs stop", "Value": _format_pct(judgment.close_vs_stop_pct)},
            {"Field": "MACD histogram", "Value": _format_float(judgment.macd_hist, 2)},
            {"Field": "Squeeze momentum", "Value": _format_float(judgment.squeeze_momentum, 2)},
            {"Field": "Williams Vix Fix", "Value": _format_float(judgment.wvf, 2)},
        ]
    )
    st.dataframe(detail, use_container_width=True, hide_index=True)


def render_diagnostics(result: dict[str, Any]) -> None:
    with st.expander("Data diagnostics", expanded=False):
        st.dataframe(result["provider_diagnostics_frame"], use_container_width=True, hide_index=True)


def main() -> None:
    configure_page()
    apply_style()
    watchlist, history_years, thresholds, refresh = build_controls()

    if refresh:
        st.cache_data.clear()
        st.sidebar.success("Cached data cleared. Rerun to fetch fresh prices.")

    try:
        with st.spinner("Loading watchlist, market breadth, and current signal context..."):
            result = load_dashboard_data(
                watchlist=tuple(watchlist),
                breadth_universe=DEFAULT_BREADTH_UNIVERSE,
                history_years=history_years,
            )
    except Exception as exc:
        st.error("Current judgment data could not be loaded.")
        st.info(str(exc))
        st.stop()

    judgment_frame, judgments = build_judgment_frame(result["indicator_frames"], thresholds)
    render_header(result["provider_name"], judgments)
    render_kpis(judgment_frame)
    render_summary_table(judgment_frame)

    if judgment_frame.empty:
        render_diagnostics(result)
        st.stop()

    selected_ticker = st.selectbox("Drill-down ticker", options=judgment_frame["ticker"].tolist(), index=0)
    render_judgment_panel(selected_ticker, judgments[selected_ticker], result["indicator_frames"][selected_ticker], result["breadth_context"], thresholds)
    render_diagnostics(result)


if __name__ == "__main__":
    main()
