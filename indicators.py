import pandas as pd
import numpy as np

INDICATOR_NAMES = [
    'mom_12_1',
    'mom_6_1',
    'prox_52w',
    'breakout_100',
    'dist_sma200',
    'low_vol_252',
    'rsi_14'
]

def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_indicators(prices: pd.DataFrame) -> dict:
    roll252 = prices.rolling(252, min_periods=84)
    roll126 = prices.rolling(126, min_periods=42)
    roll100 = prices.rolling(100, min_periods=34)
    roll200 = prices.rolling(200, min_periods=67)

    mom_12_1 = prices.shift(21) / prices.shift(252) - 1.0
    mom_6_1  = prices.shift(21) / prices.shift(126) - 1.0
    prox_52w = prices / roll252.max()
    breakout_100 = (prices >= roll100.max()).astype(float)
    sma200 = roll200.mean()
    dist_sma200 = prices / sma200
    low_vol_252 = -prices.pct_change().rolling(252, min_periods=84).std()
    rsi_14 = prices.apply(_rsi, n=14)

    return {
        'mom_12_1': mom_12_1,
        'mom_6_1': mom_6_1,
        'prox_52w': prox_52w,
        'breakout_100': breakout_100,
        'dist_sma200': dist_sma200,
        'low_vol_252': low_vol_252,
        'rsi_14': rsi_14
    }

def xsec_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1, skipna=True)
    std = df.std(axis=1, skipna=True).replace(0, np.nan)
    return (df.sub(mean, axis=0)).div(std, axis=0)

def score_from_weights(indicators: dict, weights: dict) -> pd.DataFrame:
    parts = []
    for name, ind in indicators.items():
        w = float(weights.get(name, 0.0))
        if w == 0.0:
            continue
        parts.append(xsec_zscore(ind) * w)
    if not parts:
        any_df = next(iter(indicators.values()))
        return any_df * 0.0
    return sum(parts)