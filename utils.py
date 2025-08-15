import pandas as pd
import numpy as np

def cagr(pv: pd.Series) -> float:
    if pv is None or pv.empty or pv.iloc[0] <= 0:
        return np.nan
    days = len(pv)
    return (pv.iloc[-1] / pv.iloc[0]) ** (252/days) - 1.0

def ann_vol(rets: pd.Series) -> float:
    return rets.std() * np.sqrt(252)

def sharpe(rets: pd.Series, rf: float = 0.0) -> float:
    if rets.std() == 0:
        return np.nan
    excess = rets - rf/252.0
    return (excess.mean() / rets.std()) * np.sqrt(252)

def sortino(rets: pd.Series, rf: float = 0.0) -> float:
    downside = rets[rets < 0]
    dd = downside.std()
    if dd == 0 or np.isnan(dd):
        return np.nan
    excess = rets - rf/252.0
    return (excess.mean() / dd) * np.sqrt(252)

def max_drawdown(pv: pd.Series) -> float:
    if pv.empty:
        return np.nan
    dd = pv / pv.cummax() - 1.0
    return dd.min()

def drawdown_series(pv: pd.Series) -> pd.Series:
    return pv / pv.cummax() - 1.0

def stats_from_pv(pv: pd.Series, trades: int) -> dict:
    rets = pv.pct_change().dropna()
    return {
        'CAGR': cagr(pv),
        'Ann.Vol': ann_vol(rets),
        'Sharpe(0%)': sharpe(rets),
        'Sortino(0%)': sortino(rets),
        'MaxDD': max_drawdown(pv),
        'NumTrades': trades
    }