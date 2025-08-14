import pandas as pd
import numpy as np

def monthly_rebalance_dates(index, freq='M'):
    return pd.date_range(index.min(), index.max(), freq=freq).intersection(index)

def stats_from_pv(pv: pd.Series) -> dict:
    if pv is None or pv.empty:
        return {'start': '', 'end': '', 'CAGR': np.nan, 'Ann.Vol': np.nan, 'Sharpe(0%)': np.nan, 'MaxDD': np.nan}
    rets = pv.pct_change().fillna(0.0)
    out = {
        'start': str(pv.index.min().date()),
        'end': str(pv.index.max().date()),
        'CAGR': ((pv.iloc[-1] / pv.iloc[0]) ** (252/len(pv)) - 1.0) if len(pv)>1 else np.nan,
        'Ann.Vol': rets.std() * np.sqrt(252),
        'Sharpe(0%)': (rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else np.nan,
        'MaxDD': (pv / pv.cummax() - 1.0).min()
    }
    return out
