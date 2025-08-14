import pandas as pd
import numpy as np

def compute_signals(prices: pd.DataFrame, params: dict) -> dict:
    mom_lb = params['lookback_mom_days']
    mom_skip = params['skip_last_days_for_mom']
    don_n = params['donchian_high_days']
    w_m, w_p, w_b = params['weights']

    mom = prices.shift(mom_skip) / prices.shift(mom_lb) - 1.0
    roll_max_252 = prices.rolling(mom_lb, min_periods=max(10, mom_lb//2)).max()
    prox_52w = prices / roll_max_252
    roll_max_don = prices.rolling(don_n, min_periods=max(10, don_n//2)).max()
    breakout = (prices >= roll_max_don).astype(float)

    def xzs(x):
        return (x - x.mean(skipna=True)) / x.std(skipna=True)

    score = xzs(mom) * w_m + xzs(prox_52w) * w_p + xzs(breakout) * w_b
    return {'mom': mom, 'prox_52w': prox_52w, 'breakout': breakout, 'score': score}
