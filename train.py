import itertools
import numpy as np
import pandas as pd
import config
from signals import compute_signals
from backtest import run_backtest

def grid_search(prices_train: pd.DataFrame) -> tuple:
    keys = list(config.GRID.keys())
    vals = [config.GRID[k] for k in keys]
    best = None
    best_score = -1e9

    for combo in itertools.product(*vals):
        params = dict(zip(keys, combo))
        # merge with defaults for fields that are not in grid
        full = {**config.DEFAULT_PARAMS, **params}
        sig = compute_signals(prices_train, full)
        res = run_backtest(prices_train, sig, full)
        sharpe = res['stats'].get('Sharpe(0%)', np.nan)
        if pd.notna(sharpe) and sharpe > best_score:
            best_score = sharpe
            best = full

    return best, best_score
