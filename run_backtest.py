import pandas as pd
import numpy as np
from pathlib import Path
import config
import data
from signals import compute_signals
from backtest import run_backtest
from train import grid_search

def main():
    out = config.OUT_DIR

    # Load
    prices = data.load_prices()
    prices = data.clean_prices(prices)
    ibov = data.load_ibov()

    # Align with benchmark dates (optional but keeps things tidy)
    prices, ibov = data.align_with_benchmark(prices, ibov)

    # Split
    px_train = data.split_by_dates(prices, config.TRAIN_START, config.TRAIN_END)
    px_test  = data.split_by_dates(prices, config.TEST_START,  config.TEST_END)
    ib_train = data.split_by_dates(ibov.to_frame('IBOV'), config.TRAIN_START, config.TRAIN_END)['IBOV']
    ib_test  = data.split_by_dates(ibov.to_frame('IBOV'), config.TEST_START,  config.TEST_END)['IBOV']

    # Train (coarse grid)
    best_params, best_score = grid_search(px_train)
    if best_params is None:
        best_params = config.DEFAULT_PARAMS

    # In-sample
    sig_tr = compute_signals(px_train, best_params)
    res_tr = run_backtest(px_train, sig_tr, best_params)

    # Out-of-sample
    sig_te = compute_signals(px_test, best_params)
    res_te = run_backtest(px_test, sig_te, best_params)

    # Export results
    pd.Series(best_params).to_csv(out / 'chosen_params.csv', header=False)
    (out / 'chosen_params.txt').write_text(str(best_params))

    res_tr['pv'].to_csv(out / 'pv_insample.csv')
    res_tr['trades'].to_csv(out / 'trades_insample.csv', index=False)
    pd.Series(res_tr['stats']).to_csv(out / 'stats_insample.csv', header=False)

    res_te['pv'].to_csv(out / 'pv_oos.csv')
    res_te['trades'].to_csv(out / 'trades_oos.csv', index=False)
    pd.Series(res_te['stats']).to_csv(out / 'stats_oos.csv', header=False)

    # Export benchmark returns for both periods
    ib_train.pct_change().fillna(0).to_csv(out / 'benchmark_rets_insample.csv')
    ib_test.pct_change().fillna(0).to_csv(out / 'benchmark_rets_oos.csv')

    print('Best Sharpe (train):', best_score)
    print('In-sample stats:', res_tr['stats'])
    print('Out-of-sample stats:', res_te['stats'])

if __name__ == '__main__':
    main()
