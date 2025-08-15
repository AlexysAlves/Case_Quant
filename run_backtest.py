import pandas as pd
import json
import config
import data
from signals import compute_indicators, score_from_weights, INDICATOR_NAMES
from backtest import run_backtest
from ga_optimizer import genetic_optimize_weights

def main():
    out = config.OUT_DIR
    prices = data.load_prices()
    prices = data.clean_prices(prices)
    ibov = data.load_ibov()
    prices, ibov = data.align_with_benchmark(prices, ibov)

    px_train = data.split_by_dates(prices, config.TRAIN_START, config.TRAIN_END)
    px_test  = data.split_by_dates(prices, config.TEST_START,  config.TEST_END)

    best_weights, best_fit = genetic_optimize_weights(
        px_train,
        params=config.DEFAULT_PARAMS,
        seed=42,
        pop_size=8,
        generations=5,
        crossover_rate=0.8,
        mutation_rate=0.15,
        elitism=2
    )
    (out / 'chosen_weights.json').write_text(json.dumps(best_weights, indent=2, ensure_ascii=False))
    pd.Series(best_weights).to_csv(out / 'chosen_weights.csv', header=False)
    with open(out / 'ga_train_log.txt', 'w') as f:
        f.write(f'Best training fitness (CAGR): {best_fit}\n')

    sc_tr = score_from_weights(compute_indicators(px_train), best_weights)
    res_tr = run_backtest(px_train, {'score': sc_tr}, config.DEFAULT_PARAMS)

    sc_te = score_from_weights(compute_indicators(px_test), best_weights)
    res_te = run_backtest(px_test, {'score': sc_te}, config.DEFAULT_PARAMS)

    pd.Series(res_tr['stats']).to_csv(out / 'stats_insample.csv', header=False)
    pd.Series(res_te['stats']).to_csv(out / 'stats_oos.csv', header=False)
    res_tr['pv'].to_csv(out / 'pv_insample.csv')
    res_tr['trades'].to_csv(out / 'trades_insample.csv', index=False)
    res_te['pv'].to_csv(out / 'pv_oos.csv')
    res_te['trades'].to_csv(out / 'trades_oos.csv', index=False)

    print('GA best training CAGR:', best_fit)
    print('Weights:', best_weights)
    print('In-sample stats:', res_tr['stats'])
    print('Out-of-sample stats:', res_te['stats'])

if __name__ == '__main__':
    main()