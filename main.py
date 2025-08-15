import json
import pandas as pd
import config, data
from indicators import compute_indicators, score_from_weights
from genetic_algorithm import optimize_weights
from backtest import run_backtest
from utils import stats_from_pv
from reporting import export_stats_and_plots

def main():
    out = config.OUT_DIR

    # Load & clean
    prices = data.load_prices()
    prices = data.clean_prices(prices)
    ibov = data.load_ibov()

    # Align
    prices, ibov = data.align_with_benchmark(prices, ibov)

    # Split
    tr = prices.loc[config.TRAIN_START:config.TRAIN_END]
    te = prices.loc[config.TEST_START:config.TEST_END]
    ibov_tr = ibov.loc[tr.index.min():tr.index.max()]
    ibov_te = ibov.loc[te.index.min():te.index.max()]

    # GA optimize on training
    best_w, best_fit = optimize_weights(
        tr,
        top_n=config.TOP_N,
        seed=config.GA_SEED,
        pop_size=config.GA_POP_SIZE,
        generations=config.GA_GENERATIONS,
        crossover_rate=config.GA_CROSSOVER_RATE,
        mutation_rate=config.GA_MUTATION_RATE,
        elitism=config.GA_ELITISM
    )
    (out / 'chosen_weights.json').write_text(json.dumps(best_w, indent=2, ensure_ascii=False))
    with open(out / 'ga_log.txt', 'w') as f:
        f.write(f'Best training CAGR: {best_fit}\n')

    # Backtest train
    sc_tr = score_from_weights(compute_indicators(tr), best_w)
    res_tr = run_backtest(tr, sc_tr, config.TOP_N)
    pv_tr, trades_tr = res_tr['pv'], res_tr['trades']
    pv_tr.to_csv(out / 'pv_train.csv')
    trades_tr.to_csv(out / 'trades_train.csv', index=False)

    # Backtest test
    sc_te = score_from_weights(compute_indicators(te), best_w)
    res_te = run_backtest(te, sc_te, config.TOP_N)
    pv_te, trades_te = res_te['pv'], res_te['trades']
    pv_te.to_csv(out / 'pv_test.csv')
    trades_te.to_csv(out / 'trades_test.csv', index=False)

    # Benchmark PV aligned (normalize to same initial capital)
    ibov_tr_pv = (ibov_tr / ibov_tr.iloc[0]) * config.INITIAL_CASH
    ibov_te_pv = (ibov_te / ibov_te.iloc[0]) * config.INITIAL_CASH

    # Export stats + plots for train and test
    export_stats_and_plots(pv_tr, ibov_tr_pv, trades_tr, out)
    export_stats_and_plots(pv_te, ibov_te_pv, trades_te, out)

    # Save combined stats CSVs
    import pandas as pd
    from utils import stats_from_pv
    train_stats = stats_from_pv(pv_tr, len(trades_tr))
    test_stats  = stats_from_pv(pv_te, len(trades_te))
    pd.Series(train_stats).to_csv(out / 'strategy_stats_train.csv', header=False)
    pd.Series(test_stats).to_csv(out / 'strategy_stats_test.csv', header=False)

    print('Done. Outputs in:', out)

if __name__ == '__main__':
    main()