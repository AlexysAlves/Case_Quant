import pandas as pd
import matplotlib.pyplot as plt
from utils import stats_from_pv, drawdown_series

def export_stats_and_plots(strategy_pv, ibov_pv, trades_df, out_dir):
    # Stats
    strat_stats = stats_from_pv(strategy_pv, len(trades_df))
    bench_trades = 0
    bench_stats = stats_from_pv(ibov_pv, bench_trades)
    pd.Series(strat_stats).to_csv(out_dir / 'strategy_stats.csv', header=False)
    pd.Series(bench_stats).to_csv(out_dir / 'ibov_stats.csv', header=False)

    # Curves chart
    plt.figure()
    (strategy_pv / strategy_pv.iloc[0]).plot()
    (ibov_pv / ibov_pv.iloc[0]).plot()
    plt.title('Evolução — Estratégia vs IBOV')
    plt.xlabel('Data'); plt.ylabel('Índice (base=1)')
    plt.legend(['Estratégia', 'IBOV'])
    plt.tight_layout()
    plt.savefig(out_dir / 'curve_strategy_vs_ibov.png')
    plt.close()

    # Drawdown chart (strategy)
    dd = drawdown_series(strategy_pv)
    plt.figure()
    dd.plot()
    plt.title('Drawdown — Estratégia')
    plt.xlabel('Data'); plt.ylabel('Drawdown')
    plt.tight_layout()
    plt.savefig(out_dir / 'drawdown_strategy.png')
    plt.close()