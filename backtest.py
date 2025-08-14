import pandas as pd
import numpy as np
import math
import config
from utils import monthly_rebalance_dates, stats_from_pv

def run_backtest(prices: pd.DataFrame, signals: dict, params: dict) -> dict:
    dates = prices.index
    rebalance_dates = monthly_rebalance_dates(dates, config.REB_FREQ)

    score = signals['score']
    pv = pd.Series(index=dates, dtype=float)
    cash = config.INITIAL_CASH
    positions = {}
    trades = []

    def close_position(dt, tkr, price, reason):
        nonlocal cash
        pos = positions.pop(tkr)
        proceeds = pos['shares'] * price * (1 - config.SLIPPAGE_BPS/10000.0)
        cash += proceeds
        trades.append({'date': dt, 'ticker': tkr, 'side': 'SELL', 'price': price, 'shares': pos['shares'], 'reason': reason})

    for dt in dates:
        px_today = prices.loc[dt]

        # Stops
        to_close = []
        for tkr, pos in list(positions.items()):
            p = px_today.get(tkr, np.nan)
            if np.isnan(p):
                continue
            pos['running_peak'] = max(pos['running_peak'], p)
            dd_from_peak = (pos['running_peak'] - p) / pos['running_peak'] if pos['running_peak']>0 else 0.0
            loss_from_entry = (pos['entry_price'] - p) / pos['entry_price'] if pos['entry_price']>0 else 0.0
            if dd_from_peak >= params['trailing_stop']:
                to_close.append((tkr, p, 'TRAIL_STOP'))
            elif loss_from_entry >= params['fixed_stop_loss']:
                to_close.append((tkr, p, 'STOP_LOSS'))
        for tkr, px_, why in to_close:
            if tkr in positions:
                close_position(dt, tkr, px_, why)

        # Rebalance
        if dt in rebalance_dates:
            daily_scores = score.loc[dt].dropna().sort_values(ascending=False)
            picks = list(daily_scores.head(params['top_n']).index)

            for tkr in list(positions.keys()):
                if tkr not in picks:
                    p = px_today.get(tkr, np.nan)
                    if not np.isnan(p):
                        close_position(dt, tkr, p, 'REBAL_DROP')

            num_new = len([t for t in picks if t not in positions])
            alloc_per_pos = cash / num_new if num_new > 0 else 0.0
            for tkr in picks:
                if tkr in positions:
                    continue
                p = px_today.get(tkr, np.nan)
                if np.isnan(p) or p <= 0:
                    continue
                shares = math.floor(alloc_per_pos / (p * (1 + config.SLIPPAGE_BPS/10000.0)))
                if shares <= 0:
                    continue
                cost = shares * p * (1 + config.SLIPPAGE_BPS/10000.0)
                cash -= cost
                positions[tkr] = {'entry_date': dt, 'entry_price': p, 'shares': shares, 'running_peak': p}
                trades.append({'date': dt, 'ticker': tkr, 'side': 'BUY', 'price': p, 'shares': shares, 'reason': 'REBAL_ADD'})

        # Mark-to-market
        m2m = 0.0
        for tkr, pos in positions.items():
            p = px_today.get(tkr, np.nan)
            if not np.isnan(p):
                m2m += pos['shares'] * p
        pv.loc[dt] = cash + m2m

    # Liquidate at end
    last_dt = dates[-1]
    px_last = prices.loc[last_dt]
    for tkr in list(positions.keys()):
        p = px_last.get(tkr, np.nan)
        if not np.isnan(p):
            close_position(last_dt, tkr, p, 'FINAL')

    stats = stats_from_pv(pv.dropna())
    return {'pv': pv.dropna(), 'trades': pd.DataFrame(trades), 'stats': stats}
