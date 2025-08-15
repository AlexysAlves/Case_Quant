import pandas as pd
import numpy as np
import math
import config

def monthly_rebalance_dates(index, freq='M'):
    return pd.date_range(index.min(), index.max(), freq=freq).intersection(index)

def run_backtest(prices: pd.DataFrame, score: pd.DataFrame, top_n: int) -> dict:
    dates = prices.index
    rebalance_dates = monthly_rebalance_dates(dates, config.REB_FREQ)

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

        # Evaluate stops
        to_close = []
        for tkr, pos in list(positions.items()):
            p = px_today.get(tkr, np.nan)
            if np.isnan(p):
                continue
            pos['running_peak'] = max(pos['running_peak'], p)
            dd_from_peak = (pos['running_peak'] - p) / pos['running_peak'] if pos['running_peak']>0 else 0.0
            loss_from_entry = (pos['entry_price'] - p) / pos['entry_price'] if pos['entry_price']>0 else 0.0
            if dd_from_peak >= config.TRAILING_STOP:
                to_close.append((tkr, p, 'TRAIL_STOP'))
            elif loss_from_entry >= config.FIXED_STOP_LOSS:
                to_close.append((tkr, p, 'STOP_LOSS'))
        for tkr, p, why in to_close:
            if tkr in positions:
                close_position(dt, tkr, p, why)

        # Rebalance monthly
        if dt in rebalance_dates:
            daily_scores = score.loc[dt].dropna().sort_values(ascending=False)
            picks = list(daily_scores.head(top_n).index)

            # sell those that fell out
            for tkr in list(positions.keys()):
                if tkr not in picks:
                    p = px_today.get(tkr, np.nan)
                    if not np.isnan(p):
                        close_position(dt, tkr, p, 'REBAL_DROP')

            # equal allocation of available cash to new names
            new_names = [t for t in picks if t not in positions]
            alloc = cash / len(new_names) if new_names else 0.0
            for tkr in new_names:
                p = px_today.get(tkr, np.nan)
                if np.isnan(p) or p <= 0: 
                    continue
                shares = math.floor(alloc / (p * (1 + config.SLIPPAGE_BPS/10000.0)))
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

    return {'pv': pv.dropna(), 'trades': pd.DataFrame(trades)}