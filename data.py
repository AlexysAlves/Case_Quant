import pandas as pd
import numpy as np
import config

def load_prices() -> pd.DataFrame:
    path = config.ADJCLOSE_CSV if config.USE_ADJCLOSE else config.PRICES_CSV
    df = pd.read_csv(path, parse_dates=['Date']).sort_values('Date').set_index('Date')
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def load_ibov() -> pd.Series:
    df = pd.read_csv(config.IBOV_CSV, parse_dates=['Date']).sort_values('Date')
    cols = [c for c in df.columns if c.lower() != 'date']
    if not cols:
        raise ValueError('IBOV CSV must have a price column besides Date.')
    s = df.set_index('Date')[cols[0]]
    if s.dtype == object:
        s = (
            s.astype(str)
             .str.replace('\u00a0', '', regex=False)
             .str.replace(' ', '', regex=False)
             .str.replace('.', '', regex=False)
             .str.replace(',', '.', regex=False)
        )
    s = pd.to_numeric(s, errors='coerce')
    return s

def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    missing_ratio = prices.isna().mean()
    keep = missing_ratio[missing_ratio <= config.MAX_MISSING_RATIO].index
    prices = prices[keep]

    med = prices.median(skipna=True)
    prices = prices.loc[:, med[med >= config.MIN_PRICE_BRL].index]

    traded = prices.notna().mean()
    prices = prices.loc[:, traded[traded >= config.MIN_TRADED_DAYS_RATIO].index]

    rets = prices.pct_change()
    bad = rets.columns[(rets.abs() > config.MAX_ABS_DAILY_RET_FOR_TICKER).any()]
    prices = prices.drop(columns=bad, errors='ignore')
    return prices

def align_with_benchmark(prices: pd.DataFrame, ibov: pd.Series):
    idx = prices.index.intersection(ibov.index)
    return prices.loc[idx], ibov.loc[idx]