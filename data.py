
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
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
    price_col = cols[0]
    s = df.set_index('Date')[price_col]
    # Try robust numeric parsing: remove thousand separators and normalize decimal comma
    if s.dtype == object:
        s = (
            s.astype(str)
             .str.replace('\u00a0', '', regex=False)  # non-breaking space
             .str.replace(' ', '', regex=False)
             .str.replace('.', '', regex=False)        # remove thousands dot
             .str.replace(',', '.', regex=False)       # decimal comma -> dot
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

def split_by_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]

def align_with_benchmark(prices: pd.DataFrame, ibov: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    idx = prices.index.intersection(ibov.index)
    return prices.loc[idx], ibov.loc[idx]
