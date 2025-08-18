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
    """
    Limpeza de universo de preços:
      - remove colunas com missing_ratio > MAX_MISSING_RATIO
      - remove tickers com mediana de preço < MIN_PRICE_BRL
      - remove tickers com coverage < MIN_TRADED_DAYS_RATIO
      - remove tickers com qualquer dia de retorno absoluto > MAX_ABS_DAILY_RET_FOR_TICKER
      - remove tickers com fração de dias com retorno == 0 maior que MAX_ZERO_RETURNS (novo filtro de liquidez)
    """
    # 1) Missing ratio filter
    missing_ratio = prices.isna().mean()
    keep = missing_ratio[missing_ratio <= config.MAX_MISSING_RATIO].index
    prices = prices[keep]

    # 2) Median price filter
    med = prices.median(skipna=True)
    prices = prices.loc[:, med[med >= config.MIN_PRICE_BRL].index]

    # 3) Traded-days coverage filter
    traded = prices.notna().mean()
    prices = prices.loc[:, traded[traded >= config.MIN_TRADED_DAYS_RATIO].index]

    # 4) Remove tickers com spikes absurdos (|ret| > threshold)
    rets = prices.pct_change()
    bad = rets.columns[(rets.abs() > config.MAX_ABS_DAILY_RET_FOR_TICKER).any()]
    prices = prices.drop(columns=bad, errors='ignore')

    # 5) Novo: filtro de "zero-returns" indicando pouca liquidez
    # usa config.MAX_ZERO_RETURNS se existir, senão fallback 0.30
    max_zero = getattr(config, "MAX_ZERO_RETURNS", 0.05)

    # recalcula retornos (após as remoções anteriores)
    rets_after = prices.pct_change()

    # contagens por coluna
    zero_counts = (rets_after == 0).sum(axis=0)              # número de dias com retorno == 0
    valid_counts = rets_after.notna().sum(axis=0)           # número de dias com retorno definido

    # fração de dias "zero" — quando valid_counts == 0, consideramos fração = 1 (ilíquido)
    frac_zero = pd.Series(index=prices.columns, dtype=float)
    for col in prices.columns:
        vc = valid_counts.get(col, 0)
        if vc <= 0:
            frac_zero[col] = 1.0
        else:
            frac_zero[col] = zero_counts.get(col, 0) / vc

    illiquid_cols = frac_zero[frac_zero > max_zero].index.tolist()
    if illiquid_cols:
        print(f"[INFO] Removendo {len(illiquid_cols)} tickers por baixa liquidez (zero-returns > {max_zero:.2f}).")
        prices = prices.drop(columns=illiquid_cols, errors='ignore')

    return prices

def align_with_benchmark(prices: pd.DataFrame, ibov: pd.Series):
    idx = prices.index.intersection(ibov.index)
    return prices.loc[idx], ibov.loc[idx]