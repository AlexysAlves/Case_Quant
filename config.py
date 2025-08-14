from pathlib import Path

# Paths (use your uploaded files)
DATA_DIR = Path('')

PRICES_CSV = DATA_DIR / 'precos_b3_202010-2024.csv'                 # raw close
ADJCLOSE_CSV = DATA_DIR / 'precos_b3_202010-2024_adjclose.csv'      # adjusted close
IBOV_CSV = DATA_DIR / 'ibov_2010_2024.csv'                          # IBOV benchmark

USE_ADJCLOSE = True  # prefer adjusted close if available

# Cleaning / liquidity heuristics
MIN_PRICE_BRL = 2.0
MAX_ABS_DAILY_RET_FOR_TICKER = 0.50    # if any |ret| > 50% -> drop ticker
MAX_MISSING_RATIO = 0.30               # drop tickers with >30% NaN
MIN_TRADED_DAYS_RATIO = 0.85           # require >=85% non-null days

# Train/Test split (no gap). Keep train at most until 2012/2013 per user request
TRAIN_START = '2010-01-01'
TRAIN_END   = '2012-12-31'   # you can set '2013-12-31' if preferred
TEST_START  = '2013-01-02'
TEST_END    = '2024-12-31'

# Backtest core
REB_FREQ = 'M'               # monthly rebalance
INITIAL_CASH = 1_000_000.0
SLIPPAGE_BPS = 0.0

# Default strategy params (used if grid search fails)
DEFAULT_PARAMS = {
    'lookback_mom_days': 252,       # 12m
    'skip_last_days_for_mom': 21,   # 12-1 momentum
    'donchian_high_days': 100,
    'top_n': 20,
    'fixed_stop_loss': 0.10,
    'trailing_stop': 0.15,
    'weights': (0.6, 0.3, 0.1)      # (mom, prox_52w, breakout)
}

# Coarse grid to avoid overfitting
GRID = {
    'lookback_mom_days': [252, 126],
    'skip_last_days_for_mom': [21],
    'donchian_high_days': [55, 100],
    'top_n': [15, 20],
    'fixed_stop_loss': [0.08, 0.10],
    'trailing_stop': [0.12, 0.15],
    'weights': [(0.6,0.3,0.1), (0.5,0.3,0.2)],
}

# Outputs
OUT_DIR = Path('/outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)
