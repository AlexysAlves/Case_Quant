from pathlib import Path

DATA_DIR = Path('/mnt/data')

# Input data (already uploaded by you)
PRICES_CSV = DATA_DIR / 'precos_b3_202010-2024.csv'
ADJCLOSE_CSV = DATA_DIR / 'precos_b3_202010-2024_adjclose.csv'
IBOV_CSV = DATA_DIR / 'ibov_2010_2024.csv'
USE_ADJCLOSE = True  # set False to use raw close

# Universe cleaning
MIN_PRICE_BRL = 2.0
MAX_ABS_DAILY_RET_FOR_TICKER = 0.50   # drop tickers that ever exceed this abs return in a day
MAX_MISSING_RATIO = 0.30              # drop tickers with >30% NAs
MIN_TRADED_DAYS_RATIO = 0.85          # keep tickers with at least 85% trading days filled

# Train / Test split (as asked)
TRAIN_START = '2010-01-01'
TRAIN_END   = '2012-12-31'
TEST_START  = '2013-01-02'
TEST_END    = '2024-12-31'

# Strategy
REB_FREQ = 'M'        # monthly rebalance
INITIAL_CASH = 1_000_000.0
SLIPPAGE_BPS = 0.0    # costs ignored per case
TOP_N = 20
FIXED_STOP_LOSS = 0.10   # 10%
TRAILING_STOP = 0.15     # 15%

# Genetic Algorithm
GA_SEED = 42
GA_POP_SIZE = 16
GA_GENERATIONS = 20
GA_CROSSOVER_RATE = 0.8
GA_MUTATION_RATE = 0.15
GA_ELITISM = 2

OUT_DIR = Path(str(Path(__file__).parent / 'outputs'))
OUT_DIR.mkdir(parents=True, exist_ok=True)