"""
Constants module for CSP Scanner

This module contains all tunable constants for the scanner.
To tune the scanner, modify these constants:
- Increase MIN_IV_RANK for higher volatility opportunities
- Decrease MIN_DELTA (more negative) for lower risk of assignment
- Adjust MIN_DAILY_PREMIUM and MIN_ANNUALIZED_ROC for yield requirements
- Modify quality filters (MIN_MARKET_CAP, MAX_PE_RATIO) for stock selection
"""

# Default Scanner Settings
DEFAULT_MAX_WEEKS = 8  # Maximum weeks to expiration
DEFAULT_MIN_IV_RANK = 20.0  # Minimum IV rank percentage
DEFAULT_SUPPORT_BUFFER = 0.02  # Default 2% buffer

# Quality Stock Filters
MIN_MARKET_CAP = 1e10  # Minimum market cap ($10B)
MAX_PE_RATIO = 40  # Maximum P/E ratio (will be 60 for NASDAQ)
MIN_FORWARD_EPS = 0  # Minimum forward EPS (must be positive)
MIN_ONE_YEAR_RETURN = -5  # Minimum 1-year return percentage (will be -10 for NASDAQ)

# NASDAQ-specific relaxed filters
NASDAQ_MAX_PE_RATIO = 60  # Relaxed P/E for NASDAQ
NASDAQ_MIN_ONE_YEAR_RETURN = -10  # Relaxed return for NASDAQ

# Option Chain Filters
MIN_VOLUME = 50  # Minimum option volume
MIN_OPEN_INTEREST = 20  # Minimum open interest
MIN_DELTA = -0.45  # Minimum delta (more negative)
MAX_DELTA = -0.05  # Maximum delta (less negative)
MIN_THETA = 0.01  # Minimum daily theta
MAX_GAMMA = 0.05  # Maximum gamma

# Premium and ROC Filters
MIN_DAILY_PREMIUM = 0.01  # Minimum daily premium in dollars
MIN_ANNUALIZED_ROC = 3.0  # Minimum annualized return on collateral percentage

# Support Level Calculations
LOCAL_MINIMA_ORDER = 20  # Window size for local minima detection
KMEANS_CLUSTERS = 3  # Number of clusters for K-means support
SMA_PERIOD_LONG = 200  # Long-term simple moving average period
SMA_PERIOD_SHORT = 50  # Short-term simple moving average period for near-term support
MIN_HISTORY_DAYS = 200  # Minimum days of history required

# Position Sizing
ACCOUNT_SIZE = 100000  # Account size for max contracts calculation
MAX_POSITION_PCT = 0.05  # Maximum position size as percentage of account

# Rate Limiting
API_SLEEP_TIME = 0.3  # Sleep time between API calls (seconds)

# Data Source Settings
NASDAQ_URL = "https://en.wikipedia.org/wiki/NASDAQ-100"
MIN_NASDAQ_STOCKS = 50  # Minimum stocks to validate NASDAQ table

# Default quality stocks (S&P 500 large-cap leaders)
DEFAULT_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM",
    "V", "JNJ", "WMT", "PG", "MA", "HD", "DIS", "BAC", "XOM", "CVX",
    "ABBV", "PFE", "TMO", "CSCO", "ACN", "MRK", "ABT", "NKE", "LLY",
    "ORCL", "TXN", "CRM", "MCD", "QCOM", "NEE", "COST", "BMY", "HON",
]

# Cache Settings
CACHE_DB_FILE = "csp_scanner_cache.db"
CACHE_LIFETIME_HOURS = 24