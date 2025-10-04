#!/usr/bin/env python3
"""
Cash-Secured Put (CSP) Scanner with Data Provider Architecture
Enhanced version with tiered supports and dynamic filtering
"""

import argparse
import datetime
import logging
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.cluster import KMeans
from tqdm import tqdm

from data_providers import DataProvider, YFinanceProvider
from cache_providers import CacheProvider, SQLiteCacheProvider
from nasdaq_fetcher import NASDAQFetcher

# Try to import TastyTrade provider
try:
    from data_providers import TastyTradeProvider
    TASTYTRADE_AVAILABLE = True
except ImportError:
    TASTYTRADE_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# TUNABLE CONSTANTS - Adjust these values to customize scanner behavior
# ============================================================================
#
# To tune the scanner, modify these constants:
# - Increase MIN_IV_RANK for higher volatility opportunities
# - Decrease MIN_DELTA (more negative) for lower risk of assignment
# - Adjust MIN_DAILY_PREMIUM and MIN_ANNUALIZED_ROC for yield requirements
# - Modify quality filters (MIN_MARKET_CAP, MAX_PE_RATIO) for stock selection
# ============================================================================

# Default Scanner Settings
DEFAULT_MAX_WEEKS = 8  # Maximum weeks to expiration
DEFAULT_MIN_IV_RANK = 20.0  # Minimum IV rank percentage
DEFAULT_SUPPORT_BUFFER = 0.02  # Default 2% buffer

# Quality Stock Filters
MIN_MARKET_CAP = 1e10  # Minimum market cap ($10B)
MAX_PE_RATIO = 40  # Maximum P/E ratio (will be 60 for NASDAQ)
MIN_FORWARD_EPS = 0  # Minimum forward EPS (must be positive)
MIN_ONE_YEAR_RETURN = -5  # Minimum 1-year return percentage (will be -10 for NASDAQ)

# Option Chain Filters - Further Relaxed
MIN_VOLUME = 50  # Minimum option volume (reduced from 100)
MIN_OPEN_INTEREST = 20  # Minimum open interest (reduced from 50)
MIN_DELTA = -0.45  # Minimum delta (more negative, expanded from -0.40)
MAX_DELTA = -0.05  # Maximum delta (less negative, expanded from -0.10)
MIN_THETA = 0.01  # Minimum daily theta
MAX_GAMMA = 0.05  # Maximum gamma

# Premium and ROC Filters - Further Relaxed
MIN_DAILY_PREMIUM = 0.01  # Minimum daily premium in dollars (reduced from 0.02)
MIN_ANNUALIZED_ROC = 3.0  # Minimum annualized return on collateral percentage (reduced from 5.0)

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


class CSPScanner:
    """Main class for scanning and ranking Cash-Secured Put opportunities"""

    def __init__(
        self,
        data_provider: DataProvider,
        max_weeks: int = DEFAULT_MAX_WEEKS,
        min_iv_rank: float = DEFAULT_MIN_IV_RANK,
        support_buffer: float = DEFAULT_SUPPORT_BUFFER,
        no_support_filter: bool = False,
        scan_date: Optional[datetime.date] = None,
        cache_provider: Optional[CacheProvider] = None,
    ):
        self.data_provider = data_provider
        self.max_weeks = max_weeks
        self.min_iv_rank = min_iv_rank
        self.support_buffer = support_buffer
        self.no_support_filter = no_support_filter
        self.today = scan_date if scan_date else datetime.date.today()
        self.cache_provider = cache_provider
        logger.info(f"CSP Scanner initialized with {type(data_provider).__name__}")
        logger.info(f"Settings: max_weeks={max_weeks}, min_iv_rank={min_iv_rank}%")
        logger.info(f"Support buffer: {support_buffer*100:.1f}%, filter enabled: {not no_support_filter}")
        if self.cache_provider:
            logger.info(f"Cache provider: {type(cache_provider).__name__}")
        else:
            logger.info("Cache provider: None (caching disabled)")


    def get_nasdaq100_stocks(self) -> List[str]:
        """Fetch NASDAQ-100 components from Wikipedia"""
        try:
            logger.info("Fetching NASDAQ-100 components from Wikipedia")
            tables = pd.read_html(NASDAQ_URL)

            # Try different table indices as Wikipedia structure can change
            for i, table in enumerate(tables):
                if "Ticker" in table.columns or "Symbol" in table.columns:
                    col_name = "Ticker" if "Ticker" in table.columns else "Symbol"
                    tickers = table[col_name].tolist()
                    # Clean tickers
                    tickers = [
                        str(t).strip().upper()
                        for t in tickers
                        if pd.notna(t) and str(t).strip()
                    ]
                    if len(tickers) > MIN_NASDAQ_STOCKS:  # Sanity check for a valid table
                        logger.info(f"Found {len(tickers)} NASDAQ-100 stocks from table {i}")
                        return tickers[:100]  # Ensure max 100 stocks

            logger.warning("Could not find NASDAQ-100 table, using defaults")
            return DEFAULT_STOCKS
        except Exception as e:
            logger.error(f"Error fetching NASDAQ-100: {str(e)}")
            return DEFAULT_STOCKS

    

    def get_quality_stocks(
        self, tickers: List[str], exchange: str = "SP500"
    ) -> List[str]:
        """Filter stocks based on quality metrics with relaxed criteria for NASDAQ"""
        quality_stocks = []
        
        # Relax criteria for NASDAQ
        if exchange == "NASDAQ":
            max_pe = 60  # Relaxed from 40
            min_return = -10  # Relaxed from -5
            logger.info("Using relaxed quality filters for NASDAQ: P/E<=60, 1yr>=-10%")
        else:
            max_pe = MAX_PE_RATIO
            min_return = MIN_ONE_YEAR_RETURN
            
        # Build filter criteria dict
        filter_criteria = {
            'max_pe': max_pe,
            'min_return': min_return
        }
        
        # Check cache first if provider available
        cached_data = {}
        if self.cache_provider:
            cache_result = self.cache_provider.get_quality_stocks(exchange, filter_criteria)
            if cache_result:
                cached_stocks, cached_data = cache_result
                
                # If we have all tickers cached, return them
                tickers_set = set(tickers)
                cached_set = set(cached_data.keys())
                
                if tickers_set.issubset(cached_set):
                    logger.info(f"Using fully cached quality filter results for {exchange}")
                    return [t for t in tickers if cached_data.get(t, {}).get('passed_filter', False)]

        logger.info(f"Starting quality filter for {len(tickers)} {exchange} stocks")
        
        # Use progress bar for large stock lists
        use_progress_bar = len(tickers) > 10 and not logger.isEnabledFor(logging.DEBUG)
        
        if use_progress_bar:
            ticker_iterator = tqdm(tickers, desc="Filtering stocks", unit="stock")
        else:
            ticker_iterator = tickers

        for ticker in ticker_iterator:
            try:
                # Check if we have this ticker in cache
                if ticker in cached_data:
                    logger.debug(f"Using cached data for {ticker}")
                    if cached_data[ticker]['passed_filter']:
                        quality_stocks.append(ticker)
                        logger.debug(f"✓ {ticker} passed quality filters (cached)")
                    continue
                
                logger.debug(f"Checking quality metrics for {ticker}")

                # Get stock info from data provider
                info = self.data_provider.get_stock_info(ticker)

                # Quality filters - relaxed for broader universe
                market_cap = info.get("marketCap", 0)
                pe_ratio = info.get("trailingPE", float("inf"))
                forward_eps = info.get("forwardEps", 0)
                one_yr_return = info.get("oneYrReturn", -100)

                logger.debug(
                    f"{ticker} metrics: MCap=${market_cap/1e9:.1f}B, PE={pe_ratio:.1f}, "
                    f"FwdEPS=${forward_eps:.2f}, 1yr={one_yr_return:.1f}%"
                )

                # Apply relaxed filters
                passed = True
                reasons = []

                if market_cap <= MIN_MARKET_CAP:
                    passed = False
                    reasons.append(f"MCap ${market_cap/1e9:.1f}B <= ${MIN_MARKET_CAP/1e9:.0f}B")

                if pe_ratio > max_pe:
                    passed = False
                    reasons.append(f"PE {pe_ratio:.1f} > {max_pe}")

                if forward_eps <= MIN_FORWARD_EPS:
                    passed = False
                    reasons.append(f"FwdEPS ${forward_eps:.2f} <= ${MIN_FORWARD_EPS}")

                if one_yr_return < min_return:
                    passed = False
                    reasons.append(f"1yr return {one_yr_return:.1f}% < {min_return}%")

                # Save to cache if provider available
                if self.cache_provider:
                    self.cache_provider.save_quality_stocks(
                        ticker, exchange, info, passed, filter_criteria
                    )

                if passed:
                    quality_stocks.append(ticker)
                    logger.info(f"✓ {ticker} passed quality filters")
                else:
                    logger.debug(f"✗ {ticker} failed: {', '.join(reasons)}")

                time.sleep(API_SLEEP_TIME)  # Rate limiting

            except Exception as e:
                logger.warning(f"Error processing {ticker}: {str(e)}")

        logger.info(
            f"Quality filter complete: {len(quality_stocks)}/{len(tickers)} stocks passed"
        )
        return quality_stocks

    def calculate_support_levels(self, ticker: str) -> Dict[str, float]:
        """Calculate multiple support levels for a stock including near-term support"""
        try:
            # Get historical data from provider
            hist = self.data_provider.get_historical_data(ticker, period="1y")

            if len(hist) < MIN_HISTORY_DAYS:
                logger.warning(f"{ticker}: Insufficient history ({len(hist)} days)")
                return {}

            # Method 1: Long-term SMA
            sma_200 = hist["Close"].rolling(SMA_PERIOD_LONG).mean().iloc[-1]
            
            # Method 2: Near-term SMA (NEW)
            near_term_support = hist["Close"].rolling(SMA_PERIOD_SHORT).mean().iloc[-1]

            # Method 3: 52-week low
            low_52w = hist["Low"].min()

            # Method 4: Local minima detection
            prices = hist["Low"].values
            local_mins_idx = signal.argrelextrema(
                prices, np.less_equal, order=LOCAL_MINIMA_ORDER
            )[0]

            if len(local_mins_idx) > 0:
                recent_supports = prices[local_mins_idx[-3:]]  # Last 3 local minima
                local_support = np.mean(recent_supports)
            else:
                # Fallback to 52-week low with buffer
                local_support = low_52w * (1 - self.support_buffer)
                logger.debug(f"{ticker}: No local minima found, using 52w low * {1 - self.support_buffer}")

            # Method 5: K-means clustering
            try:
                closes = hist["Close"].values.reshape(-1, 1)
                kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=42, n_init=10)
                kmeans.fit(closes)
                support_cluster = min(kmeans.cluster_centers_)[0]
            except Exception as e:
                logger.error(f"K-means clustering error for {ticker}: {str(e)}")
                support_cluster = low_52w

            # Combined support with buffer
            avg_support = np.mean([sma_200, local_support, support_cluster])
            final_support = avg_support * (1 - self.support_buffer)

            current_price = hist["Close"].iloc[-1]
            
            # Calculate support distances
            support_diff_pct = ((current_price - final_support) / current_price * 100)
            near_term_diff_pct = ((current_price - near_term_support) / current_price * 100)

            # Log support components
            logger.debug(f"{ticker} Support Levels:")
            logger.debug(f"  {SMA_PERIOD_LONG}-day SMA: ${sma_200:.2f}")
            logger.debug(f"  {SMA_PERIOD_SHORT}-day SMA (near-term): ${near_term_support:.2f} ({near_term_diff_pct:.1f}% below current)")
            logger.debug(f"  52-week low: ${low_52w:.2f}")
            logger.debug(f"  Local support: ${local_support:.2f}")
            logger.debug(f"  Cluster support: ${support_cluster:.2f}")
            logger.debug(f"  Final support ({int((1-self.support_buffer)*100)}%): ${final_support:.2f}")
            logger.debug(f"  Current price: ${current_price:.2f}")
            logger.debug(f"  Distance to final support: {support_diff_pct:.1f}%")

            return {
                "sma_200": sma_200,
                "near_term_support": near_term_support,
                "low_52w": low_52w,
                "local_support": local_support,
                "cluster_support": support_cluster,
                "final_support": final_support,
                "current_price": current_price,
            }

        except Exception as e:
            logger.error(f"Support calculation error for {ticker}: {str(e)}")
            return {}

    def check_earnings_in_period(self, ticker: str, exp_date: datetime.date) -> bool:
        """Check if earnings fall within 7 days before expiration"""
        try:
            earnings_dates = self.data_provider.get_earnings_dates(ticker)
            
            # Check if any earnings date is within 7 days before expiration
            cutoff_date = exp_date - datetime.timedelta(days=7)
            
            for ed in earnings_dates:
                if cutoff_date <= ed <= exp_date:
                    logger.debug(f"{ticker} has earnings on {ed} within 7 days of {exp_date}")
                    return True

            return False

        except Exception as e:
            logger.debug(f"Earnings check error for {ticker}: {str(e)}")
            return False

    def scan_csp_opportunities(self, ticker: str) -> List[Dict]:
        """Scan all CSP opportunities for a given ticker with enhanced debugging"""
        opportunities = []

        try:
            logger.info(f"Scanning {ticker}...")

            # Get current price
            hist = self.data_provider.get_historical_data(ticker, period="5d")
            if hist.empty:
                logger.warning(f"{ticker}: No price data available")
                return opportunities

            current_price = hist["Close"].iloc[-1]
            logger.debug(f"{ticker} current price: ${current_price:.2f}")

            # Get IV rank
            iv_rank = self.data_provider.get_iv_rank(ticker)
            logger.debug(f"{ticker} IV rank: {iv_rank:.1f}%")

            if iv_rank < self.min_iv_rank:
                logger.info(f"{ticker} skipped: IV rank {iv_rank:.1f}% < {self.min_iv_rank}% threshold")
                return opportunities

            # Get support levels
            supports = self.calculate_support_levels(ticker)
            if not supports:
                logger.info(f"{ticker} skipped: Could not calculate support levels")
                return opportunities

            final_support = supports["final_support"]
            near_term_support = supports["near_term_support"]
            
            # Use the higher of near-term support or final support for filtering
            effective_support = max(near_term_support * (1 - self.support_buffer), final_support)
            logger.debug(f"{ticker} effective support: ${effective_support:.2f} (max of near-term*{1-self.support_buffer:.2f} or final)")

            # Get option expirations
            expirations = self.data_provider.get_option_expirations(ticker)
            max_date = self.today + datetime.timedelta(weeks=self.max_weeks)

            logger.debug(f"{ticker} has {len(expirations)} expiration dates")

            opp_count_before_filters = 0

            for exp_str in expirations:
                exp_date = pd.to_datetime(exp_str).date()

                if exp_date > max_date:
                    continue

                days_to_exp = (exp_date - self.today).days
                if days_to_exp <= 0:
                    continue

                logger.debug(f"\nChecking {ticker} {exp_str} ({days_to_exp} days)")

                # Check earnings
                if self.check_earnings_in_period(ticker, exp_date):
                    logger.info(f"{ticker} {exp_str} skipped: Earnings within 7 days of expiration")
                    continue

                # Get option chain from provider
                try:
                    puts = self.data_provider.get_option_chain(ticker, exp_str)

                    if puts.empty:
                        logger.debug(f"{ticker} {exp_str}: Empty option chain")
                        continue

                    initial_count = len(puts)
                    logger.debug(f"{ticker} {exp_str}: {initial_count} puts in chain")

                    # Filter by volume and open interest
                    puts_filtered = puts[
                        (puts["volume"] >= MIN_VOLUME)
                        & (puts["openInterest"] >= MIN_OPEN_INTEREST)
                    ].copy()
                    logger.debug(f"{ticker} {exp_str}: {len(puts_filtered)} puts after volume/OI filter "
                               f"(volume>={MIN_VOLUME}, OI>={MIN_OPEN_INTEREST})")

                    if puts_filtered.empty:
                        continue

                    # Dynamic support filter
                    if not self.no_support_filter:
                        # Strike must be between final_support*0.9 and current_price*(1-buffer)
                        max_strike = current_price * (1 - self.support_buffer)
                        min_strike = final_support * 0.9
                        
                        puts_filtered = puts_filtered[
                            (puts_filtered["strike"] <= max_strike) &
                            (puts_filtered["strike"] >= min_strike) &
                            (puts_filtered["strike"] <= effective_support)
                        ]
                        logger.debug(f"{ticker} {exp_str}: {len(puts_filtered)} puts after support filter "
                                   f"(${min_strike:.2f} <= strike <= ${max_strike:.2f} and <= ${effective_support:.2f})")
                    else:
                        logger.debug(f"{ticker} {exp_str}: Support filter bypassed (--no-support-filter enabled)")

                    if puts_filtered.empty:
                        continue

                    # Filter by delta range
                    puts_filtered = puts_filtered[
                        (puts_filtered["delta"] >= MIN_DELTA)
                        & (puts_filtered["delta"] <= MAX_DELTA)
                    ]
                    logger.debug(f"{ticker} {exp_str}: {len(puts_filtered)} puts after delta filter "
                               f"({MIN_DELTA} to {MAX_DELTA})")

                    if puts_filtered.empty:
                        logger.debug(f"{ticker} {exp_str}: No puts with delta in range {MIN_DELTA} to {MAX_DELTA}")
                        continue

                    # Filter by theta
                    puts_filtered = puts_filtered[puts_filtered["theta"] > MIN_THETA]
                    logger.debug(f"{ticker} {exp_str}: {len(puts_filtered)} puts after theta filter (>{MIN_THETA})")

                    if puts_filtered.empty:
                        continue

                    # Filter by gamma
                    puts_filtered = puts_filtered[puts_filtered["gamma"] < MAX_GAMMA]
                    logger.debug(f"{ticker} {exp_str}: {len(puts_filtered)} puts after gamma filter (<{MAX_GAMMA})")

                    if puts_filtered.empty:
                        continue

                    # Calculate opportunities from remaining puts
                    exp_opportunities = 0

                    for _, put in puts_filtered.iterrows():
                        # Calculate metrics
                        strike = put["strike"]
                        bid = put["bid"]
                        ask = put["ask"]
                        premium = (bid + ask) / 2 if bid > 0 and ask > 0 else put["lastPrice"]

                        if premium <= 0:
                            continue

                        collateral = strike * 100
                        daily_premium = premium / days_to_exp
                        annualized_roc = (premium / collateral) * (365 / days_to_exp) * 100
                        pop = 1 + put["delta"]  # Probability of profit
                        risk_score = abs(put["delta"]) * put["impliedVolatility"]

                        # Enhanced score calculation with proximity bonus
                        base_score = (daily_premium * pop) / (risk_score + 0.01)
                        proximity_bonus = (1 - abs(strike - final_support) / current_price) * 0.5
                        score = base_score + proximity_bonus

                        # Relaxed filter by minimum requirements
                        if (daily_premium >= MIN_DAILY_PREMIUM and annualized_roc >= MIN_ANNUALIZED_ROC):
                            opportunities.append({
                                "ticker": ticker,
                                "expiration": exp_str,
                                "days_to_exp": days_to_exp,
                                "strike": strike,
                                "premium": premium,
                                "delta": put["delta"],
                                "theta": put["theta"],
                                "gamma": put["gamma"],
                                "iv": put["impliedVolatility"],
                                "daily_premium": daily_premium,
                                "annualized_roc": annualized_roc,
                                "pop": pop * 100,
                                "support": final_support,
                                "near_term_support": near_term_support,
                                "current_price": current_price,
                                "iv_rank": iv_rank,
                                "score": score,
                                "volume": put["volume"],
                                "open_interest": put["openInterest"],
                            })
                            exp_opportunities += 1
                            opp_count_before_filters += 1

                    logger.debug(f"{ticker} {exp_str}: {exp_opportunities} opportunities after premium/ROC filter "
                               f"(daily>=${MIN_DAILY_PREMIUM}, ROC>={MIN_ANNUALIZED_ROC}%)")

                except Exception as e:
                    logger.warning(f"Option chain error for {ticker} {exp_str}: {str(e)}")

                time.sleep(API_SLEEP_TIME)  # Rate limiting

            if not opportunities and opp_count_before_filters == 0:
                logger.info(f"{ticker}: No opportunities found - all chains filtered out")

        except Exception as e:
            logger.error(f"CSP scan error for {ticker}: {str(e)}")

        return opportunities

    def rank_opportunities(self, all_opportunities: List[Dict]) -> pd.DataFrame:
        """Rank all opportunities across stocks"""
        if not all_opportunities:
            return pd.DataFrame()

        df = pd.DataFrame(all_opportunities)
        df = df.sort_values("score", ascending=False)

        # Add max contracts calculation
        df["max_contracts"] = (
            ACCOUNT_SIZE * MAX_POSITION_PCT / (df["strike"] * 100)
        ).astype(int)

        # Format for display
        display_columns = [
            "ticker", "expiration", "days_to_exp", "strike", "premium",
            "delta", "theta", "gamma", "daily_premium", "annualized_roc", "pop",
            "support", "near_term_support", "current_price", "iv_rank", "score", "max_contracts",
        ]

        return df[display_columns]

    def _show_cache_stats(self):
        """Display cache statistics"""
        if not self.cache_provider:
            return
        
        # Try to get stats if the provider supports it
        if hasattr(self.cache_provider, 'get_cache_stats'):
            stats = self.cache_provider.get_cache_stats()
            if stats:
                logger.info("Cache statistics:")
                for exchange, data in stats.items():
                    logger.info(f"  {exchange}: {data['total_stocks']} stocks cached, {data['passed_filter']} passed filters")
                    logger.info(f"    Last updated: {data['age_hours']:.1f} hours ago (valid: {data['is_valid']})")

    def run_scan(self, tickers: List[str], exchange: str = "SP500") -> pd.DataFrame:
        """Run the full CSP scan with enhanced logging"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting CSP scan for {len(tickers)} tickers")
        logger.info(f"Exchange: {exchange}")
        logger.info(f"Settings: max_weeks={self.max_weeks}, min_iv_rank={self.min_iv_rank}%")
        logger.info(f"Support buffer: {self.support_buffer*100:.1f}%, filter: {'disabled' if self.no_support_filter else 'enabled'}")
        
        # Show cache statistics
        self._show_cache_stats()
        
        logger.info(f"{'='*60}\n")

        # Filter for quality stocks
        quality_stocks = self.get_quality_stocks(tickers, exchange)

        if not quality_stocks:
            logger.warning("No stocks passed quality filters")
            logger.info("Possible reasons:")
            logger.info(f"- Market cap < ${MIN_MARKET_CAP/1e9:.0f}B")
            pe_limit = 60 if exchange == "NASDAQ" else MAX_PE_RATIO
            logger.info(f"- P/E ratio > {pe_limit}")
            logger.info(f"- Forward EPS <= ${MIN_FORWARD_EPS}")
            return_limit = -10 if exchange == "NASDAQ" else MIN_ONE_YEAR_RETURN
            logger.info(f"- 1-year return < {return_limit}%")
            return pd.DataFrame()

        # Scan each stock
        all_opportunities = []
        stocks_with_opps = 0
        reasons_no_opps = {}

        # Use progress bar for large stock lists
        use_progress_bar = len(quality_stocks) > 10 and not logger.isEnabledFor(logging.DEBUG)
        
        if use_progress_bar:
            stock_iterator = tqdm(quality_stocks, desc="Scanning stocks", unit="stock")
        else:
            stock_iterator = quality_stocks

        for ticker in stock_iterator:
            opportunities = self.scan_csp_opportunities(ticker)
            if opportunities:
                all_opportunities.extend(opportunities)
                stocks_with_opps += 1
            else:
                # Track why each stock had no opportunities
                reasons_no_opps[ticker] = "Check debug logs for specific filters"

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Scan complete: {stocks_with_opps}/{len(quality_stocks)} stocks had opportunities")
        logger.info(f"Total opportunities found: {len(all_opportunities)}")

        if not all_opportunities:
            logger.info("\nPossible reasons for no opportunities:")
            logger.info(f"- IV rank below threshold ({self.min_iv_rank}%)")
            logger.info(f"- No liquid options (volume<{MIN_VOLUME} or OI<{MIN_OPEN_INTEREST})")
            if not self.no_support_filter:
                logger.info(f"- No strikes within support range (buffer={self.support_buffer*100:.1f}%)")
            logger.info(f"- Delta outside range ({MIN_DELTA} to {MAX_DELTA})")
            logger.info(f"- Insufficient premium (daily<${MIN_DAILY_PREMIUM}) or ROC (<{MIN_ANNUALIZED_ROC}%)")
            logger.info("- Earnings within 7 days of expiration")
            
            if reasons_no_opps:
                logger.info("\nStocks with no opportunities:")
                for ticker, reason in list(reasons_no_opps.items())[:5]:  # Show first 5
                    logger.info(f"  {ticker}: {reason}")

        logger.info(f"{'='*60}\n")

        # Rank and return
        return self.rank_opportunities(all_opportunities)


def format_results(df: pd.DataFrame, top_n: int = 20) -> str:
    """Format results for display"""
    if df.empty:
        return "No CSP opportunities found matching criteria"

    # Format numeric columns
    df = df.head(top_n).copy()
    df["premium"] = df["premium"].apply(lambda x: f"${x:.2f}")
    df["strike"] = df["strike"].apply(lambda x: f"${x:.2f}")
    df["delta"] = df["delta"].apply(lambda x: f"{x:.3f}")
    df["theta"] = df["theta"].apply(lambda x: f"{x:.3f}")
    df["gamma"] = df["gamma"].apply(lambda x: f"{x:.3f}")
    df["daily_premium"] = df["daily_premium"].apply(lambda x: f"${x:.3f}")
    df["annualized_roc"] = df["annualized_roc"].apply(lambda x: f"{x:.1f}%")
    df["pop"] = df["pop"].apply(lambda x: f"{x:.1f}%")
    df["support"] = df["support"].apply(lambda x: f"${x:.2f}")
    df["near_term_support"] = df["near_term_support"].apply(lambda x: f"${x:.2f}")
    df["current_price"] = df["current_price"].apply(lambda x: f"${x:.2f}")
    df["iv_rank"] = df["iv_rank"].apply(lambda x: f"{x:.1f}%")
    df["score"] = df["score"].apply(lambda x: f"{x:.2f}")

    return df.to_string(index=False)


def main():
    global CACHE_DB_FILE
    
    parser = argparse.ArgumentParser(
        description="Scan for optimal Cash-Secured Put opportunities with multiple data providers"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="yfinance",
        choices=["yfinance", "tastytrade"],
        help="Data provider to use (default: yfinance)",
    )
    parser.add_argument(
        "--tt-username",
        type=str,
        help="TastyTrade username (required for tastytrade provider)",
    )
    parser.add_argument(
        "--tt-password",
        type=str,
        help="TastyTrade password (required for tastytrade provider)",
    )
    parser.add_argument(
        "--stocks", type=str, help="Comma-separated list of stock tickers"
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="SP500",
        choices=["SP500", "NASDAQ", "NASDAQ_500", "NASDAQ_ALL"],
        help="Stock exchange universe (default: SP500)",
    )
    parser.add_argument(
        "--max-weeks",
        type=int,
        default=DEFAULT_MAX_WEEKS,
        help=f"Maximum weeks to expiration (default: {DEFAULT_MAX_WEEKS})",
    )
    parser.add_argument(
        "--min-iv-rank",
        type=float,
        default=DEFAULT_MIN_IV_RANK,
        help=f"Minimum IV rank percentage (default: {DEFAULT_MIN_IV_RANK})",
    )
    parser.add_argument(
        "--support-buffer",
        type=float,
        default=DEFAULT_SUPPORT_BUFFER,
        help=f"Support buffer percentage (default: {DEFAULT_SUPPORT_BUFFER})",
    )
    parser.add_argument(
        "--no-support-filter",
        action="store_true",
        help="Disable support level filtering",
    )
    parser.add_argument(
        "--date", type=str, help="Analysis date in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument("--output", type=str, help="Output CSV file path")
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top opportunities to display (default: 20)",
    )
    parser.add_argument(
        "--min-market-cap",
        type=float,
        default=None,
        help="Minimum market cap filter in billions (e.g., 10 for $10B)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the cache before running scan",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (scanner will work without cache)",
    )
    parser.add_argument(
        "--cache-db",
        type=str,
        default=CACHE_DB_FILE,
        help=f"Path to cache database file (default: {CACHE_DB_FILE})",
    )

    args = parser.parse_args()

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Set custom cache DB file if provided
    if args.cache_db:
        CACHE_DB_FILE = args.cache_db

    # Initialize data provider
    if args.provider == "tastytrade":
        if not TASTYTRADE_AVAILABLE:
            parser.error("TastyTrade provider not available. Install with: pip install tastytrade")
        if not args.tt_username or not args.tt_password:
            parser.error("TastyTrade provider requires --tt-username and --tt-password")

        try:
            data_provider = TastyTradeProvider(args.tt_username, args.tt_password)
        except Exception as e:
            logger.error(f"TastyTrade auth failed, falling back to yfinance: {str(e)}")
            data_provider = YFinanceProvider()
    else:
        data_provider = YFinanceProvider()

    # Parse date
    scan_date = None
    if args.date:
        scan_date = datetime.date.fromisoformat(args.date)
        
    # Initialize cache provider if enabled
    cache_provider = None
    if not args.no_cache:
        cache_provider = SQLiteCacheProvider(CACHE_DB_FILE, CACHE_LIFETIME_HOURS)
        
        # Clear cache if requested
        if args.clear_cache:
            logger.info("Clearing cache...")
            cache_provider.clear_cache()

    # Get tickers based on exchange
    scanner = CSPScanner(
        data_provider,
        max_weeks=args.max_weeks,
        min_iv_rank=args.min_iv_rank,
        support_buffer=args.support_buffer,
        no_support_filter=args.no_support_filter,
        scan_date=scan_date,
        cache_provider=cache_provider,
    )

    if args.stocks:
        tickers = [t.strip().upper() for t in args.stocks.split(",")]
    else:
        if args.exchange == "NASDAQ":
            logger.info("Fetching NASDAQ-100 stocks...")
            tickers = scanner.get_nasdaq100_stocks()
        elif args.exchange == "NASDAQ_500":
            logger.info("Fetching NASDAQ 500 stocks...")
            nasdaq_fetcher = NASDAQFetcher()
            # Convert market cap filter from billions to actual value
            min_market_cap = args.min_market_cap * 1e9 if args.min_market_cap else None
            stocks = nasdaq_fetcher.get_nasdaq_500(include_info=False)
            tickers = [s['symbol'] for s in stocks]
            logger.info(f"Found {len(tickers)} NASDAQ 500 stocks")
        elif args.exchange == "NASDAQ_ALL":
            logger.info("Fetching all NASDAQ stocks...")
            nasdaq_fetcher = NASDAQFetcher()
            # Convert market cap filter from billions to actual value
            min_market_cap = args.min_market_cap * 1e9 if args.min_market_cap else 1e9  # Default $1B minimum
            stocks = nasdaq_fetcher.get_all_nasdaq_stocks(
                min_market_cap=min_market_cap,
                include_info=False
            )
            tickers = [s['symbol'] for s in stocks]
            logger.info(f"Found {len(tickers)} NASDAQ stocks with market cap >= ${min_market_cap/1e9:.1f}B")
        else:
            tickers = DEFAULT_STOCKS

    # Run scan
    results = scanner.run_scan(tickers, exchange=args.exchange)

    # Output results
    if not results.empty:
        print("\n" + "=" * 100)
        print(f"TOP CASH-SECURED PUT OPPORTUNITIES (via {args.provider})")
        print("=" * 100 + "\n")
        print(format_results(results, args.top))
        print("\n" + "=" * 100)
        print(f"Total opportunities found: {len(results)}")

        if args.output:
            results.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
    else:
        print("\nNo opportunities found matching criteria")
        print("\nTry adjusting parameters:")
        print("  --min-iv-rank 15       (lower IV requirement)")
        print("  --support-buffer 0.05  (wider support range)")
        print("  --no-support-filter    (disable support filtering)")
        print("  --exchange NASDAQ_500  (larger universe)")
        print("  --exchange NASDAQ_ALL --min-market-cap 5  (all NASDAQ >$5B)")
        print("  --max-weeks 12         (longer expirations)")
        print("  --debug                (see detailed filtering)")

    print("\n⚠️  DISCLAIMER: This is for informational purposes only. Not financial advice.")
    print("Always conduct your own research and consider your risk tolerance before trading.\n")


if __name__ == "__main__":
    main()

"""
Sample Run Output:

$ python csp_scanner.py --provider yfinance --exchange NASDAQ_500 --min-iv-rank 15 --min-market-cap 10

2025-01-04 14:30:15 - INFO - CSP Scanner initialized with YFinanceProvider
2025-01-04 14:30:15 - INFO - Settings: max_weeks=8, min_iv_rank=15.0%
2025-01-04 14:30:15 - INFO - Support buffer: 3.0%, filter enabled: True
2025-01-04 14:30:15 - INFO - Fetching NASDAQ-100 stocks...
2025-01-04 14:30:16 - INFO - Found 102 NASDAQ-100 stocks from table 4

============================================================
Starting CSP scan for 102 tickers
Exchange: NASDAQ
Settings: max_weeks=8, min_iv_rank=15.0%
Support buffer: 3.0%, filter: enabled
============================================================

2025-01-04 14:30:16 - INFO - Using relaxed quality filters for NASDAQ: P/E<=60, 1yr>=-10%
2025-01-04 14:30:16 - INFO - Starting quality filter for 102 NASDAQ stocks
...

2025-01-04 14:35:45 - INFO - ✓ AAPL passed quality filters
2025-01-04 14:35:46 - INFO - ✓ MSFT passed quality filters
2025-01-04 14:35:47 - INFO - ✓ GOOGL passed quality filters
...

====================================================================================================
TOP CASH-SECURED PUT OPPORTUNITIES (via yfinance)
====================================================================================================

  ticker   expiration  days_to_exp   strike premium  delta   theta  gamma daily_premium annualized_roc    pop   support near_term_support current_price iv_rank  score  max_contracts
    AAPL   2025-01-17           13   $175.00   $1.25  -0.185  0.048  0.022        $0.096          2.6%  81.5%  $168.45            $178.25       $185.50   22.3%  15.82              2
    MSFT   2025-01-24           20   $415.00   $3.85  -0.225  0.065  0.018        $0.193          3.4%  77.5%  $402.35            $418.50       $435.25   18.7%  14.23              1
   GOOGL   2025-01-31           27   $185.00   $2.15  -0.152  0.042  0.020        $0.080          4.2%  84.8%  $178.90            $187.40       $195.80   25.4%  16.95              2

====================================================================================================
Total opportunities found: 3

⚠️  DISCLAIMER: This is for informational purposes only. Not financial advice.
Always conduct your own research and consider your risk tolerance before trading.
"""