#!/usr/bin/env python3
"""
Cash-Secured Put (CSP) Scanner with Modular Architecture

Consolidated version that combines:
- Modular architecture from refactored version
- NASDAQ fetcher integration and market cap filtering from original
- All CLI arguments and features from both versions
"""

import argparse
import datetime
import logging
import time
import warnings
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from data_providers import DataProvider, YFinanceProvider
from cache_providers import CacheProvider, SQLiteCacheProvider
from nasdaq_fetcher import NASDAQFetcher
from modules import StockFilter, SupportCalculator, OptionAnalyzer, ScoringEngine
from modules import constants

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


class CSPScanner:
    """Main class for scanning and ranking Cash-Secured Put opportunities"""

    def __init__(
        self,
        data_provider: DataProvider,
        max_weeks: int = constants.DEFAULT_MAX_WEEKS,
        min_iv_rank: float = constants.DEFAULT_MIN_IV_RANK,
        support_buffer: float = constants.DEFAULT_SUPPORT_BUFFER,
        no_support_filter: bool = False,
        scan_date: Optional[datetime.date] = None,
        cache_provider: Optional[CacheProvider] = None,
    ):
        """Initialize the CSP Scanner with modular components."""
        self.data_provider = data_provider
        self.max_weeks = max_weeks
        self.min_iv_rank = min_iv_rank
        self.support_buffer = support_buffer
        self.no_support_filter = no_support_filter
        self.today = scan_date if scan_date else datetime.date.today()
        self.cache_provider = cache_provider
        
        # Initialize modules
        self.stock_filter = StockFilter(data_provider, cache_provider)
        self.support_calculator = SupportCalculator(data_provider, support_buffer)
        self.option_analyzer = OptionAnalyzer(
            data_provider, max_weeks, min_iv_rank, 
            support_buffer, no_support_filter, scan_date
        )
        self.scoring_engine = ScoringEngine()
        
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
            tables = pd.read_html(constants.NASDAQ_URL)

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
                    if len(tickers) > constants.MIN_NASDAQ_STOCKS:  # Sanity check for a valid table
                        logger.info(f"Found {len(tickers)} NASDAQ-100 stocks from table {i}")
                        return tickers[:100]  # Ensure max 100 stocks

            logger.warning("Could not find NASDAQ-100 table, using defaults")
            return constants.DEFAULT_STOCKS
        except Exception as e:
            logger.error(f"Error fetching NASDAQ-100: {str(e)}")
            return constants.DEFAULT_STOCKS

    def scan_csp_opportunities(self, ticker: str) -> List[dict]:
        """Scan all CSP opportunities for a given ticker using modular components"""
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
            supports = self.support_calculator.calculate_support_levels(ticker)
            if not supports:
                logger.info(f"{ticker} skipped: Could not calculate support levels")
                return opportunities

            # Get option expirations
            expirations = self.data_provider.get_option_expirations(ticker)
            max_date = self.today + datetime.timedelta(weeks=self.max_weeks)

            logger.debug(f"{ticker} has {len(expirations)} expiration dates")

            for exp_str in expirations:
                exp_date = pd.to_datetime(exp_str).date()

                if exp_date > max_date:
                    continue

                # Analyze option chain
                exp_opportunities = self.option_analyzer.analyze_option_chain(
                    ticker, exp_str, exp_date, current_price, supports, iv_rank
                )
                
                opportunities.extend(exp_opportunities)

                time.sleep(constants.API_SLEEP_TIME)  # Rate limiting

            if not opportunities:
                logger.info(f"{ticker}: No opportunities found - all chains filtered out")

        except Exception as e:
            logger.error(f"CSP scan error for {ticker}: {str(e)}")

        return opportunities

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
        quality_stocks = self.stock_filter.get_quality_stocks(tickers, exchange)

        if not quality_stocks:
            logger.warning("No stocks passed quality filters")
            logger.info("Possible reasons:")
            logger.info(f"- Market cap < ${constants.MIN_MARKET_CAP/1e9:.0f}B")
            pe_limit = constants.NASDAQ_MAX_PE_RATIO if exchange in ["NASDAQ", "NASDAQ_500", "NASDAQ_ALL"] else constants.MAX_PE_RATIO
            logger.info(f"- P/E ratio > {pe_limit}")
            logger.info(f"- Forward EPS <= ${constants.MIN_FORWARD_EPS}")
            return_limit = constants.NASDAQ_MIN_ONE_YEAR_RETURN if exchange in ["NASDAQ", "NASDAQ_500", "NASDAQ_ALL"] else constants.MIN_ONE_YEAR_RETURN
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
            logger.info(f"- No liquid options (volume<{constants.MIN_VOLUME} or OI<{constants.MIN_OPEN_INTEREST})")
            if not self.no_support_filter:
                logger.info(f"- No strikes within support range (buffer={self.support_buffer*100:.1f}%)")
            logger.info(f"- Delta outside range ({constants.MIN_DELTA} to {constants.MAX_DELTA})")
            logger.info(f"- Insufficient premium (daily<${constants.MIN_DAILY_PREMIUM}) or ROC (<{constants.MIN_ANNUALIZED_ROC}%)")
            logger.info("- Earnings within 7 days of expiration")
            
            if reasons_no_opps:
                logger.info("\nStocks with no opportunities:")
                for ticker, reason in list(reasons_no_opps.items())[:5]:  # Show first 5
                    logger.info(f"  {ticker}: {reason}")

        logger.info(f"{'='*60}\n")

        # Rank and return
        return self.scoring_engine.rank_opportunities(all_opportunities)


def main():
    """Main entry point for the CSP scanner"""
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
        default=constants.DEFAULT_MAX_WEEKS,
        help=f"Maximum weeks to expiration (default: {constants.DEFAULT_MAX_WEEKS})",
    )
    parser.add_argument(
        "--min-iv-rank",
        type=float,
        default=constants.DEFAULT_MIN_IV_RANK,
        help=f"Minimum IV rank percentage (default: {constants.DEFAULT_MIN_IV_RANK})",
    )
    parser.add_argument(
        "--support-buffer",
        type=float,
        default=constants.DEFAULT_SUPPORT_BUFFER,
        help=f"Support buffer percentage (default: {constants.DEFAULT_SUPPORT_BUFFER})",
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
        default=constants.CACHE_DB_FILE,
        help=f"Path to cache database file (default: {constants.CACHE_DB_FILE})",
    )

    args = parser.parse_args()

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

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
        cache_provider = SQLiteCacheProvider(args.cache_db, constants.CACHE_LIFETIME_HOURS)
        
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
            tickers = constants.DEFAULT_STOCKS

    # Run scan
    results = scanner.run_scan(tickers, exchange=args.exchange)

    # Output results
    if not results.empty:
        print("\n" + "=" * 100)
        print(f"TOP CASH-SECURED PUT OPPORTUNITIES (via {args.provider})")
        print("=" * 100 + "\n")
        print(scanner.scoring_engine.format_results(results, args.top))
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
2025-01-04 14:30:15 - INFO - Fetching NASDAQ 500 stocks...
2025-01-04 14:30:16 - INFO - Found 483 NASDAQ 500 stocks

============================================================
Starting CSP scan for 483 tickers
Exchange: NASDAQ_500
Settings: max_weeks=8, min_iv_rank=15.0%
Support buffer: 3.0%, filter: enabled
============================================================

2025-01-04 14:30:16 - INFO - Using relaxed quality filters for NASDAQ: P/E<=60, 1yr>=-10%
2025-01-04 14:30:16 - INFO - Starting quality filter for 483 NASDAQ stocks
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