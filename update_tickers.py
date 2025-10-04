#!/usr/bin/env python3
"""
Ticker Data Pre-fetcher and Cache Manager

This script pre-fetches comprehensive stock data for S&P 500, NASDAQ-100, NASDAQ-500, or ALL NASDAQ stocks
and stores it in a cache for efficient retrieval. Designed to run as a daily cron job.

Features:
- Fetches comprehensive data including price, volume, financials, technicals, options
- Shows progress with ETA
- Handles errors gracefully with retry logic
- Supports resuming from interruptions
- Rate limiting to avoid API throttling
"""

import argparse
import datetime
import json
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from tqdm import tqdm

from cache_providers import SQLiteCacheProvider
from data_providers import YFinanceProvider
from modules import constants
from nasdaq_fetcher import NASDAQFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("update_tickers.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global SHUTDOWN_REQUESTED
    logger.info(f"Received signal {signum}. Requesting graceful shutdown...")
    SHUTDOWN_REQUESTED = True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class TickerDataFetcher:
    """Fetches and caches comprehensive ticker data"""

    def __init__(
        self,
        cache_provider: SQLiteCacheProvider,
        max_workers: int = 5,
        retry_count: int = 3,
        rate_limit_delay: float = 0.1,
    ):
        """
        Initialize the fetcher

        Args:
            cache_provider: Cache provider instance
            max_workers: Maximum concurrent workers
            retry_count: Number of retries for failed requests
            rate_limit_delay: Delay between API requests in seconds
        """
        self.cache_provider = cache_provider
        self.max_workers = max_workers
        self.retry_count = retry_count
        self.rate_limit_delay = rate_limit_delay
        self.yf_provider = YFinanceProvider()
        self.nasdaq_fetcher = NASDAQFetcher()

        # Track progress
        self.total_processed = 0
        self.total_errors = 0
        self.start_time = None

    def fetch_comprehensive_data(self, ticker: str) -> Optional[Dict]:
        """
        Fetch comprehensive data for a single ticker

        Returns:
            Dictionary with all ticker data or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Basic info
            data = {
                "ticker": ticker,
                "last_updated": datetime.datetime.now().isoformat(),
                # Basic information
                "company_name": info.get("longName", info.get("shortName", "")),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "employees": info.get("fullTimeEmployees", 0),
                "website": info.get("website", ""),
                "description": info.get("longBusinessSummary", "")[
                    :500
                ],  # Limit description length
                # Price data
                "current_price": info.get(
                    "regularMarketPrice", info.get("previousClose", 0)
                ),
                "previous_close": info.get("previousClose", 0),
                "open": info.get("regularMarketOpen", info.get("open", 0)),
                "day_high": info.get("dayHigh", 0),
                "day_low": info.get("dayLow", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "fifty_day_avg": info.get("fiftyDayAverage", 0),
                "two_hundred_day_avg": info.get("twoHundredDayAverage", 0),
                # Volume data
                "volume": info.get("regularMarketVolume", info.get("volume", 0)),
                "avg_volume": info.get("averageVolume", 0),
                "avg_volume_10d": info.get("averageVolume10days", 0),
                "shares_outstanding": info.get("sharesOutstanding", 0),
                "float_shares": info.get("floatShares", 0),
                # Valuation metrics
                "trailing_pe": info.get("trailingPE", None),
                "forward_pe": info.get("forwardPE", None),
                "peg_ratio": info.get("pegRatio", None),
                "price_to_sales": info.get("priceToSalesTrailing12Months", None),
                "price_to_book": info.get("priceToBook", None),
                "enterprise_to_revenue": info.get("enterpriseToRevenue", None),
                "enterprise_to_ebitda": info.get("enterpriseToEbitda", None),
                # Earnings & Dividends
                "trailing_eps": info.get("trailingEps", 0),
                "forward_eps": info.get("forwardEps", 0),
                "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth", None),
                "revenue_quarterly_growth": info.get("revenueQuarterlyGrowth", None),
                "dividend_rate": info.get("dividendRate", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "ex_dividend_date": info.get("exDividendDate", None),
                "payout_ratio": info.get("payoutRatio", 0),
                # Financial metrics
                "profit_margin": info.get("profitMargins", None),
                "operating_margin": info.get("operatingMargins", None),
                "gross_margin": info.get("grossMargins", None),
                "revenue": info.get("totalRevenue", 0),
                "revenue_per_share": info.get("revenuePerShare", 0),
                "gross_profit": info.get("grossProfits", 0),
                "ebitda": info.get("ebitda", 0),
                "net_income": info.get("netIncomeToCommon", 0),
                "total_cash": info.get("totalCash", 0),
                "total_cash_per_share": info.get("totalCashPerShare", 0),
                "total_debt": info.get("totalDebt", 0),
                "debt_to_equity": info.get("debtToEquity", None),
                "current_ratio": info.get("currentRatio", None),
                "quick_ratio": info.get("quickRatio", None),
                "return_on_assets": info.get("returnOnAssets", None),
                "return_on_equity": info.get("returnOnEquity", None),
                # Risk metrics
                "beta": info.get("beta", None),
                "beta_3y": info.get("beta3Year", None),
                # Analyst data
                "target_high": info.get("targetHighPrice", None),
                "target_low": info.get("targetLowPrice", None),
                "target_mean": info.get("targetMeanPrice", None),
                "target_median": info.get("targetMedianPrice", None),
                "recommendation_mean": info.get("recommendationMean", None),
                "recommendation_key": info.get("recommendationKey", ""),
                "number_of_analysts": info.get("numberOfAnalystOpinions", 0),
            }

            # Calculate historical volatility
            hist_volatility = self._calculate_historical_volatility(ticker)
            if hist_volatility:
                data.update(hist_volatility)

            # Get returns
            returns = self._calculate_returns(ticker)
            if returns:
                data.update(returns)

            # Get technical indicators
            technicals = self._calculate_technicals(ticker)
            if technicals:
                data.update(technicals)

            # Get next earnings date
            earnings_date = self._get_next_earnings_date(ticker)
            if earnings_date:
                data["next_earnings_date"] = earnings_date.isoformat()
            else:
                data["next_earnings_date"] = None

            # Get IV rank (using historical volatility as proxy)
            data["iv_rank"] = self.yf_provider.get_iv_rank(ticker)

            # Get options data
            options_data = self._get_options_summary(ticker)
            if options_data:
                data.update(options_data)

            return data

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def _calculate_historical_volatility(self, ticker: str) -> Optional[Dict]:
        """Calculate historical volatility metrics"""
        try:
            hist = yf.Ticker(ticker).history(period="1y")
            if hist.empty:
                return None

            # Calculate returns
            returns = hist["Close"].pct_change().dropna()

            # Rolling volatilities
            vol_10d = returns.rolling(10).std() * np.sqrt(252) * 100
            vol_30d = returns.rolling(30).std() * np.sqrt(252) * 100
            vol_60d = returns.rolling(60).std() * np.sqrt(252) * 100
            vol_90d = returns.rolling(90).std() * np.sqrt(252) * 100

            return {
                "volatility_10d": vol_10d.iloc[-1] if len(vol_10d) > 0 else None,
                "volatility_30d": vol_30d.iloc[-1] if len(vol_30d) > 0 else None,
                "volatility_60d": vol_60d.iloc[-1] if len(vol_60d) > 0 else None,
                "volatility_90d": vol_90d.iloc[-1] if len(vol_90d) > 0 else None,
                "volatility_1y": returns.std() * np.sqrt(252) * 100,
            }
        except Exception:
            return None

    def _calculate_returns(self, ticker: str) -> Optional[Dict]:
        """Calculate various return metrics"""
        try:
            stock = yf.Ticker(ticker)

            # Get historical data
            hist_1y = stock.history(period="1y")
            hist_6mo = stock.history(period="6mo")
            hist_3mo = stock.history(period="3mo")
            hist_1mo = stock.history(period="1mo")
            hist_1wk = stock.history(period="5d")

            returns = {}

            # 1 year return
            if len(hist_1y) > 0:
                returns["return_1y"] = (
                    hist_1y["Close"].iloc[-1] / hist_1y["Close"].iloc[0] - 1
                ) * 100

            # 6 month return
            if len(hist_6mo) > 0:
                returns["return_6mo"] = (
                    hist_6mo["Close"].iloc[-1] / hist_6mo["Close"].iloc[0] - 1
                ) * 100

            # 3 month return
            if len(hist_3mo) > 0:
                returns["return_3mo"] = (
                    hist_3mo["Close"].iloc[-1] / hist_3mo["Close"].iloc[0] - 1
                ) * 100

            # 1 month return
            if len(hist_1mo) > 0:
                returns["return_1mo"] = (
                    hist_1mo["Close"].iloc[-1] / hist_1mo["Close"].iloc[0] - 1
                ) * 100

            # 1 week return
            if len(hist_1wk) > 0:
                returns["return_1wk"] = (
                    hist_1wk["Close"].iloc[-1] / hist_1wk["Close"].iloc[0] - 1
                ) * 100

            # Year-to-date return
            today = datetime.date.today()
            year_start = datetime.date(today.year, 1, 1)
            ytd_hist = stock.history(start=year_start)
            if len(ytd_hist) > 0:
                returns["return_ytd"] = (
                    ytd_hist["Close"].iloc[-1] / ytd_hist["Close"].iloc[0] - 1
                ) * 100

            return returns

        except Exception:
            return None

    def _calculate_technicals(self, ticker: str) -> Optional[Dict]:
        """Calculate technical indicators"""
        try:
            hist = yf.Ticker(ticker).history(period="3mo")
            if len(hist) < 20:
                return None

            close = hist["Close"]
            high = hist["High"]
            low = hist["Low"]
            volume = hist["Volume"]

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Support and Resistance
            # Simple method: use recent highs and lows
            support_1 = low.tail(20).min()
            support_2 = low.tail(50).min()
            resistance_1 = high.tail(20).max()
            resistance_2 = high.tail(50).max()

            # Moving averages
            sma_20 = close.rolling(window=20).mean().iloc[-1]
            sma_50 = (
                close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else None
            )

            # MACD
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()

            # Bollinger Bands
            bb_sma = close.rolling(window=20).mean()
            bb_std = close.rolling(window=20).std()
            bb_upper = bb_sma + (bb_std * 2)
            bb_lower = bb_sma - (bb_std * 2)

            # Average True Range (ATR)
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]

            return {
                "rsi_14": rsi.iloc[-1] if not rsi.empty else None,
                "support_level_1": support_1,
                "support_level_2": support_2,
                "resistance_level_1": resistance_1,
                "resistance_level_2": resistance_2,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "macd": macd.iloc[-1] if not macd.empty else None,
                "macd_signal": signal.iloc[-1] if not signal.empty else None,
                "bb_upper": bb_upper.iloc[-1] if not bb_upper.empty else None,
                "bb_lower": bb_lower.iloc[-1] if not bb_lower.empty else None,
                "atr_14": atr,
                "relative_volume": (
                    volume.iloc[-1] / volume.tail(20).mean()
                    if volume.tail(20).mean() > 0
                    else None
                ),
            }

        except Exception:
            return None

    def _get_next_earnings_date(self, ticker: str) -> Optional[datetime.date]:
        """Get next earnings date"""
        try:
            earnings_dates = self.yf_provider.get_earnings_dates(ticker)
            if earnings_dates:
                # Return the first future earnings date
                today = datetime.date.today()
                for date in earnings_dates:
                    if date > today:
                        return date
            return None
        except Exception:
            return None

    def _get_options_summary(self, ticker: str) -> Optional[Dict]:
        """Get options summary data"""
        try:
            stock = yf.Ticker(ticker)
            options = stock.options

            if not options:
                return None

            # Get the nearest expiration
            nearest_exp = options[0]
            opt_chain = stock.option_chain(nearest_exp)

            puts = opt_chain.puts
            calls = opt_chain.calls

            # Calculate put/call ratio
            put_volume = puts["volume"].sum()
            call_volume = calls["volume"].sum()
            put_call_ratio = put_volume / call_volume if call_volume > 0 else None

            # Get average IV for ATM options
            current_price = stock.info.get(
                "regularMarketPrice", stock.info.get("previousClose", 0)
            )
            if current_price > 0:
                # Find ATM puts
                atm_puts = puts[
                    abs(puts["strike"] - current_price) <= current_price * 0.05
                ]
                avg_put_iv = (
                    atm_puts["impliedVolatility"].mean() * 100
                    if not atm_puts.empty
                    else None
                )

                # Find ATM calls
                atm_calls = calls[
                    abs(calls["strike"] - current_price) <= current_price * 0.05
                ]
                avg_call_iv = (
                    atm_calls["impliedVolatility"].mean() * 100
                    if not atm_calls.empty
                    else None
                )
            else:
                avg_put_iv = None
                avg_call_iv = None

            return {
                "options_volume": put_volume + call_volume,
                "put_call_ratio": put_call_ratio,
                "avg_put_iv": avg_put_iv,
                "avg_call_iv": avg_call_iv,
                "num_expirations": len(options),
            }

        except Exception:
            return None

    def fetch_with_retry(self, ticker: str) -> Optional[Dict]:
        """Fetch ticker data with retry logic"""
        for attempt in range(self.retry_count):
            if SHUTDOWN_REQUESTED:
                logger.info(f"Shutdown requested, skipping {ticker}")
                return None

            try:
                time.sleep(self.rate_limit_delay)  # Rate limiting
                data = self.fetch_comprehensive_data(ticker)
                if data:
                    return data
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(2**attempt)  # Exponential backoff

        return None

    def update_tickers(
        self, tickers: List[str], exchange: str, resume: bool = True
    ) -> Dict[str, int]:
        """
        Update ticker data for given list of tickers

        Args:
            tickers: List of ticker symbols
            exchange: Exchange name
            resume: Whether to resume from last position

        Returns:
            Dictionary with update statistics
        """
        self.start_time = time.time()

        # Get already processed tickers if resuming
        processed_tickers = set()
        if resume:
            cached_tickers = self.cache_provider.get_cached_tickers(exchange)
            processed_tickers = set(cached_tickers) if cached_tickers else set()
            logger.info(f"Found {len(processed_tickers)} already cached tickers")

        # Filter out already processed tickers
        remaining_tickers = [t for t in tickers if t not in processed_tickers]

        if not remaining_tickers:
            logger.info("All tickers already cached")
            return {"processed": len(tickers), "errors": 0, "skipped": 0}

        logger.info(f"Processing {len(remaining_tickers)} tickers...")

        # Process tickers
        with tqdm(
            total=len(remaining_tickers), desc=f"Updating {exchange} tickers"
        ) as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_ticker = {
                    executor.submit(self.fetch_with_retry, ticker): ticker
                    for ticker in remaining_tickers
                }

                # Process completed tasks
                for future in as_completed(future_to_ticker):
                    if SHUTDOWN_REQUESTED:
                        logger.info("Shutdown requested, cancelling remaining tasks...")
                        executor.shutdown(wait=False)
                        break

                    ticker = future_to_ticker[future]
                    try:
                        data = future.result()
                        if data:
                            # Save to cache
                            self.cache_provider.save_ticker_data(ticker, exchange, data)
                            self.total_processed += 1
                        else:
                            self.total_errors += 1
                            logger.error(f"Failed to fetch data for {ticker}")
                    except Exception as e:
                        self.total_errors += 1
                        logger.error(f"Error processing {ticker}: {str(e)}")

                    # Update progress
                    pbar.update(1)

                    # Show ETA
                    elapsed = time.time() - self.start_time
                    if self.total_processed > 0:
                        rate = self.total_processed / elapsed
                        remaining = len(remaining_tickers) - (
                            self.total_processed + self.total_errors
                        )
                        eta = remaining / rate if rate > 0 else 0
                        pbar.set_postfix(
                            {
                                "Processed": self.total_processed,
                                "Errors": self.total_errors,
                                "ETA": f"{eta/60:.1f}m",
                            }
                        )

        # Final statistics
        elapsed_time = time.time() - self.start_time
        stats = {
            "processed": self.total_processed,
            "errors": self.total_errors,
            "skipped": len(processed_tickers),
            "elapsed_seconds": elapsed_time,
            "tickers_per_second": (
                self.total_processed / elapsed_time if elapsed_time > 0 else 0
            ),
        }

        return stats


def get_exchange_tickers(
    exchange: str, min_market_cap: Optional[float] = None
) -> List[str]:
    """Get list of tickers for specified exchange"""
    if exchange == "SP500":
        # Fetch S&P 500 from Wikipedia
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            sp500_table = tables[0]
            tickers = sp500_table["Symbol"].str.replace(".", "-", regex=False).tolist()
            logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            logger.error(f"Error fetching S&P 500 list: {str(e)}")
            return []

    elif exchange in ["NASDAQ", "NASDAQ_500", "NASDAQ_ALL"]:
        fetcher = NASDAQFetcher()

        if exchange == "NASDAQ":
            # NASDAQ-100
            return fetcher.get_nasdaq_100()
        elif exchange == "NASDAQ_500":
            # Top 500 NASDAQ stocks by market cap
            stocks = fetcher.get_nasdaq_500(include_info=False)
            return [s["symbol"] for s in stocks]
        else:  # NASDAQ_ALL
            # All NASDAQ stocks with optional market cap filter
            stocks = fetcher.get_all_nasdaq_stocks(
                min_market_cap=min_market_cap, include_info=False
            )
            return [s["symbol"] for s in stocks]

    else:
        logger.error(f"Unknown exchange: {exchange}")
        return []


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Pre-fetch and cache comprehensive stock data"
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="SP500",
        choices=["SP500", "NASDAQ", "NASDAQ_500", "NASDAQ_ALL"],
        help="Stock exchange to update (default: SP500)",
    )
    parser.add_argument(
        "--min-market-cap",
        type=float,
        help="Minimum market cap in billions (e.g., 1 for $1B)",
    )
    parser.add_argument(
        "--cache-db",
        type=str,
        default=constants.CACHE_DB_FILE,
        help=f"Cache database file (default: {constants.CACHE_DB_FILE})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent workers (default: 5)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, don't resume from last position",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear existing cache before updating",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.1,
        help="Rate limit delay between requests in seconds (default: 0.1)",
    )

    args = parser.parse_args()

    # Initialize cache provider with extended schema
    cache_provider = SQLiteCacheProvider(args.cache_db)

    if args.clear_cache:
        logger.info("Clearing existing cache...")
        cache_provider.clear_ticker_data()

    # Convert market cap to actual value
    min_market_cap = args.min_market_cap * 1e9 if args.min_market_cap else None

    # Get tickers
    logger.info(f"Fetching {args.exchange} tickers...")
    tickers = get_exchange_tickers(args.exchange, min_market_cap)

    if not tickers:
        logger.error("No tickers found")
        return 1

    logger.info(f"Found {len(tickers)} tickers to process")

    # Initialize fetcher
    fetcher = TickerDataFetcher(
        cache_provider=cache_provider,
        max_workers=args.workers,
        rate_limit_delay=args.rate_limit,
    )

    # Run update
    stats = fetcher.update_tickers(
        tickers=tickers, exchange=args.exchange, resume=not args.no_resume
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Update Summary")
    logger.info("=" * 60)
    logger.info(f"Exchange: {args.exchange}")
    logger.info(f"Total tickers: {len(tickers)}")
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Skipped (already cached): {stats['skipped']}")
    logger.info(f"Elapsed time: {stats['elapsed_seconds']/60:.1f} minutes")
    logger.info(f"Processing rate: {stats['tickers_per_second']:.2f} tickers/second")

    if SHUTDOWN_REQUESTED:
        logger.info("\nUpdate interrupted by user")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
