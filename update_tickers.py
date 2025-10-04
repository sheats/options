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
import sqlite3
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
        Fetch comprehensive data for a single ticker with CSP hotlist criteria

        Returns:
            Dictionary with all ticker data or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get all available data - no filtering
            market_cap = info.get("marketCap", 0)
            forward_eps = info.get("forwardEps", 0)

            # Basic info
            data = {
                "ticker": ticker,
                "last_updated": datetime.datetime.now().isoformat(),
                # Basic information
                "company_name": info.get("longName", info.get("shortName", "")),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": market_cap,
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
                # CSP-critical ownership and quality metrics
                "institutional_ownership": info.get("heldPercentInstitutions", 0),
                "insider_ownership": info.get("heldPercentInsiders", 0),
                "short_ratio": info.get("shortRatio", 0),
                "shares_short": info.get("sharesShort", 0),
                # Valuation metrics
                "trailing_pe": info.get("trailingPE", None),
                "forward_pe": info.get("forwardPE", None),
                "peg_ratio": info.get("pegRatio", None),
                "price_to_sales": info.get("priceToSalesTrailing12Months", None),
                "price_to_book": info.get("priceToBook", None),
                "enterprise_to_revenue": info.get("enterpriseToRevenue", None),
                "enterprise_to_ebitda": info.get("enterpriseToEbitda", None),
                # Earnings & Growth (CSP critical)
                "trailing_eps": info.get("trailingEps", 0),
                "forward_eps": forward_eps,
                "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth", None),
                "earnings_growth": info.get("earningsGrowth", None),  # Annual EPS growth
                "revenue_quarterly_growth": info.get("revenueQuarterlyGrowth", None),
                "revenue_growth": info.get("revenueGrowth", None),  # Annual revenue growth
                # Dividends (CSP filter: yield > 1%)
                "dividend_rate": info.get("dividendRate", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "ex_dividend_date": info.get("exDividendDate", None),
                "payout_ratio": info.get("payoutRatio", 0),
                "five_year_avg_dividend_yield": info.get("fiveYearAvgDividendYield", 0),
                # Financial metrics (CSP critical: debt/equity < 1)
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
                "free_cash_flow": info.get("freeCashflow", 0),
                "operating_cash_flow": info.get("operatingCashflow", 0),
                # Risk metrics (CSP filter: beta < 1.2)
                "beta": info.get("beta", None),
                "beta_3y": info.get("beta3Year", None),
                # Analyst data (CSP filter: recommendation < 2.5)
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

            # Get IV rank (critical for CSP hotlist)
            # Try to get from OpenBB or other sources if available, else use proxy
            iv_rank = self.yf_provider.get_iv_rank(ticker)
            data["iv_rank"] = iv_rank
            
            # Also store IV rank proxy for comparison
            if data.get("volatility_30d") and data.get("volatility_1y"):
                # Simple proxy: current 30d vol percentile vs 1y range
                vol_min = min(data.get("volatility_10d", 100), data.get("volatility_30d", 100), 
                             data.get("volatility_60d", 100), data.get("volatility_90d", 100))
                vol_max = data.get("volatility_1y", 0)
                if vol_max > vol_min:
                    iv_rank_proxy = ((data["volatility_30d"] - vol_min) / (vol_max - vol_min)) * 100
                    data["iv_rank_proxy"] = min(100, max(0, iv_rank_proxy))
                else:
                    data["iv_rank_proxy"] = 50.0  # Default if no range
            else:
                data["iv_rank_proxy"] = iv_rank  # Use same as iv_rank if no vol data

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

    def generate_hotlist(self, exchange: str, min_iv_rank: float = 50.0) -> pd.DataFrame:
        """
        Generate CSP hotlist from cached data based on strict criteria
        """
        logger.info(f"\nGenerating CSP hotlist for {exchange}...")
        
        # Get all cached tickers for exchange
        cached_tickers = self.cache_provider.get_cached_tickers(exchange)
        if not cached_tickers:
            logger.warning("No cached tickers found. Run update first.")
            return pd.DataFrame()
        
        hotlist_candidates = []
        
        for ticker in cached_tickers:
            try:
                # Get cached data
                data = self.cache_provider.get_ticker_data(ticker)
                if not data:
                    continue
                
                # Apply CSP hotlist filters
                market_cap = data.get("market_cap", 0)
                institutional_ownership = data.get("institutional_ownership", 0)
                debt_to_equity = data.get("debt_to_equity", float('inf'))
                earnings_growth = data.get("earnings_growth", 0)
                recommendation_mean = data.get("recommendation_mean", 5)
                beta = data.get("beta", 2.0)
                dividend_yield = data.get("dividend_yield", 0)
                rsi_14 = data.get("rsi_14", 100)
                macd = data.get("macd", -1)
                iv_rank = data.get("iv_rank", 0)
                iv_rank_proxy = data.get("iv_rank_proxy", 0)
                return_1y = data.get("return_1y", -100)
                
                # Apply filters only if we have the data
                if market_cap and market_cap < 50e9:  # $50B minimum
                    logger.debug(f"Skipping {ticker}: Market cap < $50B")
                    continue
                    
                if (iv_rank or iv_rank_proxy) and max(iv_rank, iv_rank_proxy) < min_iv_rank:
                    logger.debug(f"Skipping {ticker}: IV rank {max(iv_rank, iv_rank_proxy):.1f}% < {min_iv_rank}%")
                    continue
                    
                if institutional_ownership and institutional_ownership < 0.5:  # 50% minimum
                    logger.debug(f"Skipping {ticker}: Institutional ownership {institutional_ownership:.1%} < 50%")
                    continue
                    
                if debt_to_equity is not None and debt_to_equity >= 1.0:  # Less than 1.0 required
                    logger.debug(f"Skipping {ticker}: Debt/Equity {debt_to_equity:.2f} >= 1.0")
                    continue
                    
                if earnings_growth and earnings_growth < 0.05:  # 5% minimum
                    logger.debug(f"Skipping {ticker}: EPS growth {earnings_growth:.1%} < 5%")
                    continue
                    
                if recommendation_mean and recommendation_mean > 2.5:  # Buy rating required
                    logger.debug(f"Skipping {ticker}: Analyst rating {recommendation_mean:.1f} > 2.5")
                    continue
                    
                if beta is not None and beta >= 1.2:  # Stability required
                    logger.debug(f"Skipping {ticker}: Beta {beta:.2f} >= 1.2")
                    continue
                    
                if dividend_yield is not None and dividend_yield < 0.01:  # 1% minimum
                    logger.debug(f"Skipping {ticker}: Dividend yield {dividend_yield:.1%} < 1%")
                    continue
                
                # Calculate hotlist score
                # Higher IV rank, lower beta, higher institutional ownership, oversold RSI = higher score
                effective_iv_rank = max(iv_rank, iv_rank_proxy)
                rsi_factor = 1.0 if rsi_14 < 50 else 0.5  # Bonus for oversold
                momentum_factor = 1.2 if macd > 0 else 1.0  # Bonus for positive momentum
                
                score = (
                    effective_iv_rank * 
                    (2.0 - beta) *  # Lower beta = higher multiplier
                    institutional_ownership * 
                    rsi_factor *
                    momentum_factor
                )
                
                hotlist_candidates.append({
                    "ticker": ticker,
                    "score": score,
                    "company_name": data.get("company_name", ""),
                    "sector": data.get("sector", ""),
                    "market_cap_b": market_cap / 1e9,
                    "iv_rank": effective_iv_rank,
                    "beta": beta,
                    "institutional_ownership": institutional_ownership,
                    "debt_to_equity": debt_to_equity,
                    "earnings_growth": earnings_growth,
                    "dividend_yield": dividend_yield,
                    "recommendation_mean": recommendation_mean,
                    "rsi_14": rsi_14,
                    "macd": macd,
                    "return_1y": return_1y,
                    "current_price": data.get("current_price", 0),
                    "next_earnings_date": data.get("next_earnings_date", ""),
                })
                
            except Exception as e:
                logger.error(f"Error processing {ticker} for hotlist: {str(e)}")
        
        if not hotlist_candidates:
            logger.warning("No stocks matched hotlist criteria. Consider relaxing filters.")
            return pd.DataFrame()
        
        # Create DataFrame and sort by score
        hotlist_df = pd.DataFrame(hotlist_candidates)
        hotlist_df = hotlist_df.sort_values("score", ascending=False).head(50)  # Top 50
        
        # Save to cache
        self._save_hotlist_to_cache(hotlist_df, exchange)
        
        # Export to CSV
        csv_filename = f"hotlist_{exchange}_{datetime.date.today()}.csv"
        hotlist_df.to_csv(csv_filename, index=False)
        logger.info(f"Exported hotlist to {csv_filename}")
        
        logger.info(f"Generated hotlist with {len(hotlist_df)} stocks")
        
        return hotlist_df
    
    def _save_hotlist_to_cache(self, hotlist_df: pd.DataFrame, exchange: str):
        """Save hotlist to cache database"""
        try:
            # Access the database directly since cache provider doesn't expose connection
            import sqlite3
            with sqlite3.connect(self.cache_provider.db_path) as conn:
                cursor = conn.cursor()
                
                # Create hotlist table if not exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS hotlist (
                        ticker TEXT,
                        exchange TEXT,
                        score REAL,
                        iv_rank REAL,
                        beta REAL,
                        rsi_14 REAL,
                        institutional_ownership REAL,
                        debt_to_equity REAL,
                        earnings_growth REAL,
                        dividend_yield REAL,
                        recommendation_mean REAL,
                        market_cap_b REAL,
                        data TEXT,
                        last_updated TIMESTAMP,
                        PRIMARY KEY (ticker, exchange)
                    )
                """)
                
                # Clear existing hotlist for this exchange
                cursor.execute("DELETE FROM hotlist WHERE exchange = ?", (exchange,))
                
                # Insert new hotlist
                for _, row in hotlist_df.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO hotlist VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row["ticker"],
                        exchange,
                        row["score"],
                        row["iv_rank"],
                        row["beta"],
                        row["rsi_14"],
                        row["institutional_ownership"],
                        row["debt_to_equity"],
                        row.get("earnings_growth", 0),
                        row["dividend_yield"],
                        row["recommendation_mean"],
                        row["market_cap_b"],
                        json.dumps(row.to_dict()),
                        datetime.datetime.now()
                    ))
                
                conn.commit()
                logger.info(f"Saved {len(hotlist_df)} stocks to hotlist cache")
            
        except Exception as e:
            logger.error(f"Error saving hotlist to cache: {str(e)}")
    
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


def get_exchange_tickers(exchange: str) -> List[str]:
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
            # All NASDAQ stocks
            stocks = fetcher.get_all_nasdaq_stocks(include_info=False)
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
    parser.add_argument(
        "--hotlist-only",
        action="store_true",
        help="Skip ticker update and only generate hotlist from cache",
    )
    parser.add_argument(
        "--min-iv-rank",
        type=float,
        default=50.0,
        help="Minimum IV rank for hotlist (default: 50.0)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output (for cron jobs)",
    )

    args = parser.parse_args()

    # Initialize cache provider with extended schema
    cache_provider = SQLiteCacheProvider(args.cache_db)

    if args.clear_cache:
        logger.info("Clearing existing cache...")
        cache_provider.clear_ticker_data()


    # Configure logging for quiet mode
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Initialize fetcher
    fetcher = TickerDataFetcher(
        cache_provider=cache_provider,
        max_workers=args.workers,
        rate_limit_delay=args.rate_limit,
    )
    
    # If hotlist-only mode, skip ticker update
    if args.hotlist_only:
        logger.info("Hotlist-only mode: Generating hotlist from cache")
        hotlist = fetcher.generate_hotlist(args.exchange, args.min_iv_rank)
        if not hotlist.empty:
            print(f"\n{'='*80}")
            print(f"CSP Hotlist for {args.exchange} (Top 10)")
            print(f"{'='*80}")
            print(hotlist[['ticker', 'score', 'iv_rank', 'beta', 'rsi_14', 'institutional_ownership', 'dividend_yield']].head(10).to_string(index=False))
            print(f"\nFull hotlist saved to: hotlist_{args.exchange}_{datetime.date.today()}.csv")
        return 0
    
    # Get tickers
    logger.info(f"Fetching {args.exchange} tickers...")
    tickers = get_exchange_tickers(args.exchange)

    if not tickers:
        logger.error("No tickers found")
        return 1

    logger.info(f"Found {len(tickers)} tickers to process")

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
    
    # Generate hotlist after update
    if stats['processed'] > 0:
        logger.info("\nGenerating CSP hotlist...")
        hotlist = fetcher.generate_hotlist(args.exchange, args.min_iv_rank)
        
        if not hotlist.empty:
            # Display sample results
            print(f"\n{'='*100}")
            print(f"CSP Hotlist for {args.exchange} - Top Stocks for Cash-Secured Puts")
            print(f"{'='*100}")
            print("\nSample (Top 3):")
            for idx, row in hotlist.head(3).iterrows():
                print(f"\n{row['ticker']}: Score={row['score']:.1f}, IV_rank={row['iv_rank']:.1f}%, Beta={row['beta']:.2f}")
                print(f"  Inst. Own: {row['institutional_ownership']:.1%}, D/E: {row['debt_to_equity']:.2f}, RSI: {row['rsi_14']:.1f}")
                print(f"  Div Yield: {row['dividend_yield']:.1%}, 1Y Return: {row['return_1y']:.1f}%, Price: ${row['current_price']:.2f}")
            
            print(f"\nFull hotlist with {len(hotlist)} stocks saved to: hotlist_{args.exchange}_{datetime.date.today()}.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
