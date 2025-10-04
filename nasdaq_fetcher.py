#!/usr/bin/env python3
"""
NASDAQ Stock Fetcher - Comprehensive solution for fetching all NASDAQ stocks

This module provides multiple methods to fetch NASDAQ stocks:
1. NASDAQ FTP server (official source)
2. nasdaq-trader.com API
3. Yahoo Finance screening
4. Cache management for efficiency

Features:
- Fetch ALL NASDAQ stocks (not just NASDAQ-100)
- Filter by market cap
- Company information (name, market cap, sector)
- Rate limiting and error handling
- Caching to avoid repeated downloads
"""

import ftplib
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".nasdaq_cache")
CACHE_EXPIRY_HOURS = 24  # Cache expires after 24 hours
os.makedirs(CACHE_DIR, exist_ok=True)

# Rate limiting
RATE_LIMIT_DELAY = 0.1  # 100ms between requests

# Data sources
NASDAQ_FTP_HOST = "ftp.nasdaqtrader.com"
NASDAQ_FTP_PATH = "/SymbolDirectory/nasdaqlisted.txt"
NASDAQ_TRADER_API = "http://www.nasdaqtrader.com/dynamic/SymDirData/nasdaqlisted.txt"
FINVIZ_URL = "https://finviz.com/screener.ashx"


class NASDAQFetcher:
    """Comprehensive NASDAQ stock fetcher with multiple data sources"""

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize NASDAQ fetcher

        Args:
            cache_enabled: Whether to use caching (default: True)
        """
        self.cache_enabled = cache_enabled
        self._cache = {}

    def get_all_nasdaq_stocks(
        self,
        min_market_cap: Optional[float] = None,
        include_info: bool = True,
        source: str = "auto",
    ) -> List[Dict[str, any]]:
        """
        Get all NASDAQ stocks with optional filtering

        Args:
            min_market_cap: Minimum market cap filter (e.g., 1e9 for $1B)
            include_info: Include company info (name, market cap, etc.)
            source: Data source ("ftp", "api", "yahoo", "auto")

        Returns:
            List of dictionaries with stock information
        """
        # Check cache first
        cache_key = f"nasdaq_all_{min_market_cap}_{include_info}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            logger.info(f"Loaded {len(cached_data)} stocks from cache")
            return cached_data

        # Fetch from source
        stocks = []

        if source == "auto":
            # Try sources in order of preference
            for src in ["ftp", "api", "yahoo"]:
                try:
                    stocks = self._fetch_from_source(src)
                    if stocks:
                        logger.info(
                            f"Successfully fetched {len(stocks)} stocks from {src}"
                        )
                        break
                except Exception as e:
                    logger.warning(f"Failed to fetch from {src}: {e}")
        else:
            stocks = self._fetch_from_source(source)

        if not stocks:
            logger.error("Failed to fetch stocks from any source")
            return []

        # Enhance with additional info if requested
        if include_info:
            stocks = self._enhance_stock_info(stocks, min_market_cap)
        elif min_market_cap:
            # If not including info but need to filter by market cap,
            # we still need to fetch market cap data
            stocks = self._filter_by_market_cap(stocks, min_market_cap)

        # Cache the results
        self._save_to_cache(cache_key, stocks)

        return stocks

    def _fetch_from_source(self, source: str) -> List[Dict[str, str]]:
        """Fetch stocks from specified source"""
        if source == "ftp":
            return self._fetch_from_ftp()
        elif source == "api":
            return self._fetch_from_api()
        elif source == "yahoo":
            return self._fetch_from_yahoo()
        else:
            raise ValueError(f"Unknown source: {source}")

    def _fetch_from_ftp(self) -> List[Dict[str, str]]:
        """Fetch NASDAQ stocks from official FTP server"""
        logger.info("Fetching from NASDAQ FTP server...")

        stocks = []

        try:
            # Connect to FTP
            ftp = ftplib.FTP(NASDAQ_FTP_HOST)
            ftp.login()  # Anonymous login

            # Download the file
            data = io.BytesIO()
            ftp.retrbinary(f"RETR {NASDAQ_FTP_PATH}", data.write)
            ftp.quit()

            # Parse the data
            data.seek(0)
            content = data.read().decode("utf-8")

            # Skip header and parse lines
            lines = content.strip().split("\n")[1:]  # Skip header

            for line in lines:
                if line.strip() and not line.startswith("File Creation Time"):
                    parts = line.split("|")
                    if len(parts) >= 2:
                        symbol = parts[0].strip()
                        name = parts[1].strip()

                        # Skip test symbols
                        if symbol and not symbol.startswith("TEST"):
                            stocks.append(
                                {"symbol": symbol, "name": name, "exchange": "NASDAQ"}
                            )

            logger.info(f"Fetched {len(stocks)} stocks from FTP")
            return stocks

        except Exception as e:
            logger.error(f"FTP fetch failed: {e}")
            raise

    def _fetch_from_api(self) -> List[Dict[str, str]]:
        """Fetch NASDAQ stocks from nasdaq-trader.com API"""
        logger.info("Fetching from NASDAQ Trader API...")

        stocks = []

        try:
            response = requests.get(NASDAQ_TRADER_API, timeout=30)
            response.raise_for_status()

            # Parse the pipe-delimited file
            lines = response.text.strip().split("\n")[1:]  # Skip header

            for line in lines:
                if line.strip() and not line.startswith("File Creation Time"):
                    parts = line.split("|")
                    if len(parts) >= 2:
                        symbol = parts[0].strip()
                        name = parts[1].strip()

                        # Skip test symbols
                        if symbol and not symbol.startswith("TEST"):
                            stocks.append(
                                {"symbol": symbol, "name": name, "exchange": "NASDAQ"}
                            )

            logger.info(f"Fetched {len(stocks)} stocks from API")
            return stocks

        except Exception as e:
            logger.error(f"API fetch failed: {e}")
            raise

    def _fetch_from_yahoo(self) -> List[Dict[str, str]]:
        """Fetch NASDAQ stocks using Yahoo Finance screening"""
        logger.info("Fetching from Yahoo Finance...")

        stocks = []

        try:
            # Use yfinance screener capabilities
            # This is a workaround since yfinance doesn't have a direct screener
            # We'll use a known list of NASDAQ stocks and validate them

            # First, try to get a comprehensive list from other sources
            # For now, return empty as Yahoo doesn't provide a direct listing API
            logger.warning(
                "Yahoo Finance direct listing not implemented - use FTP or API source"
            )
            return []

        except Exception as e:
            logger.error(f"Yahoo fetch failed: {e}")
            raise

    def _enhance_stock_info(
        self, stocks: List[Dict[str, str]], min_market_cap: Optional[float] = None
    ) -> List[Dict[str, any]]:
        """Enhance stock list with additional information"""
        logger.info(f"Enhancing info for {len(stocks)} stocks...")

        enhanced_stocks = []

        # Process in batches to avoid overwhelming APIs
        batch_size = 50
        total_batches = (len(stocks) + batch_size - 1) // batch_size

        with tqdm(total=len(stocks), desc="Fetching stock info") as pbar:
            for i in range(0, len(stocks), batch_size):
                batch = stocks[i : i + batch_size]

                # Use ThreadPoolExecutor for parallel processing
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {
                        executor.submit(self._get_stock_info, stock): stock
                        for stock in batch
                    }

                    for future in as_completed(futures):
                        stock_info = future.result()
                        if stock_info:
                            # Apply market cap filter if specified
                            if (
                                min_market_cap is None
                                or stock_info.get("market_cap", 0) >= min_market_cap
                            ):
                                enhanced_stocks.append(stock_info)

                        pbar.update(1)

                # Rate limiting between batches
                if i + batch_size < len(stocks):
                    time.sleep(1)

        logger.info(f"Enhanced info complete: {len(enhanced_stocks)} stocks")
        return enhanced_stocks

    def _get_stock_info(self, stock: Dict[str, str]) -> Optional[Dict[str, any]]:
        """Get detailed information for a single stock"""
        symbol = stock["symbol"]

        try:
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)

            # Use yfinance to get stock info
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract relevant information
            enhanced_info = {
                "symbol": symbol,
                "name": stock.get(
                    "name", info.get("longName", info.get("shortName", symbol))
                ),
                "exchange": stock.get("exchange", "NASDAQ"),
                "market_cap": info.get("marketCap", 0),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price": info.get("regularMarketPrice", info.get("previousClose")),
                "volume": info.get("regularMarketVolume", info.get("volume")),
                "avg_volume": info.get("averageVolume"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "employees": info.get("fullTimeEmployees"),
                "founded": info.get("founded"),
                "website": info.get("website"),
                "description": (
                    info.get("longBusinessSummary", "")[:200] + "..."
                    if info.get("longBusinessSummary")
                    else ""
                ),
            }

            return enhanced_info

        except Exception as e:
            logger.debug(f"Failed to get info for {symbol}: {e}")
            # Return basic info even if enhancement fails
            return {
                "symbol": symbol,
                "name": stock.get("name", symbol),
                "exchange": stock.get("exchange", "NASDAQ"),
                "market_cap": 0,
                "error": str(e),
            }

    def _filter_by_market_cap(
        self, stocks: List[Dict[str, str]], min_market_cap: float
    ) -> List[Dict[str, any]]:
        """Filter stocks by market cap only (lighter than full enhancement)"""
        logger.info(
            f"Filtering {len(stocks)} stocks by market cap >= ${min_market_cap/1e9:.1f}B..."
        )

        filtered_stocks = []

        with tqdm(total=len(stocks), desc="Checking market caps") as pbar:
            for stock in stocks:
                symbol = stock["symbol"]

                try:
                    time.sleep(RATE_LIMIT_DELAY)
                    ticker = yf.Ticker(symbol)
                    market_cap = ticker.info.get("marketCap", 0)

                    if market_cap >= min_market_cap:
                        stock["market_cap"] = market_cap
                        filtered_stocks.append(stock)

                except Exception:
                    pass  # Skip if can't get market cap

                pbar.update(1)

        logger.info(
            f"Found {len(filtered_stocks)} stocks with market cap >= ${min_market_cap/1e9:.1f}B"
        )
        return filtered_stocks

    def get_nasdaq_500(self, include_info: bool = True) -> List[Dict[str, any]]:
        """Get top 500 NASDAQ stocks by market cap"""
        logger.info("Fetching NASDAQ 500...")

        # Get all NASDAQ stocks with info
        all_stocks = self.get_all_nasdaq_stocks(include_info=True)

        # Sort by market cap and take top 500
        sorted_stocks = sorted(
            [s for s in all_stocks if s.get("market_cap", 0) > 0],
            key=lambda x: x["market_cap"],
            reverse=True,
        )

        nasdaq_500 = sorted_stocks[:500]

        if not include_info:
            # Strip down to just symbols if info not needed
            nasdaq_500 = [
                {"symbol": s["symbol"], "name": s["name"]} for s in nasdaq_500
            ]

        logger.info(f"Selected top {len(nasdaq_500)} NASDAQ stocks by market cap")
        return nasdaq_500

    def get_nasdaq_100(self) -> List[str]:
        """Get NASDAQ-100 index components (for compatibility)"""
        # Try to fetch from Wikipedia as before
        try:
            nasdaq_url = "https://en.wikipedia.org/wiki/NASDAQ-100"
            tables = pd.read_html(nasdaq_url)

            for table in tables:
                if "Ticker" in table.columns or "Symbol" in table.columns:
                    col_name = "Ticker" if "Ticker" in table.columns else "Symbol"
                    tickers = table[col_name].tolist()
                    tickers = [str(t).strip().upper() for t in tickers if pd.notna(t)]
                    if len(tickers) > 50:
                        return tickers[:100]

        except Exception as e:
            logger.warning(f"Failed to fetch NASDAQ-100 from Wikipedia: {e}")

        # Fallback: get top 100 by market cap
        nasdaq_500 = self.get_nasdaq_500()
        return [s["symbol"] for s in nasdaq_500[:100]]

    def _load_from_cache(self, key: str) -> Optional[List[Dict]]:
        """Load data from cache if valid"""
        if not self.cache_enabled:
            return None

        cache_file = os.path.join(CACHE_DIR, f"{key}.json")

        try:
            if os.path.exists(cache_file):
                # Check cache age
                file_age = datetime.now() - datetime.fromtimestamp(
                    os.path.getmtime(cache_file)
                )
                if file_age < timedelta(hours=CACHE_EXPIRY_HOURS):
                    with open(cache_file, "r") as f:
                        return json.load(f)
        except Exception as e:
            logger.debug(f"Cache load failed: {e}")

        return None

    def _save_to_cache(self, key: str, data: List[Dict]):
        """Save data to cache"""
        if not self.cache_enabled:
            return

        cache_file = os.path.join(CACHE_DIR, f"{key}.json")

        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")

    def clear_cache(self):
        """Clear all cached data"""
        logger.info("Clearing cache...")

        for file in os.listdir(CACHE_DIR):
            if file.endswith(".json"):
                os.remove(os.path.join(CACHE_DIR, file))

        logger.info("Cache cleared")


# Convenience functions
def get_nasdaq_stocks(
    count: int = 500, min_market_cap: Optional[float] = None, include_info: bool = True
) -> List[Dict[str, any]]:
    """
    Convenience function to get NASDAQ stocks

    Args:
        count: Number of stocks to return (by market cap)
        min_market_cap: Minimum market cap filter
        include_info: Include detailed stock information

    Returns:
        List of stock dictionaries
    """
    fetcher = NASDAQFetcher()

    if count == 500:
        return fetcher.get_nasdaq_500(include_info=include_info)
    elif count == 100:
        if include_info:
            # Get full info for NASDAQ-100
            all_stocks = fetcher.get_all_nasdaq_stocks(include_info=True)
            nasdaq_100_symbols = fetcher.get_nasdaq_100()
            return [s for s in all_stocks if s["symbol"] in nasdaq_100_symbols]
        else:
            return [{"symbol": s} for s in fetcher.get_nasdaq_100()]
    else:
        # Get all stocks and return top N by market cap
        all_stocks = fetcher.get_all_nasdaq_stocks(
            min_market_cap=min_market_cap, include_info=True
        )
        sorted_stocks = sorted(
            [s for s in all_stocks if s.get("market_cap", 0) > 0],
            key=lambda x: x["market_cap"],
            reverse=True,
        )

        result = sorted_stocks[:count]

        if not include_info:
            result = [{"symbol": s["symbol"], "name": s["name"]} for s in result]

        return result


if __name__ == "__main__":
    # Example usage
    print("NASDAQ Stock Fetcher Demo\n")

    fetcher = NASDAQFetcher()

    # Get NASDAQ 500
    print("Fetching NASDAQ 500...")
    nasdaq_500 = fetcher.get_nasdaq_500(include_info=False)
    print(f"Found {len(nasdaq_500)} stocks")
    print(f"First 10: {[s['symbol'] for s in nasdaq_500[:10]]}")

    # Get stocks with market cap filter
    print("\nFetching large-cap NASDAQ stocks (>$50B)...")
    large_caps = fetcher.get_all_nasdaq_stocks(min_market_cap=50e9, include_info=True)
    print(f"Found {len(large_caps)} large-cap stocks")

    if large_caps:
        print("\nTop 5 by market cap:")
        for stock in large_caps[:5]:
            print(
                f"  {stock['symbol']}: {stock['name']} - ${stock['market_cap']/1e9:.1f}B"
            )
