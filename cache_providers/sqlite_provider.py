"""SQLite implementation of cache provider"""

import datetime
import logging
import sqlite3
from typing import Dict, List, Optional, Tuple

from .base import CacheProvider

logger = logging.getLogger(__name__)


class SQLiteCacheProvider(CacheProvider):
    """SQLite-based cache provider for CSP scanner"""
    
    def __init__(self, db_path: str, cache_lifetime_hours: int = 24):
        """
        Initialize SQLite cache provider
        
        Args:
            db_path: Path to SQLite database file
            cache_lifetime_hours: Default cache lifetime in hours
        """
        self.db_path = db_path
        self.cache_lifetime_hours = cache_lifetime_hours
        self._init_database()
        
    def _init_database(self) -> None:
        """Initialize database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create quality_stocks table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quality_stocks (
                        ticker TEXT,
                        exchange TEXT,
                        market_cap REAL,
                        pe_ratio REAL,
                        forward_eps REAL,
                        one_yr_return REAL,
                        passed_filter INTEGER,
                        max_pe REAL,
                        min_return REAL,
                        last_updated TIMESTAMP,
                        PRIMARY KEY (ticker, exchange, max_pe, min_return)
                    )
                """)
                
                # Create index for faster lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_quality_stocks_lookup 
                    ON quality_stocks(exchange, last_updated)
                """)
                
                conn.commit()
                logger.debug(f"Cache database initialized: {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize cache database: {str(e)}")
            raise
    
    def save_quality_stocks(
        self, 
        ticker: str, 
        exchange: str, 
        stock_data: Dict, 
        passed_filter: bool, 
        filter_criteria: Dict
    ) -> None:
        """Save quality stock data to cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO quality_stocks 
                    (ticker, exchange, market_cap, pe_ratio, forward_eps, one_yr_return, 
                     passed_filter, max_pe, min_return, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    exchange,
                    stock_data.get('marketCap', 0),
                    stock_data.get('trailingPE', float('inf')),
                    stock_data.get('forwardEps', 0),
                    stock_data.get('oneYrReturn', -100),
                    int(passed_filter),
                    filter_criteria.get('max_pe', 40),
                    filter_criteria.get('min_return', -5),
                    datetime.datetime.now()
                ))
                
                conn.commit()
                logger.debug(f"Saved cache entry for {ticker} ({exchange})")
                
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
    
    def get_quality_stocks(
        self, 
        exchange: str, 
        filter_criteria: Dict
    ) -> Optional[Tuple[List[str], Dict[str, Dict]]]:
        """Get cached quality stocks"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get cache cutoff time
                cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=self.cache_lifetime_hours)
                
                # Query cached stocks for this exchange and filter criteria
                cursor.execute("""
                    SELECT ticker, market_cap, pe_ratio, forward_eps, one_yr_return, passed_filter
                    FROM quality_stocks
                    WHERE exchange = ? 
                      AND max_pe = ?
                      AND min_return = ?
                      AND last_updated > ?
                """, (
                    exchange, 
                    filter_criteria.get('max_pe', 40),
                    filter_criteria.get('min_return', -5),
                    cutoff_time
                ))
                
                rows = cursor.fetchall()
                
                if not rows:
                    logger.info(f"No cached data found for {exchange} (PE<={filter_criteria.get('max_pe')}, 1yr>={filter_criteria.get('min_return')})")
                    return None
                
                # Process cached results
                cached_tickers = []
                cached_data = {}
                
                for row in rows:
                    ticker, market_cap, pe_ratio, forward_eps, one_yr_return, passed = row
                    cached_data[ticker] = {
                        'marketCap': market_cap,
                        'trailingPE': pe_ratio,
                        'forwardEps': forward_eps,
                        'oneYrReturn': one_yr_return,
                        'passed_filter': bool(passed)
                    }
                    if passed:
                        cached_tickers.append(ticker)
                
                logger.info(f"Found {len(cached_tickers)} quality stocks in cache for {exchange}")
                return cached_tickers, cached_data
                
        except Exception as e:
            logger.error(f"Error reading from cache: {str(e)}")
            return None
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM quality_stocks")
                conn.commit()
                logger.info("Cache cleared successfully")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    def is_cache_valid(
        self, 
        last_updated: datetime.datetime, 
        lifetime_hours: int
    ) -> bool:
        """Check if cache entry is still valid"""
        age_hours = (datetime.datetime.now() - last_updated).total_seconds() / 3600
        return age_hours < lifetime_hours
    
    def get_cache_stats(self) -> Optional[Dict]:
        """Get cache statistics (additional utility method)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get cache statistics
                cursor.execute("""
                    SELECT exchange, COUNT(*) as count, 
                           SUM(passed_filter) as passed,
                           MAX(last_updated) as newest
                    FROM quality_stocks
                    GROUP BY exchange
                """)
                
                rows = cursor.fetchall()
                if not rows:
                    return None
                    
                stats = {}
                for row in rows:
                    exchange, count, passed, newest = row
                    newest_dt = datetime.datetime.fromisoformat(newest)
                    age_hours = (datetime.datetime.now() - newest_dt).total_seconds() / 3600
                    
                    stats[exchange] = {
                        'total_stocks': count,
                        'passed_filter': passed,
                        'age_hours': age_hours,
                        'is_valid': age_hours < self.cache_lifetime_hours
                    }
                
                return stats
                
        except Exception as e:
            logger.debug(f"Could not get cache stats: {str(e)}")
            return None