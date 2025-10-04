"""Abstract base class for cache providers"""

import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class CacheProvider(ABC):
    """Abstract base class for cache providers"""

    @abstractmethod
    def save_quality_stocks(
        self, 
        ticker: str, 
        exchange: str, 
        stock_data: Dict, 
        passed_filter: bool, 
        filter_criteria: Dict
    ) -> None:
        """
        Save quality stock data to cache
        
        Args:
            ticker: Stock ticker symbol
            exchange: Exchange name (e.g., 'SP500', 'NASDAQ')
            stock_data: Dictionary containing stock metrics (marketCap, trailingPE, forwardEps, oneYrReturn)
            passed_filter: Whether the stock passed quality filters
            filter_criteria: Dictionary containing filter criteria (max_pe, min_return)
        """
        pass

    @abstractmethod
    def get_quality_stocks(
        self, 
        exchange: str, 
        filter_criteria: Dict
    ) -> Optional[Tuple[List[str], Dict[str, Dict]]]:
        """
        Get cached quality stocks for the given exchange and filter criteria
        
        Args:
            exchange: Exchange name (e.g., 'SP500', 'NASDAQ')
            filter_criteria: Dictionary containing filter criteria (max_pe, min_return)
            
        Returns:
            Tuple of (list of tickers that passed filter, dict of all cached data)
            Returns None if cache miss or expired
        """
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear all cached data"""
        pass

    @abstractmethod
    def is_cache_valid(
        self, 
        last_updated: datetime.datetime, 
        lifetime_hours: int
    ) -> bool:
        """
        Check if cache entry is still valid based on age
        
        Args:
            last_updated: Timestamp of last update
            lifetime_hours: Cache lifetime in hours
            
        Returns:
            True if cache is still valid, False otherwise
        """
        pass
    
    @abstractmethod
    def save_ticker_data(self, ticker: str, exchange: str, data: Dict) -> None:
        """
        Save comprehensive ticker data to cache
        
        Args:
            ticker: Stock ticker symbol
            exchange: Exchange name
            data: Dictionary containing all ticker data
        """
        pass
    
    @abstractmethod
    def get_ticker_data(self, ticker: str, max_age_hours: Optional[int] = None) -> Optional[Dict]:
        """
        Get cached ticker data
        
        Args:
            ticker: Stock ticker symbol
            max_age_hours: Maximum age in hours (uses default if None)
            
        Returns:
            Dictionary with ticker data or None if not found/expired
        """
        pass
    
    @abstractmethod
    def get_cached_tickers(self, exchange: Optional[str] = None) -> List[str]:
        """
        Get list of cached tickers
        
        Args:
            exchange: Filter by exchange (optional)
            
        Returns:
            List of ticker symbols
        """
        pass
    
    @abstractmethod
    def clear_ticker_data(self, exchange: Optional[str] = None) -> None:
        """
        Clear ticker data cache
        
        Args:
            exchange: Clear only specific exchange (optional)
        """
        pass