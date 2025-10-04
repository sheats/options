"""
Stock Filter Module

This module handles filtering stocks based on quality metrics.
"""

import logging
import time
from typing import Dict, List, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_providers.base import DataProvider
from cache_providers.base import CacheProvider
from modules import constants


logger = logging.getLogger(__name__)


class StockFilter:
    """
    Filters stocks based on quality metrics with support for different exchanges.
    
    This class handles the filtering of stocks based on fundamental metrics
    such as market cap, P/E ratio, forward EPS, and 1-year return.
    """
    
    def __init__(
        self, 
        data_provider: DataProvider, 
        cache_provider: Optional[CacheProvider] = None
    ):
        """
        Initialize the StockFilter.
        
        Args:
            data_provider: The data provider for fetching stock information
            cache_provider: Optional cache provider for storing/retrieving results
        """
        self.data_provider = data_provider
        self.cache_provider = cache_provider
    
    def get_quality_stocks(
        self, 
        tickers: List[str], 
        exchange: str = "SP500"
    ) -> List[str]:
        """
        Filter stocks based on quality metrics with relaxed criteria for NASDAQ.
        
        Args:
            tickers: List of stock ticker symbols to filter
            exchange: Exchange name ('SP500' or 'NASDAQ')
            
        Returns:
            List of ticker symbols that passed quality filters
        """
        quality_stocks = []
        
        # Determine filter criteria based on exchange
        if exchange == "NASDAQ":
            max_pe = constants.NASDAQ_MAX_PE_RATIO
            min_return = constants.NASDAQ_MIN_ONE_YEAR_RETURN
            logger.info(f"Using relaxed quality filters for NASDAQ: P/E<={max_pe}, 1yr>={min_return}%")
        else:
            max_pe = constants.MAX_PE_RATIO
            min_return = constants.MIN_ONE_YEAR_RETURN
            
        # Build filter criteria dict for caching
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
        
        for ticker in tickers:
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
                
                # Extract metrics
                market_cap = info.get("marketCap", 0)
                pe_ratio = info.get("trailingPE", float("inf"))
                forward_eps = info.get("forwardEps", 0)
                one_yr_return = info.get("oneYrReturn", -100)
                
                logger.debug(
                    f"{ticker} metrics: MCap=${market_cap/1e9:.1f}B, PE={pe_ratio:.1f}, "
                    f"FwdEPS=${forward_eps:.2f}, 1yr={one_yr_return:.1f}%"
                )
                
                # Apply filters
                passed = self._apply_quality_filters(
                    ticker, market_cap, pe_ratio, forward_eps, one_yr_return, max_pe, min_return
                )
                
                # Save to cache if provider available
                if self.cache_provider:
                    self.cache_provider.save_quality_stocks(
                        ticker, exchange, info, passed, filter_criteria
                    )
                
                if passed:
                    quality_stocks.append(ticker)
                    logger.info(f"✓ {ticker} passed quality filters")
                
                time.sleep(constants.API_SLEEP_TIME)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error processing {ticker}: {str(e)}")
        
        logger.info(
            f"Quality filter complete: {len(quality_stocks)}/{len(tickers)} stocks passed"
        )
        return quality_stocks
    
    def _apply_quality_filters(
        self,
        ticker: str,
        market_cap: float,
        pe_ratio: float,
        forward_eps: float,
        one_yr_return: float,
        max_pe: float,
        min_return: float
    ) -> bool:
        """
        Apply quality filters to a stock.
        
        Args:
            ticker: Stock ticker symbol
            market_cap: Market capitalization
            pe_ratio: Trailing P/E ratio
            forward_eps: Forward earnings per share
            one_yr_return: One-year return percentage
            max_pe: Maximum allowed P/E ratio
            min_return: Minimum required 1-year return
            
        Returns:
            True if stock passes all filters, False otherwise
        """
        passed = True
        reasons = []
        
        if market_cap <= constants.MIN_MARKET_CAP:
            passed = False
            reasons.append(f"MCap ${market_cap/1e9:.1f}B <= ${constants.MIN_MARKET_CAP/1e9:.0f}B")
        
        if pe_ratio > max_pe:
            passed = False
            reasons.append(f"PE {pe_ratio:.1f} > {max_pe}")
        
        if forward_eps <= constants.MIN_FORWARD_EPS:
            passed = False
            reasons.append(f"FwdEPS ${forward_eps:.2f} <= ${constants.MIN_FORWARD_EPS}")
        
        if one_yr_return < min_return:
            passed = False
            reasons.append(f"1yr return {one_yr_return:.1f}% < {min_return}%")
        
        if not passed:
            logger.debug(f"✗ {ticker} failed: {', '.join(reasons)}")
        
        return passed