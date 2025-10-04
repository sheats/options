#!/usr/bin/env python3
"""Test script for the cache provider architecture"""

import logging
import os
import tempfile
from cache_providers import SQLiteCacheProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_cache_provider():
    """Test the cache provider functionality"""
    
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Initialize cache provider
        cache = SQLiteCacheProvider(db_path, cache_lifetime_hours=24)
        logger.info(f"Created cache provider with db: {db_path}")
        
        # Test data
        ticker = "AAPL"
        exchange = "SP500"
        stock_data = {
            'marketCap': 3e12,
            'trailingPE': 28.5,
            'forwardEps': 6.5,
            'oneYrReturn': 35.2
        }
        filter_criteria = {
            'max_pe': 40,
            'min_return': -5
        }
        
        # Save to cache
        logger.info(f"Saving {ticker} to cache...")
        cache.save_quality_stocks(ticker, exchange, stock_data, True, filter_criteria)
        
        # Also save a failed stock
        cache.save_quality_stocks("XYZ", exchange, 
                                {'marketCap': 1e9, 'trailingPE': 50}, 
                                False, filter_criteria)
        
        # Retrieve from cache
        logger.info("Retrieving from cache...")
        result = cache.get_quality_stocks(exchange, filter_criteria)
        
        if result:
            passed_tickers, all_data = result
            logger.info(f"Passed tickers: {passed_tickers}")
            logger.info(f"All cached data: {all_data}")
        else:
            logger.info("No data in cache")
        
        # Get cache statistics
        stats = cache.get_cache_stats()
        if stats:
            logger.info(f"Cache statistics: {stats}")
        
        # Test cache miss with different criteria
        logger.info("\nTesting cache miss with different criteria...")
        different_criteria = {'max_pe': 30, 'min_return': 0}
        result2 = cache.get_quality_stocks(exchange, different_criteria)
        if result2:
            logger.info("Unexpected: Found data with different criteria")
        else:
            logger.info("Expected: No data found with different criteria")
        
        # Clear cache
        logger.info("\nClearing cache...")
        cache.clear_cache()
        
        # Verify cache is empty
        result3 = cache.get_quality_stocks(exchange, filter_criteria)
        if result3:
            logger.info("Unexpected: Found data after clearing")
        else:
            logger.info("Expected: Cache is empty after clearing")
            
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info(f"Cleaned up temporary database: {db_path}")

if __name__ == "__main__":
    logger.info("Starting cache provider test...")
    test_cache_provider()
    logger.info("Test completed successfully!")