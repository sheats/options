"""
Tests for the StockFilter module.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from modules.stock_filter import StockFilter
from modules import constants


class TestStockFilter:
    """Test suite for StockFilter class."""
    
    def test_init(self, mock_data_provider, mock_cache_provider):
        """Test StockFilter initialization."""
        # Without cache
        filter = StockFilter(mock_data_provider)
        assert filter.data_provider == mock_data_provider
        assert filter.cache_provider is None
        
        # With cache
        filter = StockFilter(mock_data_provider, mock_cache_provider)
        assert filter.data_provider == mock_data_provider
        assert filter.cache_provider == mock_cache_provider
    
    def test_get_quality_stocks_sp500(self, mock_data_provider):
        """Test quality stock filtering for S&P 500."""
        filter = StockFilter(mock_data_provider)
        
        # Mock successful stock info
        mock_data_provider.get_stock_info.return_value = {
            "marketCap": 2e12,
            "trailingPE": 25.0,
            "forwardEps": 5.0,
            "oneYrReturn": 10.0
        }
        
        tickers = ["AAPL", "MSFT", "GOOGL"]
        result = filter.get_quality_stocks(tickers, "SP500")
        
        # All should pass with good metrics
        assert len(result) == 3
        assert result == tickers
        assert mock_data_provider.get_stock_info.call_count == 3
    
    def test_get_quality_stocks_nasdaq(self, mock_data_provider):
        """Test quality stock filtering for NASDAQ with relaxed criteria."""
        filter = StockFilter(mock_data_provider)
        
        # Mock stock info with high P/E that would fail SP500 but pass NASDAQ
        mock_data_provider.get_stock_info.return_value = {
            "marketCap": 2e12,
            "trailingPE": 55.0,  # Above SP500 limit but below NASDAQ limit
            "forwardEps": 5.0,
            "oneYrReturn": -8.0  # Above NASDAQ limit of -10%
        }
        
        tickers = ["NVDA", "TSLA"]
        result = filter.get_quality_stocks(tickers, "NASDAQ")
        
        # Should pass with NASDAQ criteria
        assert len(result) == 2
        assert result == tickers
    
    def test_get_quality_stocks_filtered_out(self, mock_data_provider):
        """Test stocks that fail quality filters."""
        filter = StockFilter(mock_data_provider)
        
        # Mock poor quality metrics
        mock_data_provider.get_stock_info.side_effect = [
            {"marketCap": 5e9, "trailingPE": 25.0, "forwardEps": 5.0, "oneYrReturn": 10.0},  # Low market cap
            {"marketCap": 2e12, "trailingPE": 50.0, "forwardEps": 5.0, "oneYrReturn": 10.0},  # High P/E
            {"marketCap": 2e12, "trailingPE": 25.0, "forwardEps": -1.0, "oneYrReturn": 10.0},  # Negative EPS
            {"marketCap": 2e12, "trailingPE": 25.0, "forwardEps": 5.0, "oneYrReturn": -10.0},  # Poor return
        ]
        
        tickers = ["BAD1", "BAD2", "BAD3", "BAD4"]
        result = filter.get_quality_stocks(tickers, "SP500")
        
        # All should fail
        assert len(result) == 0
        assert mock_data_provider.get_stock_info.call_count == 4
    
    def test_get_quality_stocks_with_cache_hit(self, mock_data_provider, mock_cache_provider):
        """Test quality filtering with full cache hit."""
        filter = StockFilter(mock_data_provider, mock_cache_provider)
        
        # Mock cache hit with all tickers
        cached_data = {
            "AAPL": {"passed_filter": True},
            "MSFT": {"passed_filter": True},
            "GOOGL": {"passed_filter": False}
        }
        mock_cache_provider.get_quality_stocks.return_value = (["AAPL", "MSFT"], cached_data)
        
        tickers = ["AAPL", "MSFT", "GOOGL"]
        result = filter.get_quality_stocks(tickers, "SP500")
        
        # Should use cache and not call data provider
        assert result == ["AAPL", "MSFT"]
        assert mock_data_provider.get_stock_info.call_count == 0
        assert mock_cache_provider.get_quality_stocks.called
    
    def test_get_quality_stocks_with_partial_cache(self, mock_data_provider, mock_cache_provider):
        """Test quality filtering with partial cache hit."""
        filter = StockFilter(mock_data_provider, mock_cache_provider)
        
        # Mock partial cache hit
        cached_data = {
            "AAPL": {"passed_filter": True}
        }
        mock_cache_provider.get_quality_stocks.return_value = (["AAPL"], cached_data)
        
        # Mock stock info for non-cached ticker
        mock_data_provider.get_stock_info.return_value = {
            "marketCap": 2e12,
            "trailingPE": 25.0,
            "forwardEps": 5.0,
            "oneYrReturn": 10.0
        }
        
        tickers = ["AAPL", "MSFT"]
        result = filter.get_quality_stocks(tickers, "SP500")
        
        # Should use cache for AAPL, fetch MSFT
        assert len(result) == 2
        assert "AAPL" in result
        assert "MSFT" in result
        assert mock_data_provider.get_stock_info.call_count == 1  # Only for MSFT
        assert mock_cache_provider.save_quality_stocks.called
    
    def test_get_quality_stocks_exception_handling(self, mock_data_provider):
        """Test exception handling during stock filtering."""
        filter = StockFilter(mock_data_provider)
        
        # Mock exception for one ticker
        mock_data_provider.get_stock_info.side_effect = [
            {"marketCap": 2e12, "trailingPE": 25.0, "forwardEps": 5.0, "oneYrReturn": 10.0},
            Exception("API Error"),
            {"marketCap": 2e12, "trailingPE": 25.0, "forwardEps": 5.0, "oneYrReturn": 10.0},
        ]
        
        tickers = ["AAPL", "BAD", "GOOGL"]
        result = filter.get_quality_stocks(tickers, "SP500")
        
        # Should handle exception and continue
        assert len(result) == 2
        assert "AAPL" in result
        assert "GOOGL" in result
        assert "BAD" not in result
    
    def test_apply_quality_filters(self, mock_data_provider):
        """Test the _apply_quality_filters method."""
        filter = StockFilter(mock_data_provider)
        
        # Test passing all filters
        assert filter._apply_quality_filters(
            "AAPL", 2e12, 25.0, 5.0, 10.0, 40.0, -5.0
        ) is True
        
        # Test failing market cap
        assert filter._apply_quality_filters(
            "SMALL", 5e9, 25.0, 5.0, 10.0, 40.0, -5.0
        ) is False
        
        # Test failing P/E
        assert filter._apply_quality_filters(
            "HIGHPE", 2e12, 50.0, 5.0, 10.0, 40.0, -5.0
        ) is False
        
        # Test failing EPS
        assert filter._apply_quality_filters(
            "NEGEPS", 2e12, 25.0, -1.0, 10.0, 40.0, -5.0
        ) is False
        
        # Test failing return
        assert filter._apply_quality_filters(
            "POORRET", 2e12, 25.0, 5.0, -10.0, 40.0, -5.0
        ) is False
    
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_rate_limiting(self, mock_sleep, mock_data_provider):
        """Test that rate limiting is applied."""
        filter = StockFilter(mock_data_provider)
        
        mock_data_provider.get_stock_info.return_value = {
            "marketCap": 2e12,
            "trailingPE": 25.0,
            "forwardEps": 5.0,
            "oneYrReturn": 10.0
        }
        
        tickers = ["AAPL", "MSFT", "GOOGL"]
        filter.get_quality_stocks(tickers)
        
        # Should sleep between API calls
        assert mock_sleep.call_count == 3
        mock_sleep.assert_called_with(constants.API_SLEEP_TIME)