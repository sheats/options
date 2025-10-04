"""
Tests for the SupportCalculator module.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from modules.support_calculator import SupportCalculator
from modules import constants


class TestSupportCalculator:
    """Test suite for SupportCalculator class."""
    
    def test_init(self, mock_data_provider):
        """Test SupportCalculator initialization."""
        # Default buffer
        calc = SupportCalculator(mock_data_provider)
        assert calc.data_provider == mock_data_provider
        assert calc.support_buffer == constants.DEFAULT_SUPPORT_BUFFER
        
        # Custom buffer
        calc = SupportCalculator(mock_data_provider, support_buffer=0.05)
        assert calc.support_buffer == 0.05
    
    def test_calculate_support_levels_success(self, mock_data_provider, sample_historical_data):
        """Test successful support level calculation."""
        calc = SupportCalculator(mock_data_provider)
        mock_data_provider.get_historical_data.return_value = sample_historical_data
        
        result = calc.calculate_support_levels("AAPL")
        
        # Check all expected keys
        assert "sma_200" in result
        assert "near_term_support" in result
        assert "low_52w" in result
        assert "local_support" in result
        assert "cluster_support" in result
        assert "final_support" in result
        assert "current_price" in result
        
        # Verify calculations
        assert result["current_price"] == sample_historical_data["Close"].iloc[-1]
        assert result["low_52w"] == sample_historical_data["Low"].min()
        assert result["sma_200"] == sample_historical_data["Close"].rolling(200).mean().iloc[-1]
        assert result["near_term_support"] == sample_historical_data["Close"].rolling(50).mean().iloc[-1]
        
        # Final support should be less than components due to buffer
        assert result["final_support"] < result["sma_200"]
        assert result["final_support"] < result["current_price"]
    
    def test_calculate_support_levels_insufficient_data(self, mock_data_provider):
        """Test handling of insufficient historical data."""
        calc = SupportCalculator(mock_data_provider)
        
        # Mock insufficient data (less than MIN_HISTORY_DAYS)
        short_data = pd.DataFrame({
            'Close': [100] * 50,
            'Low': [99] * 50
        })
        mock_data_provider.get_historical_data.return_value = short_data
        
        result = calc.calculate_support_levels("AAPL")
        
        # Should return empty dict
        assert result == {}
    
    def test_calculate_support_levels_exception(self, mock_data_provider):
        """Test exception handling in support calculation."""
        calc = SupportCalculator(mock_data_provider)
        
        # Mock data provider throwing exception
        mock_data_provider.get_historical_data.side_effect = Exception("API Error")
        
        result = calc.calculate_support_levels("AAPL")
        
        # Should return empty dict
        assert result == {}
    
    def test_calculate_local_support_with_minima(self, mock_data_provider):
        """Test local support calculation with detected minima."""
        calc = SupportCalculator(mock_data_provider)
        
        # Create data with clear local minima
        prices = np.array([100, 95, 90, 95, 100, 95, 90, 95, 100])
        
        result = calc._calculate_local_support(prices, 90.0)
        
        # Should find local minima and average them
        assert result > 0
        assert result <= 95  # Should be around the minima values
    
    def test_calculate_local_support_no_minima(self, mock_data_provider):
        """Test local support calculation with no minima."""
        calc = SupportCalculator(mock_data_provider, support_buffer=0.02)
        
        # Monotonically increasing prices (no local minima)
        prices = np.array([90 + i for i in range(100)])
        
        result = calc._calculate_local_support(prices, 90.0)
        
        # Should fall back to 52w low with buffer
        expected = 90.0 * (1 - 0.02)
        assert result == expected
    
    def test_calculate_kmeans_support_success(self, mock_data_provider):
        """Test K-means clustering support calculation."""
        calc = SupportCalculator(mock_data_provider)
        
        # Create data with clear clusters
        closes = np.array([90] * 30 + [95] * 30 + [100] * 30)
        
        result = calc._calculate_kmeans_support(closes, "AAPL", 90.0)
        
        # Should find lowest cluster around 90
        assert 89 <= result <= 91
    
    def test_calculate_kmeans_support_exception(self, mock_data_provider):
        """Test K-means clustering with exception."""
        calc = SupportCalculator(mock_data_provider)
        
        # Create data that might cause clustering issues
        closes = np.array([100])  # Single data point
        
        result = calc._calculate_kmeans_support(closes, "AAPL", 90.0)
        
        # Should fall back to 52w low
        assert result == 90.0
    
    def test_support_buffer_application(self, mock_data_provider, sample_historical_data):
        """Test that support buffer is correctly applied."""
        # Test with different buffers
        for buffer in [0.01, 0.02, 0.05]:
            calc = SupportCalculator(mock_data_provider, support_buffer=buffer)
            mock_data_provider.get_historical_data.return_value = sample_historical_data
            
            result = calc.calculate_support_levels("AAPL")
            
            # Calculate expected final support
            avg_support = np.mean([
                result["sma_200"],
                result["local_support"],
                result["cluster_support"]
            ])
            expected_final = avg_support * (1 - buffer)
            
            # Allow small floating point difference
            assert abs(result["final_support"] - expected_final) < 0.01
    
    def test_log_support_levels(self, mock_data_provider, caplog):
        """Test that support levels are properly logged."""
        calc = SupportCalculator(mock_data_provider)
        
        # Enable debug logging
        import logging
        caplog.set_level(logging.DEBUG)
        
        calc._log_support_levels(
            "AAPL", 95.0, 98.0, 2.0, 90.0, 94.0, 93.0, 92.0, 100.0, 8.0
        )
        
        # Check that all support levels are logged
        assert "AAPL Support Levels:" in caplog.text
        assert "200-day SMA: $95.00" in caplog.text
        assert "50-day SMA (near-term): $98.00" in caplog.text
        assert "52-week low: $90.00" in caplog.text
        assert "Local support: $94.00" in caplog.text
        assert "Cluster support: $93.00" in caplog.text
        assert "Final support" in caplog.text
        assert "Current price: $100.00" in caplog.text
    
    def test_edge_cases(self, mock_data_provider):
        """Test edge cases in support calculation."""
        calc = SupportCalculator(mock_data_provider)
        
        # Test with all same prices
        flat_data = pd.DataFrame({
            'Open': [100] * 252,
            'High': [100] * 252,
            'Low': [100] * 252,
            'Close': [100] * 252,
            'Volume': [1000000] * 252
        })
        mock_data_provider.get_historical_data.return_value = flat_data
        
        result = calc.calculate_support_levels("FLAT")
        
        # All supports should be around 100
        assert 95 <= result["sma_200"] <= 100
        assert 95 <= result["near_term_support"] <= 100
        assert result["low_52w"] == 100
        assert result["current_price"] == 100