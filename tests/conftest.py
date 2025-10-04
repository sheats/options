"""
Pytest configuration and shared fixtures for CSP Scanner tests.
"""

import datetime
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest


@pytest.fixture
def mock_data_provider():
    """Create a mock data provider for testing."""
    provider = MagicMock()
    
    # Mock get_stock_info
    provider.get_stock_info.return_value = {
        "marketCap": 2e12,  # $2T
        "trailingPE": 25.0,
        "forwardEps": 5.0,
        "oneYrReturn": 15.0
    }
    
    # Mock get_historical_data
    dates = pd.date_range(end=datetime.date.today(), periods=252, freq='D')
    provider.get_historical_data.return_value = pd.DataFrame({
        'Open': [100 + i*0.1 for i in range(252)],
        'High': [101 + i*0.1 for i in range(252)],
        'Low': [99 + i*0.1 for i in range(252)],
        'Close': [100 + i*0.1 for i in range(252)],
        'Volume': [1000000] * 252
    }, index=dates)
    
    # Mock get_iv_rank
    provider.get_iv_rank.return_value = 30.0
    
    # Mock get_earnings_dates
    provider.get_earnings_dates.return_value = [
        datetime.date.today() + datetime.timedelta(days=30),
        datetime.date.today() + datetime.timedelta(days=120)
    ]
    
    # Mock get_option_chain
    provider.get_option_chain.return_value = pd.DataFrame({
        'strike': [95, 96, 97, 98, 99, 100, 101, 102],
        'bid': [0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5],
        'ask': [0.6, 0.7, 0.9, 1.1, 1.4, 1.7, 2.1, 2.6],
        'lastPrice': [0.55, 0.65, 0.85, 1.05, 1.35, 1.65, 2.05, 2.55],
        'volume': [100, 150, 200, 300, 500, 600, 400, 200],
        'openInterest': [500, 600, 800, 1000, 1500, 2000, 1200, 800],
        'impliedVolatility': [0.25, 0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18],
        'delta': [-0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.45, -0.50],
        'theta': [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055],
        'gamma': [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
    })
    
    # Mock get_option_expirations
    today = datetime.date.today()
    provider.get_option_expirations.return_value = [
        (today + datetime.timedelta(days=7)).strftime('%Y-%m-%d'),
        (today + datetime.timedelta(days=14)).strftime('%Y-%m-%d'),
        (today + datetime.timedelta(days=21)).strftime('%Y-%m-%d'),
        (today + datetime.timedelta(days=28)).strftime('%Y-%m-%d'),
        (today + datetime.timedelta(days=35)).strftime('%Y-%m-%d'),
    ]
    
    return provider


@pytest.fixture
def mock_cache_provider():
    """Create a mock cache provider for testing."""
    provider = MagicMock()
    
    # Mock get_quality_stocks to return None (cache miss)
    provider.get_quality_stocks.return_value = None
    
    # Mock save_quality_stocks
    provider.save_quality_stocks.return_value = None
    
    # Mock clear_cache
    provider.clear_cache.return_value = None
    
    # Mock is_cache_valid
    provider.is_cache_valid.return_value = True
    
    return provider


@pytest.fixture
def sample_option_chain():
    """Create a sample option chain DataFrame for testing."""
    return pd.DataFrame({
        'strike': [95, 96, 97, 98, 99, 100],
        'bid': [0.5, 0.6, 0.8, 1.0, 1.3, 1.6],
        'ask': [0.6, 0.7, 0.9, 1.1, 1.4, 1.7],
        'lastPrice': [0.55, 0.65, 0.85, 1.05, 1.35, 1.65],
        'volume': [100, 150, 200, 300, 500, 600],
        'openInterest': [500, 600, 800, 1000, 1500, 2000],
        'impliedVolatility': [0.25, 0.24, 0.23, 0.22, 0.21, 0.20],
        'delta': [-0.15, -0.20, -0.25, -0.30, -0.35, -0.40],
        'theta': [0.02, 0.025, 0.03, 0.035, 0.04, 0.045],
        'gamma': [0.01, 0.015, 0.02, 0.025, 0.03, 0.035]
    })


@pytest.fixture
def sample_historical_data():
    """Create sample historical price data for testing."""
    dates = pd.date_range(end=datetime.date.today(), periods=252, freq='D')
    return pd.DataFrame({
        'Open': [100 + i*0.1 for i in range(252)],
        'High': [101 + i*0.1 for i in range(252)],
        'Low': [99 + i*0.1 for i in range(252)],
        'Close': [100 + i*0.1 for i in range(252)],
        'Volume': [1000000] * 252
    }, index=dates)


@pytest.fixture
def sample_stock_info():
    """Create sample stock info for testing."""
    return {
        "marketCap": 2e12,  # $2T
        "trailingPE": 25.0,
        "forwardEps": 5.0,
        "oneYrReturn": 15.0
    }


@pytest.fixture
def sample_supports():
    """Create sample support levels for testing."""
    return {
        "sma_200": 95.0,
        "near_term_support": 98.0,
        "low_52w": 90.0,
        "local_support": 94.0,
        "cluster_support": 93.0,
        "final_support": 92.0,
        "current_price": 100.0
    }