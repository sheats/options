"""
Tests for the OptionAnalyzer module.
"""

import datetime
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from modules.option_analyzer import OptionAnalyzer
from modules import constants


class TestOptionAnalyzer:
    """Test suite for OptionAnalyzer class."""
    
    def test_init(self, mock_data_provider):
        """Test OptionAnalyzer initialization."""
        # Default parameters
        analyzer = OptionAnalyzer(mock_data_provider)
        assert analyzer.data_provider == mock_data_provider
        assert analyzer.max_weeks == constants.DEFAULT_MAX_WEEKS
        assert analyzer.min_iv_rank == constants.DEFAULT_MIN_IV_RANK
        assert analyzer.support_buffer == constants.DEFAULT_SUPPORT_BUFFER
        assert analyzer.no_support_filter is False
        assert analyzer.today == datetime.date.today()
        
        # Custom parameters
        custom_date = datetime.date(2025, 1, 1)
        analyzer = OptionAnalyzer(
            mock_data_provider,
            max_weeks=12,
            min_iv_rank=30.0,
            support_buffer=0.05,
            no_support_filter=True,
            scan_date=custom_date
        )
        assert analyzer.max_weeks == 12
        assert analyzer.min_iv_rank == 30.0
        assert analyzer.support_buffer == 0.05
        assert analyzer.no_support_filter is True
        assert analyzer.today == custom_date
    
    def test_check_earnings_in_period_true(self, mock_data_provider):
        """Test earnings check when earnings fall within period."""
        analyzer = OptionAnalyzer(mock_data_provider)
        
        exp_date = datetime.date.today() + datetime.timedelta(days=10)
        earnings_date = exp_date - datetime.timedelta(days=5)  # 5 days before expiration
        
        mock_data_provider.get_earnings_dates.return_value = [earnings_date]
        
        result = analyzer.check_earnings_in_period("AAPL", exp_date)
        assert result is True
    
    def test_check_earnings_in_period_false(self, mock_data_provider):
        """Test earnings check when no earnings in period."""
        analyzer = OptionAnalyzer(mock_data_provider)
        
        exp_date = datetime.date.today() + datetime.timedelta(days=10)
        earnings_date = exp_date - datetime.timedelta(days=10)  # 10 days before expiration
        
        mock_data_provider.get_earnings_dates.return_value = [earnings_date]
        
        result = analyzer.check_earnings_in_period("AAPL", exp_date)
        assert result is False
    
    def test_check_earnings_in_period_exception(self, mock_data_provider):
        """Test earnings check with exception."""
        analyzer = OptionAnalyzer(mock_data_provider)
        
        mock_data_provider.get_earnings_dates.side_effect = Exception("API Error")
        
        exp_date = datetime.date.today() + datetime.timedelta(days=10)
        result = analyzer.check_earnings_in_period("AAPL", exp_date)
        
        # Should return False on exception
        assert result is False
    
    def test_analyze_option_chain_success(self, mock_data_provider, sample_option_chain, sample_supports):
        """Test successful option chain analysis."""
        analyzer = OptionAnalyzer(mock_data_provider)
        mock_data_provider.get_option_chain.return_value = sample_option_chain
        mock_data_provider.get_earnings_dates.return_value = []  # No earnings
        
        exp_date = datetime.date.today() + datetime.timedelta(days=30)
        exp_str = exp_date.strftime('%Y-%m-%d')
        
        opportunities = analyzer.analyze_option_chain(
            "AAPL", exp_str, exp_date, 100.0, sample_supports, 30.0
        )
        
        # Should find some opportunities
        assert len(opportunities) > 0
        
        # Check opportunity structure
        opp = opportunities[0]
        assert "ticker" in opp
        assert "expiration" in opp
        assert "days_to_exp" in opp
        assert "strike" in opp
        assert "premium" in opp
        assert "delta" in opp
        assert "theta" in opp
        assert "gamma" in opp
        assert "daily_premium" in opp
        assert "annualized_roc" in opp
        assert "pop" in opp
        assert "score" in opp
    
    def test_analyze_option_chain_past_expiration(self, mock_data_provider, sample_supports):
        """Test option chain analysis with past expiration date."""
        analyzer = OptionAnalyzer(mock_data_provider)
        
        exp_date = datetime.date.today() - datetime.timedelta(days=1)  # Yesterday
        exp_str = exp_date.strftime('%Y-%m-%d')
        
        opportunities = analyzer.analyze_option_chain(
            "AAPL", exp_str, exp_date, 100.0, sample_supports, 30.0
        )
        
        # Should return empty list
        assert opportunities == []
    
    def test_analyze_option_chain_with_earnings(self, mock_data_provider, sample_supports):
        """Test option chain analysis when earnings within period."""
        analyzer = OptionAnalyzer(mock_data_provider)
        
        exp_date = datetime.date.today() + datetime.timedelta(days=30)
        exp_str = exp_date.strftime('%Y-%m-%d')
        earnings_date = exp_date - datetime.timedelta(days=5)
        
        mock_data_provider.get_earnings_dates.return_value = [earnings_date]
        
        opportunities = analyzer.analyze_option_chain(
            "AAPL", exp_str, exp_date, 100.0, sample_supports, 30.0
        )
        
        # Should skip due to earnings
        assert opportunities == []
    
    def test_analyze_option_chain_empty_chain(self, mock_data_provider, sample_supports):
        """Test option chain analysis with empty chain."""
        analyzer = OptionAnalyzer(mock_data_provider)
        
        mock_data_provider.get_option_chain.return_value = pd.DataFrame()
        mock_data_provider.get_earnings_dates.return_value = []
        
        exp_date = datetime.date.today() + datetime.timedelta(days=30)
        exp_str = exp_date.strftime('%Y-%m-%d')
        
        opportunities = analyzer.analyze_option_chain(
            "AAPL", exp_str, exp_date, 100.0, sample_supports, 30.0
        )
        
        # Should return empty list
        assert opportunities == []
    
    def test_apply_option_filters_volume_oi(self, mock_data_provider):
        """Test volume and open interest filtering."""
        analyzer = OptionAnalyzer(mock_data_provider)
        
        puts = pd.DataFrame({
            'strike': [95, 96, 97],
            'volume': [10, 100, 200],  # First one below MIN_VOLUME
            'openInterest': [500, 15, 1000],  # Second one below MIN_OPEN_INTEREST
            'delta': [-0.3, -0.3, -0.3],
            'theta': [0.03, 0.03, 0.03],
            'gamma': [0.02, 0.02, 0.02]
        })
        
        supports = {'final_support': 90.0, 'near_term_support': 92.0}
        
        result = analyzer._apply_option_filters(puts, "AAPL", "2025-01-30", 100.0, supports)
        
        # Only the third option should pass
        assert len(result) == 1
        assert result.iloc[0]['strike'] == 97
    
    def test_apply_option_filters_support_filter(self, mock_data_provider):
        """Test support level filtering."""
        analyzer = OptionAnalyzer(mock_data_provider, support_buffer=0.02)
        
        puts = pd.DataFrame({
            'strike': [85, 92, 95, 98, 102],  # Various strikes relative to support
            'volume': [100] * 5,
            'openInterest': [100] * 5,
            'delta': [-0.3] * 5,
            'theta': [0.03] * 5,
            'gamma': [0.02] * 5
        })
        
        supports = {'final_support': 90.0, 'near_term_support': 94.0}
        current_price = 100.0
        
        result = analyzer._apply_option_filters(puts, "AAPL", "2025-01-30", current_price, supports)
        
        # Check which strikes pass support filter
        # max_strike = 100 * 0.98 = 98, min_strike = 90 * 0.9 = 81
        # effective_support = max(94 * 0.98, 90) = 92.12
        # Should include strikes between 81 and 92.12
        assert len(result) > 0
        assert all(result['strike'] <= 92.12)
    
    def test_apply_option_filters_no_support_filter(self, mock_data_provider):
        """Test filtering with support filter disabled."""
        analyzer = OptionAnalyzer(mock_data_provider, no_support_filter=True)
        
        puts = pd.DataFrame({
            'strike': [85, 92, 95, 98, 102],
            'volume': [100] * 5,
            'openInterest': [100] * 5,
            'delta': [-0.3] * 5,
            'theta': [0.03] * 5,
            'gamma': [0.02] * 5
        })
        
        supports = {'final_support': 90.0, 'near_term_support': 94.0}
        
        result = analyzer._apply_option_filters(puts, "AAPL", "2025-01-30", 100.0, supports)
        
        # All should pass (no support filter)
        assert len(result) == 5
    
    def test_apply_option_filters_greeks(self, mock_data_provider):
        """Test Greeks-based filtering."""
        analyzer = OptionAnalyzer(mock_data_provider, no_support_filter=True)
        
        puts = pd.DataFrame({
            'strike': [95, 96, 97, 98, 99],
            'volume': [100] * 5,
            'openInterest': [100] * 5,
            'delta': [-0.01, -0.2, -0.3, -0.5, -0.6],  # First too high, last too low
            'theta': [0.005, 0.02, 0.03, 0.04, 0.05],  # First too low
            'gamma': [0.01, 0.02, 0.03, 0.04, 0.06]  # Last too high
        })
        
        supports = {'final_support': 90.0, 'near_term_support': 94.0}
        
        result = analyzer._apply_option_filters(puts, "AAPL", "2025-01-30", 100.0, supports)
        
        # Only middle options should pass all Greek filters
        assert len(result) == 2
        assert 96 in result['strike'].values
        assert 97 in result['strike'].values
    
    def test_create_opportunity_success(self, mock_data_provider):
        """Test successful opportunity creation."""
        analyzer = OptionAnalyzer(mock_data_provider)
        
        put = pd.Series({
            'strike': 95.0,
            'bid': 1.0,
            'ask': 1.2,
            'lastPrice': 1.1,
            'delta': -0.3,
            'theta': 0.03,
            'gamma': 0.02,
            'impliedVolatility': 0.25,
            'volume': 100,
            'openInterest': 500
        })
        
        opp = analyzer._create_opportunity(
            "AAPL", "2025-01-30", 30, put, 90.0, 92.0, 100.0, 30.0
        )
        
        # Check opportunity structure
        assert opp is not None
        assert opp["ticker"] == "AAPL"
        assert opp["expiration"] == "2025-01-30"
        assert opp["days_to_exp"] == 30
        assert opp["strike"] == 95.0
        assert opp["premium"] == 1.1  # (bid + ask) / 2
        assert opp["delta"] == -0.3
        assert opp["pop"] == 70.0  # (1 + delta) * 100
        assert opp["daily_premium"] > 0
        assert opp["annualized_roc"] > 0
        assert opp["score"] > 0
    
    def test_create_opportunity_zero_premium(self, mock_data_provider):
        """Test opportunity creation with zero premium."""
        analyzer = OptionAnalyzer(mock_data_provider)
        
        put = pd.Series({
            'strike': 95.0,
            'bid': 0.0,
            'ask': 0.0,
            'lastPrice': 0.0,
            'delta': -0.3,
            'theta': 0.03,
            'gamma': 0.02,
            'impliedVolatility': 0.25,
            'volume': 100,
            'openInterest': 500
        })
        
        opp = analyzer._create_opportunity(
            "AAPL", "2025-01-30", 30, put, 90.0, 92.0, 100.0, 30.0
        )
        
        # Should return None for zero premium
        assert opp is None
    
    def test_create_opportunity_filter_by_minimums(self, mock_data_provider):
        """Test opportunity creation with minimum filters."""
        analyzer = OptionAnalyzer(mock_data_provider)
        
        # Create put with very low premium
        put = pd.Series({
            'strike': 50.0,  # Low strike
            'bid': 0.01,
            'ask': 0.03,
            'lastPrice': 0.02,
            'delta': -0.05,
            'theta': 0.001,
            'gamma': 0.001,
            'impliedVolatility': 0.1,
            'volume': 100,
            'openInterest': 500
        })
        
        opp = analyzer._create_opportunity(
            "AAPL", "2025-01-30", 30, put, 90.0, 92.0, 100.0, 30.0
        )
        
        # Should return None due to low daily premium and ROC
        assert opp is None