"""
Tests for the Scoring module.
"""

import pandas as pd
import pytest

from modules.scoring import ScoringEngine
from modules import constants


class TestScoringEngine:
    """Test suite for ScoringEngine class."""
    
    def test_init(self):
        """Test ScoringEngine initialization."""
        engine = ScoringEngine()
        assert engine is not None
    
    def test_calculate_score(self):
        """Test score calculation."""
        engine = ScoringEngine()
        
        # Test basic score calculation
        score = engine.calculate_score(
            daily_premium=0.10,
            probability_of_profit=0.7,
            delta=-0.3,
            implied_volatility=0.25,
            strike=95.0,
            final_support=90.0,
            current_price=100.0
        )
        
        assert score > 0
        
        # Test that higher daily premium increases score
        score_high_premium = engine.calculate_score(
            daily_premium=0.20,  # Higher premium
            probability_of_profit=0.7,
            delta=-0.3,
            implied_volatility=0.25,
            strike=95.0,
            final_support=90.0,
            current_price=100.0
        )
        assert score_high_premium > score
        
        # Test that higher probability increases score
        score_high_pop = engine.calculate_score(
            daily_premium=0.10,
            probability_of_profit=0.9,  # Higher PoP
            delta=-0.3,
            implied_volatility=0.25,
            strike=95.0,
            final_support=90.0,
            current_price=100.0
        )
        assert score_high_pop > score
        
        # Test that lower risk (lower delta*IV) increases score
        score_low_risk = engine.calculate_score(
            daily_premium=0.10,
            probability_of_profit=0.7,
            delta=-0.1,  # Lower delta (less negative)
            implied_volatility=0.25,
            strike=95.0,
            final_support=90.0,
            current_price=100.0
        )
        assert score_low_risk > score
        
        # Test that proximity to support increases score
        score_near_support = engine.calculate_score(
            daily_premium=0.10,
            probability_of_profit=0.7,
            delta=-0.3,
            implied_volatility=0.25,
            strike=91.0,  # Closer to support
            final_support=90.0,
            current_price=100.0
        )
        assert score_near_support > score
    
    def test_rank_opportunities_empty(self):
        """Test ranking empty opportunities list."""
        engine = ScoringEngine()
        
        result = engine.rank_opportunities([])
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_rank_opportunities_single(self):
        """Test ranking single opportunity."""
        engine = ScoringEngine()
        
        opportunities = [{
            "ticker": "AAPL",
            "expiration": "2025-01-30",
            "days_to_exp": 30,
            "strike": 95.0,
            "premium": 1.5,
            "delta": -0.3,
            "theta": 0.03,
            "gamma": 0.02,
            "daily_premium": 0.05,
            "annualized_roc": 6.0,
            "pop": 70.0,
            "support": 90.0,
            "near_term_support": 92.0,
            "current_price": 100.0,
            "iv_rank": 30.0,
            "score": 1.5,
            "volume": 100,
            "open_interest": 500
        }]
        
        result = engine.rank_opportunities(opportunities)
        
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"
        assert "max_contracts" in result.columns
        assert result.iloc[0]["max_contracts"] > 0
    
    def test_rank_opportunities_multiple(self):
        """Test ranking multiple opportunities."""
        engine = ScoringEngine()
        
        opportunities = [
            {
                "ticker": "AAPL",
                "expiration": "2025-01-30",
                "days_to_exp": 30,
                "strike": 95.0,
                "premium": 1.5,
                "delta": -0.3,
                "theta": 0.03,
                "gamma": 0.02,
                "daily_premium": 0.05,
                "annualized_roc": 6.0,
                "pop": 70.0,
                "support": 90.0,
                "near_term_support": 92.0,
                "current_price": 100.0,
                "iv_rank": 30.0,
                "score": 1.5,
                "volume": 100,
                "open_interest": 500
            },
            {
                "ticker": "MSFT",
                "expiration": "2025-01-30",
                "days_to_exp": 30,
                "strike": 190.0,
                "premium": 3.0,
                "delta": -0.25,
                "theta": 0.04,
                "gamma": 0.015,
                "daily_premium": 0.10,
                "annualized_roc": 7.0,
                "pop": 75.0,
                "support": 185.0,
                "near_term_support": 187.0,
                "current_price": 200.0,
                "iv_rank": 25.0,
                "score": 2.0,  # Higher score
                "volume": 200,
                "open_interest": 1000
            },
            {
                "ticker": "GOOGL",
                "expiration": "2025-01-30",
                "days_to_exp": 30,
                "strike": 140.0,
                "premium": 2.0,
                "delta": -0.35,
                "theta": 0.035,
                "gamma": 0.025,
                "daily_premium": 0.067,
                "annualized_roc": 5.5,
                "pop": 65.0,
                "support": 135.0,
                "near_term_support": 137.0,
                "current_price": 150.0,
                "iv_rank": 35.0,
                "score": 1.2,  # Lower score
                "volume": 150,
                "open_interest": 750
            }
        ]
        
        result = engine.rank_opportunities(opportunities)
        
        # Should be sorted by score descending
        assert len(result) == 3
        assert result.iloc[0]["ticker"] == "MSFT"  # Highest score
        assert result.iloc[1]["ticker"] == "AAPL"  # Middle score
        assert result.iloc[2]["ticker"] == "GOOGL"  # Lowest score
        
        # Check max contracts calculation
        assert all(result["max_contracts"] > 0)
    
    def test_calculate_max_contracts(self):
        """Test max contracts calculation."""
        engine = ScoringEngine()
        
        strikes = pd.Series([50.0, 100.0, 200.0, 500.0])
        
        max_contracts = engine._calculate_max_contracts(strikes)
        
        # With ACCOUNT_SIZE=100000 and MAX_POSITION_PCT=0.05
        # Max position = 5000
        assert max_contracts.iloc[0] == 10  # 5000 / (50 * 100)
        assert max_contracts.iloc[1] == 5   # 5000 / (100 * 100)
        assert max_contracts.iloc[2] == 2   # 5000 / (200 * 100)
        assert max_contracts.iloc[3] == 1   # 5000 / (500 * 100)
    
    def test_format_results_empty(self):
        """Test formatting empty results."""
        engine = ScoringEngine()
        
        df = pd.DataFrame()
        result = engine.format_results(df)
        
        assert result == "No CSP opportunities found matching criteria"
    
    def test_format_results_single(self):
        """Test formatting single result."""
        engine = ScoringEngine()
        
        df = pd.DataFrame([{
            "ticker": "AAPL",
            "expiration": "2025-01-30",
            "days_to_exp": 30,
            "strike": 95.0,
            "premium": 1.5,
            "delta": -0.3,
            "theta": 0.03,
            "gamma": 0.02,
            "daily_premium": 0.05,
            "annualized_roc": 6.0,
            "pop": 70.0,
            "support": 90.0,
            "near_term_support": 92.0,
            "current_price": 100.0,
            "iv_rank": 30.0,
            "score": 1.5,
            "max_contracts": 5
        }])
        
        result = engine.format_results(df)
        
        # Check formatting
        assert "AAPL" in result
        assert "$95.00" in result  # Strike formatted
        assert "$1.50" in result   # Premium formatted
        assert "-0.300" in result  # Delta formatted
        assert "6.0%" in result    # ROC formatted
        assert "70.0%" in result   # PoP formatted
        assert "$90.00" in result  # Support formatted
        assert "$100.00" in result # Current price formatted
        assert "30.0%" in result   # IV rank formatted
        assert "1.50" in result    # Score formatted
    
    def test_format_results_top_n(self):
        """Test formatting with top N limit."""
        engine = ScoringEngine()
        
        # Create 25 opportunities
        opportunities = []
        for i in range(25):
            opportunities.append({
                "ticker": f"TICK{i}",
                "expiration": "2025-01-30",
                "days_to_exp": 30,
                "strike": 95.0 + i,
                "premium": 1.5,
                "delta": -0.3,
                "theta": 0.03,
                "gamma": 0.02,
                "daily_premium": 0.05,
                "annualized_roc": 6.0,
                "pop": 70.0,
                "support": 90.0,
                "near_term_support": 92.0,
                "current_price": 100.0,
                "iv_rank": 30.0,
                "score": 1.5,
                "max_contracts": 5
            })
        
        df = pd.DataFrame(opportunities)
        
        # Format with default top 20
        result = engine.format_results(df)
        
        # Should include first 20 tickers
        for i in range(20):
            assert f"TICK{i}" in result
        
        # Should not include 21st ticker
        assert "TICK20" not in result
        
        # Format with custom top 5
        result_top5 = engine.format_results(df, top_n=5)
        
        # Should only include first 5
        for i in range(5):
            assert f"TICK{i}" in result_top5
        assert "TICK5" not in result_top5
    
    def test_display_columns_order(self):
        """Test that display columns are in correct order."""
        engine = ScoringEngine()
        
        opportunities = [{
            "ticker": "AAPL",
            "expiration": "2025-01-30",
            "days_to_exp": 30,
            "strike": 95.0,
            "premium": 1.5,
            "delta": -0.3,
            "theta": 0.03,
            "gamma": 0.02,
            "daily_premium": 0.05,
            "annualized_roc": 6.0,
            "pop": 70.0,
            "support": 90.0,
            "near_term_support": 92.0,
            "current_price": 100.0,
            "iv_rank": 30.0,
            "score": 1.5,
            "volume": 100,
            "open_interest": 500
        }]
        
        result = engine.rank_opportunities(opportunities)
        
        expected_columns = [
            "ticker", "expiration", "days_to_exp", "strike", "premium",
            "delta", "theta", "gamma", "daily_premium", "annualized_roc", "pop",
            "support", "near_term_support", "current_price", "iv_rank", "score", "max_contracts",
        ]
        
        assert list(result.columns) == expected_columns