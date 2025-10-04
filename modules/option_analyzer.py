"""
Option Analyzer Module

This module handles option chain analysis and filtering.
"""

import datetime
import logging
import time
from typing import Dict, List, Optional

import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_providers.base import DataProvider
from modules import constants


logger = logging.getLogger(__name__)


class OptionAnalyzer:
    """
    Analyzes and filters option chains for CSP opportunities.
    
    This class handles:
    - Option chain retrieval and filtering
    - Greeks-based filtering (delta, theta, gamma)
    - Volume and open interest filtering
    - Premium and ROC calculations
    - Earnings date checking
    """
    
    def __init__(
        self, 
        data_provider: DataProvider,
        max_weeks: int = constants.DEFAULT_MAX_WEEKS,
        min_iv_rank: float = constants.DEFAULT_MIN_IV_RANK,
        support_buffer: float = constants.DEFAULT_SUPPORT_BUFFER,
        no_support_filter: bool = False,
        scan_date: Optional[datetime.date] = None
    ):
        """
        Initialize the OptionAnalyzer.
        
        Args:
            data_provider: The data provider for fetching option data
            max_weeks: Maximum weeks to expiration
            min_iv_rank: Minimum IV rank percentage
            support_buffer: Support buffer percentage
            no_support_filter: Whether to disable support level filtering
            scan_date: Date to use for scanning (default: today)
        """
        self.data_provider = data_provider
        self.max_weeks = max_weeks
        self.min_iv_rank = min_iv_rank
        self.support_buffer = support_buffer
        self.no_support_filter = no_support_filter
        self.today = scan_date if scan_date else datetime.date.today()
    
    def check_earnings_in_period(self, ticker: str, exp_date: datetime.date) -> bool:
        """
        Check if earnings fall within 7 days before expiration.
        
        Args:
            ticker: Stock ticker symbol
            exp_date: Option expiration date
            
        Returns:
            True if earnings within 7 days of expiration, False otherwise
        """
        try:
            earnings_dates = self.data_provider.get_earnings_dates(ticker)
            
            # Check if any earnings date is within 7 days before expiration
            cutoff_date = exp_date - datetime.timedelta(days=7)
            
            for ed in earnings_dates:
                if cutoff_date <= ed <= exp_date:
                    logger.debug(f"{ticker} has earnings on {ed} within 7 days of {exp_date}")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Earnings check error for {ticker}: {str(e)}")
            return False
    
    def analyze_option_chain(
        self, 
        ticker: str, 
        exp_str: str, 
        exp_date: datetime.date,
        current_price: float,
        supports: Dict[str, float],
        iv_rank: float
    ) -> List[Dict]:
        """
        Analyze a single option chain for CSP opportunities.
        
        Args:
            ticker: Stock ticker symbol
            exp_str: Expiration date string
            exp_date: Expiration date as date object
            current_price: Current stock price
            supports: Dictionary of support levels
            iv_rank: Current IV rank
            
        Returns:
            List of opportunity dictionaries
        """
        opportunities = []
        days_to_exp = (exp_date - self.today).days
        
        if days_to_exp <= 0:
            return opportunities
        
        logger.debug(f"\nChecking {ticker} {exp_str} ({days_to_exp} days)")
        
        # Check earnings
        if self.check_earnings_in_period(ticker, exp_date):
            logger.info(f"{ticker} {exp_str} skipped: Earnings within 7 days of expiration")
            return opportunities
        
        try:
            # Get option chain
            puts = self.data_provider.get_option_chain(ticker, exp_str)
            
            if puts.empty:
                logger.debug(f"{ticker} {exp_str}: Empty option chain")
                return opportunities
            
            initial_count = len(puts)
            logger.debug(f"{ticker} {exp_str}: {initial_count} puts in chain")
            
            # Apply filters
            puts_filtered = self._apply_option_filters(
                puts, ticker, exp_str, current_price, supports
            )
            
            if puts_filtered.empty:
                return opportunities
            
            # Calculate opportunities from remaining puts
            final_support = supports["final_support"]
            near_term_support = supports["near_term_support"]
            
            for _, put in puts_filtered.iterrows():
                opp = self._create_opportunity(
                    ticker, exp_str, days_to_exp, put, 
                    final_support, near_term_support, 
                    current_price, iv_rank
                )
                
                if opp:
                    opportunities.append(opp)
            
            logger.debug(
                f"{ticker} {exp_str}: {len(opportunities)} opportunities after premium/ROC filter "
                f"(daily>=${constants.MIN_DAILY_PREMIUM}, ROC>={constants.MIN_ANNUALIZED_ROC}%)"
            )
            
        except Exception as e:
            logger.warning(f"Option chain error for {ticker} {exp_str}: {str(e)}")
        
        return opportunities
    
    def _apply_option_filters(
        self, 
        puts: pd.DataFrame, 
        ticker: str, 
        exp_str: str, 
        current_price: float,
        supports: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Apply all filters to option chain.
        
        Args:
            puts: DataFrame of put options
            ticker: Stock ticker symbol
            exp_str: Expiration date string
            current_price: Current stock price
            supports: Dictionary of support levels
            
        Returns:
            Filtered DataFrame of put options
        """
        # Filter by volume and open interest
        puts_filtered = puts[
            (puts["volume"] >= constants.MIN_VOLUME)
            & (puts["openInterest"] >= constants.MIN_OPEN_INTEREST)
        ].copy()
        
        logger.debug(
            f"{ticker} {exp_str}: {len(puts_filtered)} puts after volume/OI filter "
            f"(volume>={constants.MIN_VOLUME}, OI>={constants.MIN_OPEN_INTEREST})"
        )
        
        if puts_filtered.empty:
            return puts_filtered
        
        # Dynamic support filter
        if not self.no_support_filter:
            final_support = supports["final_support"]
            near_term_support = supports["near_term_support"]
            
            # Use the higher of near-term support or final support for filtering
            effective_support = max(
                near_term_support * (1 - self.support_buffer), 
                final_support
            )
            
            # Strike must be between final_support*0.9 and current_price*(1-buffer)
            max_strike = current_price * (1 - self.support_buffer)
            min_strike = final_support * 0.9
            
            puts_filtered = puts_filtered[
                (puts_filtered["strike"] <= max_strike) &
                (puts_filtered["strike"] >= min_strike) &
                (puts_filtered["strike"] <= effective_support)
            ]
            
            logger.debug(
                f"{ticker} {exp_str}: {len(puts_filtered)} puts after support filter "
                f"(${min_strike:.2f} <= strike <= ${max_strike:.2f} and <= ${effective_support:.2f})"
            )
        else:
            logger.debug(f"{ticker} {exp_str}: Support filter bypassed (--no-support-filter enabled)")
        
        if puts_filtered.empty:
            return puts_filtered
        
        # Filter by delta range
        puts_filtered = puts_filtered[
            (puts_filtered["delta"] >= constants.MIN_DELTA)
            & (puts_filtered["delta"] <= constants.MAX_DELTA)
        ]
        logger.debug(
            f"{ticker} {exp_str}: {len(puts_filtered)} puts after delta filter "
            f"({constants.MIN_DELTA} to {constants.MAX_DELTA})"
        )
        
        if puts_filtered.empty:
            return puts_filtered
        
        # Filter by theta
        puts_filtered = puts_filtered[puts_filtered["theta"] > constants.MIN_THETA]
        logger.debug(
            f"{ticker} {exp_str}: {len(puts_filtered)} puts after theta filter "
            f"(>{constants.MIN_THETA})"
        )
        
        if puts_filtered.empty:
            return puts_filtered
        
        # Filter by gamma
        puts_filtered = puts_filtered[puts_filtered["gamma"] < constants.MAX_GAMMA]
        logger.debug(
            f"{ticker} {exp_str}: {len(puts_filtered)} puts after gamma filter "
            f"(<{constants.MAX_GAMMA})"
        )
        
        return puts_filtered
    
    def _create_opportunity(
        self,
        ticker: str,
        exp_str: str,
        days_to_exp: int,
        put: pd.Series,
        final_support: float,
        near_term_support: float,
        current_price: float,
        iv_rank: float
    ) -> Optional[Dict]:
        """
        Create an opportunity dictionary from a put option.
        
        Args:
            ticker: Stock ticker symbol
            exp_str: Expiration date string
            days_to_exp: Days to expiration
            put: Put option data series
            final_support: Final calculated support level
            near_term_support: Near-term support level
            current_price: Current stock price
            iv_rank: Current IV rank
            
        Returns:
            Opportunity dictionary or None if filters not met
        """
        # Calculate metrics
        strike = put["strike"]
        bid = put["bid"]
        ask = put["ask"]
        premium = (bid + ask) / 2 if bid > 0 and ask > 0 else put["lastPrice"]
        
        if premium <= 0:
            return None
        
        collateral = strike * 100
        daily_premium = premium / days_to_exp
        annualized_roc = (premium / collateral) * (365 / days_to_exp) * 100
        pop = 1 + put["delta"]  # Probability of profit
        risk_score = abs(put["delta"]) * put["impliedVolatility"]
        
        # Enhanced score calculation with proximity bonus
        base_score = (daily_premium * pop) / (risk_score + 0.01)
        proximity_bonus = (1 - abs(strike - final_support) / current_price) * 0.5
        score = base_score + proximity_bonus
        
        # Apply minimum requirements filter
        if (daily_premium >= constants.MIN_DAILY_PREMIUM and 
            annualized_roc >= constants.MIN_ANNUALIZED_ROC):
            
            return {
                "ticker": ticker,
                "expiration": exp_str,
                "days_to_exp": days_to_exp,
                "strike": strike,
                "premium": premium,
                "delta": put["delta"],
                "theta": put["theta"],
                "gamma": put["gamma"],
                "iv": put["impliedVolatility"],
                "daily_premium": daily_premium,
                "annualized_roc": annualized_roc,
                "pop": pop * 100,
                "support": final_support,
                "near_term_support": near_term_support,
                "current_price": current_price,
                "iv_rank": iv_rank,
                "score": score,
                "volume": put["volume"],
                "open_interest": put["openInterest"],
            }
        
        return None