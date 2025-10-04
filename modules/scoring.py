"""
Scoring Module

This module handles scoring and ranking of CSP opportunities.
"""

from typing import Dict, List

import pandas as pd

from modules import constants


class ScoringEngine:
    """
    Scores and ranks CSP opportunities based on various factors.
    
    The scoring algorithm considers:
    - Daily premium (higher is better)
    - Probability of profit (higher is better)
    - Risk score (lower is better)
    - Proximity to support levels (closer is better)
    """
    
    def __init__(self):
        """Initialize the ScoringEngine."""
        pass
    
    def calculate_score(
        self,
        daily_premium: float,
        probability_of_profit: float,
        delta: float,
        implied_volatility: float,
        strike: float,
        final_support: float,
        current_price: float
    ) -> float:
        """
        Calculate opportunity score based on multiple factors.
        
        Args:
            daily_premium: Daily premium in dollars
            probability_of_profit: Probability of profit (0-1)
            delta: Option delta
            implied_volatility: Implied volatility
            strike: Strike price
            final_support: Final support level
            current_price: Current stock price
            
        Returns:
            Calculated score
        """
        # Risk score combines delta and IV
        risk_score = abs(delta) * implied_volatility
        
        # Base score: daily premium * PoP / risk
        base_score = (daily_premium * probability_of_profit) / (risk_score + 0.01)
        
        # Proximity bonus: closer to support is better
        proximity_to_support = abs(strike - final_support) / current_price
        proximity_bonus = (1 - proximity_to_support) * 0.5
        
        # Final score
        score = base_score + proximity_bonus
        
        return score
    
    def rank_opportunities(self, opportunities: List[Dict]) -> pd.DataFrame:
        """
        Rank all opportunities by score and add position sizing.
        
        Args:
            opportunities: List of opportunity dictionaries
            
        Returns:
            DataFrame sorted by score with max contracts column added
        """
        if not opportunities:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(opportunities)
        
        # Sort by score (descending)
        df = df.sort_values("score", ascending=False)
        
        # Add max contracts calculation
        df["max_contracts"] = self._calculate_max_contracts(df["strike"])
        
        # Select display columns in order
        display_columns = [
            "ticker", "expiration", "days_to_exp", "strike", "premium",
            "delta", "theta", "gamma", "daily_premium", "annualized_roc", "pop",
            "support", "near_term_support", "current_price", "iv_rank", "score", "max_contracts",
        ]
        
        return df[display_columns]
    
    def _calculate_max_contracts(self, strikes: pd.Series) -> pd.Series:
        """
        Calculate maximum contracts based on position sizing rules.
        
        Args:
            strikes: Series of strike prices
            
        Returns:
            Series of maximum contract counts
        """
        max_position_value = constants.ACCOUNT_SIZE * constants.MAX_POSITION_PCT
        contract_values = strikes * 100  # Each contract = 100 shares
        max_contracts = (max_position_value / contract_values).astype(int)
        
        return max_contracts
    
    def format_results(self, df: pd.DataFrame, top_n: int = 20) -> str:
        """
        Format results DataFrame for display.
        
        Args:
            df: Results DataFrame
            top_n: Number of top results to show
            
        Returns:
            Formatted string representation
        """
        if df.empty:
            return "No CSP opportunities found matching criteria"
        
        # Take top N and copy
        df = df.head(top_n).copy()
        
        # Format numeric columns
        df["premium"] = df["premium"].apply(lambda x: f"${x:.2f}")
        df["strike"] = df["strike"].apply(lambda x: f"${x:.2f}")
        df["delta"] = df["delta"].apply(lambda x: f"{x:.3f}")
        df["theta"] = df["theta"].apply(lambda x: f"{x:.3f}")
        df["gamma"] = df["gamma"].apply(lambda x: f"{x:.3f}")
        df["daily_premium"] = df["daily_premium"].apply(lambda x: f"${x:.3f}")
        df["annualized_roc"] = df["annualized_roc"].apply(lambda x: f"{x:.1f}%")
        df["pop"] = df["pop"].apply(lambda x: f"{x:.1f}%")
        df["support"] = df["support"].apply(lambda x: f"${x:.2f}")
        df["near_term_support"] = df["near_term_support"].apply(lambda x: f"${x:.2f}")
        df["current_price"] = df["current_price"].apply(lambda x: f"${x:.2f}")
        df["iv_rank"] = df["iv_rank"].apply(lambda x: f"{x:.1f}%")
        df["score"] = df["score"].apply(lambda x: f"{x:.2f}")
        
        return df.to_string(index=False)