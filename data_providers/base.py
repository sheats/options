"""Abstract base class for data providers"""

import datetime
from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd


class DataProvider(ABC):
    """Abstract base class for market data providers"""

    @abstractmethod
    def get_stock_info(self, ticker: str) -> Dict:
        """
        Get stock fundamental information

        Returns:
            Dict with keys: 'marketCap', 'trailingPE', 'forwardEps', 'oneYrReturn'
        """
        pass

    @abstractmethod
    def get_historical_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Get historical OHLCV data

        Args:
            ticker: Stock symbol
            period: Time period (e.g., '1y', '6mo', '3mo')

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        pass

    @abstractmethod
    def get_iv_rank(self, ticker: str) -> float:
        """
        Get Implied Volatility rank (0-100)

        Returns:
            IV rank as percentage
        """
        pass

    @abstractmethod
    def get_earnings_dates(self, ticker: str) -> List[datetime.date]:
        """
        Get upcoming earnings dates

        Returns:
            List of upcoming earnings dates
        """
        pass

    @abstractmethod
    def get_option_chain(self, ticker: str, exp_date: str) -> pd.DataFrame:
        """
        Get put options chain for a specific expiration

        Args:
            ticker: Stock symbol
            exp_date: Expiration date as string

        Returns:
            DataFrame with columns: strike, bid, ask, lastPrice, volume,
            openInterest, impliedVolatility, delta, theta, gamma
        """
        pass

    @abstractmethod
    def get_option_expirations(self, ticker: str) -> List[str]:
        """
        Get available option expiration dates

        Returns:
            List of expiration dates as strings
        """
        pass
