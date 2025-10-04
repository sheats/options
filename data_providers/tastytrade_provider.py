"""TastyTrade data provider implementation"""

import datetime
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import tastytrade
    from tastytrade import ProductionSession
    from tastytrade.instruments import Equity
    from tastytrade.metrics import get_market_metrics

    TASTYTRADE_AVAILABLE = True
except ImportError:
    TASTYTRADE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "tastytrade package not installed. Install with: pip install tastytrade"
    )

from .base import DataProvider

logger = logging.getLogger(__name__)


class TastyTradeProvider(DataProvider):
    """Data provider using TastyTrade SDK"""

    def __init__(self, username: str, password: str):
        if not TASTYTRADE_AVAILABLE:
            raise ImportError(
                "TastyTrade SDK not available. Install with: pip install tastytrade"
            )

        self.username = username
        self.password = password
        self.session = None
        self._yf_fallback = None
        self._authenticate()

    @property
    def yf_fallback(self):
        """Lazy load yfinance fallback to avoid circular imports"""
        if self._yf_fallback is None:
            from .yfinance_provider import YFinanceProvider

            self._yf_fallback = YFinanceProvider()
        return self._yf_fallback

    def _authenticate(self):
        """Authenticate with TastyTrade"""
        try:
            logger.info("Authenticating with TastyTrade...")
            self.session = ProductionSession(self.username, self.password)
            logger.info("TastyTrade authentication successful")
        except Exception as e:
            logger.error(f"TastyTrade authentication failed: {str(e)}")
            logger.error("Falling back to yfinance provider")
            raise

    def get_stock_info(self, ticker: str) -> Dict:
        """Get stock fundamental information"""
        try:
            # TastyTrade focuses on trading data, fallback to yfinance for fundamentals
            logger.debug(f"Using yfinance fallback for {ticker} fundamentals")
            return self.yf_fallback.get_stock_info(ticker)

        except Exception as e:
            logger.error(f"Error getting stock info for {ticker}: {str(e)}")
            return {
                "marketCap": 0,
                "trailingPE": float("inf"),
                "forwardEps": 0,
                "oneYrReturn": -100,
            }

    def get_historical_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Get historical OHLCV data"""
        try:
            # Calculate date range
            today = datetime.date.today()
            if period == "1y":
                from_date = today - datetime.timedelta(days=365)
            elif period == "6mo":
                from_date = today - datetime.timedelta(days=180)
            elif period == "3mo":
                from_date = today - datetime.timedelta(days=90)
            else:
                from_date = today - datetime.timedelta(days=365)

            # Get quotes history from TastyTrade
            equity = Equity.get_equity(self.session, ticker)
            history = self.session.get_candle_chart(
                equity, interval="1d", start_date=from_date, end_date=today
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                [
                    {
                        "Open": candle.open,
                        "High": candle.high,
                        "Low": candle.low,
                        "Close": candle.close,
                        "Volume": candle.volume,
                        "datetime": candle.datetime,
                    }
                    for candle in history
                ]
            )

            if not df.empty:
                df.set_index("datetime", inplace=True)
                return df[["Open", "High", "Low", "Close", "Volume"]]

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {str(e)}")
            logger.debug("Falling back to yfinance for historical data")
            return self.yf_fallback.get_historical_data(ticker, period)

    def get_iv_rank(self, ticker: str) -> float:
        """Get IV rank from TastyTrade metrics"""
        try:
            equity = Equity.get_equity(self.session, ticker)
            metrics = get_market_metrics(self.session, [equity.symbol])

            if metrics and equity.symbol in metrics:
                metric = metrics[equity.symbol]
                # TastyTrade provides IV rank directly
                if hasattr(metric, "iv_rank"):
                    return float(metric.iv_rank * 100)  # Convert to percentage

            # Fallback to historical volatility calculation
            logger.debug(f"IV rank not available for {ticker}, using HV proxy")
            return self.yf_fallback.get_iv_rank(ticker)

        except Exception as e:
            logger.error(f"Error getting IV rank for {ticker}: {str(e)}")
            return 0.0

    def get_earnings_dates(self, ticker: str) -> List[datetime.date]:
        """Get upcoming earnings dates from TastyTrade"""
        try:
            # Get corporate actions including earnings
            equity = Equity.get_equity(self.session, ticker)

            # Use the events endpoint
            response = self.session.get(
                f"/corporate-actions/events/{equity.symbol}",
                params={"event-types": "Earnings"},
            )

            if response and hasattr(response, "data"):
                earnings_dates = []
                for event in response.data:
                    if hasattr(event, "event_date"):
                        event_date = pd.to_datetime(event.event_date).date()
                        if event_date >= datetime.date.today():
                            earnings_dates.append(event_date)
                return earnings_dates

            return []

        except Exception as e:
            logger.debug(f"Error getting earnings dates for {ticker}: {str(e)}")
            return self.yf_fallback.get_earnings_dates(ticker)

    def get_option_expirations(self, ticker: str) -> List[str]:
        """Get available option expiration dates"""
        try:
            equity = Equity.get_equity(self.session, ticker)
            chains = self.session.get_option_chains(equity)

            if chains and hasattr(chains, "expirations"):
                return [exp.strftime("%Y-%m-%d") for exp in chains.expirations]

            return []

        except Exception as e:
            logger.error(f"Error getting option expirations for {ticker}: {str(e)}")
            return self.yf_fallback.get_option_expirations(ticker)

    def get_option_chain(self, ticker: str, exp_date: str) -> pd.DataFrame:
        """Get put options chain with native Greeks from TastyTrade"""
        try:
            equity = Equity.get_equity(self.session, ticker)

            # Parse expiration date
            exp_datetime = pd.to_datetime(exp_date)

            # Get option chain for specific expiration
            chains = self.session.get_chains(ticker, expiration=exp_datetime.date())

            if not chains or not hasattr(chains, "options"):
                logger.warning(f"No option chain data for {ticker} {exp_date}")
                return pd.DataFrame()

            # Filter for puts and extract data
            puts_data = []
            for option in chains.options:
                if option.option_type == "P":  # Put options
                    puts_data.append(
                        {
                            "strike": float(option.strike),
                            "bid": float(option.bid) if option.bid else 0.0,
                            "ask": float(option.ask) if option.ask else 0.0,
                            "lastPrice": float(option.last) if option.last else 0.0,
                            "volume": int(option.volume) if option.volume else 0,
                            "openInterest": (
                                int(option.open_interest) if option.open_interest else 0
                            ),
                            "impliedVolatility": (
                                float(option.implied_volatility)
                                if option.implied_volatility
                                else 0.0
                            ),
                            "delta": float(option.delta) if option.delta else 0.0,
                            "theta": float(option.theta) if option.theta else 0.0,
                            "gamma": float(option.gamma) if option.gamma else 0.0,
                        }
                    )

            if puts_data:
                df = pd.DataFrame(puts_data)
                # Sort by strike descending (highest to lowest)
                df.sort_values("strike", ascending=False, inplace=True)
                return df

            return pd.DataFrame()

        except Exception as e:
            logger.error(
                f"Error getting option chain for {ticker} {exp_date}: {str(e)}"
            )
            logger.debug("Falling back to yfinance for option chain")
            return self.yf_fallback.get_option_chain(ticker, exp_date)
