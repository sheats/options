"""YFinance data provider implementation"""

import datetime
import logging
from typing import Dict, List
from math import log, sqrt, exp

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from .base import DataProvider

logger = logging.getLogger(__name__)


class YFinanceProvider(DataProvider):
    """Data provider using yfinance library"""
    
    def __init__(self):
        self.risk_free_rate = self._fetch_risk_free_rate()
        
    def _fetch_risk_free_rate(self) -> float:
        """Fetch current risk-free rate from 10Y Treasury, default to 4% if fails"""
        try:
            tnx = yf.Ticker('^TNX')
            hist = tnx.history(period='1d')
            if not hist.empty:
                # TNX is in percentage, divide by 100
                rate = hist['Close'].iloc[-1] / 100
                logger.info(f"Fetched risk-free rate: {rate:.2%}")
                return rate
        except Exception as e:
            logger.debug(f"Failed to fetch risk-free rate: {str(e)}")
        
        default_rate = 0.04
        logger.info(f"Using default risk-free rate: {default_rate:.2%}")
        return default_rate
        
    def black_scholes_greeks(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'put') -> tuple:
        """Calculate Black-Scholes Greeks for options with enhanced edge case handling"""
        # Handle edge cases
        if T <= 0:
            logger.debug(f"T <= 0, returning zero Greeks")
            return 0.0, 0.0, 0.0
        
        if pd.isna(sigma) or sigma <= 0:
            logger.debug(f"Invalid sigma={sigma}, returning zero Greeks")
            return 0.0, 0.0, 0.0
        
        try:
            d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)
            
            if option_type == 'put':
                delta = -norm.cdf(-d1)
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * norm.cdf(-d2)) / 365  # Daily theta
                gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
            else:
                delta = norm.cdf(d1)
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2)) / 365
                gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
            
            return delta, theta, gamma
            
        except Exception as e:
            logger.debug(f"Error in Greeks calculation: {str(e)}")
            return 0.0, 0.0, 0.0
    
    def get_stock_info(self, ticker: str) -> Dict:
        """Get stock fundamental information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Calculate 1-year return
            hist = stock.history(period='1y')
            if len(hist) > 0:
                one_yr_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
            else:
                one_yr_return = 0.0
            
            return {
                'marketCap': info.get('marketCap', 0),
                'trailingPE': info.get('trailingPE', float('inf')),
                'forwardEps': info.get('forwardEps', 0),
                'oneYrReturn': one_yr_return
            }
        except Exception as e:
            logger.error(f"Error getting stock info for {ticker}: {str(e)}")
            return {
                'marketCap': 0,
                'trailingPE': float('inf'),
                'forwardEps': 0,
                'oneYrReturn': -100
            }
    
    def get_historical_data(self, ticker: str, period: str = '1y') -> pd.DataFrame:
        """Get historical OHLCV data"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_iv_rank(self, ticker: str) -> float:
        """Calculate IV rank using historical volatility as proxy"""
        try:
            hist = self.get_historical_data(ticker, period='1y')
            if hist.empty:
                return 0.0
            
            # Calculate rolling 30-day volatility as proxy for IV
            returns = hist['Close'].pct_change()
            rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
            
            current_vol = rolling_vol.iloc[-1]
            min_vol = rolling_vol.min()
            max_vol = rolling_vol.max()
            
            if max_vol > min_vol:
                iv_rank = (current_vol - min_vol) / (max_vol - min_vol) * 100
            else:
                iv_rank = 50.0
                
            return iv_rank
        except Exception as e:
            logger.error(f"Error calculating IV rank for {ticker}: {str(e)}")
            return 0.0
    
    def get_earnings_dates(self, ticker: str) -> List[datetime.date]:
        """Get upcoming earnings dates"""
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            if calendar is not None and 'Earnings Date' in calendar:
                earnings_dates = calendar['Earnings Date']
                if isinstance(earnings_dates, pd.Timestamp):
                    earnings_dates = [earnings_dates]
                
                return [ed.date() for ed in earnings_dates if pd.notna(ed)]
            
            return []
        except Exception as e:
            logger.debug(f"Error getting earnings dates for {ticker}: {str(e)}")
            return []
    
    def get_option_expirations(self, ticker: str) -> List[str]:
        """Get available option expiration dates"""
        try:
            stock = yf.Ticker(ticker)
            return list(stock.options)
        except Exception as e:
            logger.error(f"Error getting option expirations for {ticker}: {str(e)}")
            return []
    
    def get_option_chain(self, ticker: str, exp_date: str) -> pd.DataFrame:
        """Get put options chain with computed Greeks"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get current price for Greeks calculation
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            
            # Get option chain
            opt_chain = stock.option_chain(exp_date)
            puts = opt_chain.puts.copy()
            
            # Add Greeks computation
            puts['delta'] = 0.0
            puts['theta'] = 0.0
            puts['gamma'] = 0.0
            
            # Calculate days to expiration
            exp_datetime = pd.to_datetime(exp_date)
            days_to_exp = (exp_datetime.date() - datetime.date.today()).days
            T = days_to_exp / 365.0
            
            # Log invalid IV rows before computing Greeks
            invalid_iv_count = puts[pd.isna(puts['impliedVolatility']) | (puts['impliedVolatility'] <= 0)].shape[0]
            if invalid_iv_count > 0:
                logger.debug(f"{ticker} {exp_date}: {invalid_iv_count} puts with invalid IV will have zero Greeks")
            
            # Compute Greeks for each option
            for idx, row in puts.iterrows():
                if pd.notna(row['impliedVolatility']) and row['impliedVolatility'] > 0 and T > 0:
                    delta, theta, gamma = self.black_scholes_greeks(
                        current_price, 
                        row['strike'], 
                        T, 
                        self.risk_free_rate, 
                        row['impliedVolatility']
                    )
                    puts.at[idx, 'delta'] = delta
                    puts.at[idx, 'theta'] = theta
                    puts.at[idx, 'gamma'] = gamma
            
            # Return required columns
            return puts[['strike', 'bid', 'ask', 'lastPrice', 'volume', 
                        'openInterest', 'impliedVolatility', 'delta', 'theta', 'gamma']]
            
        except Exception as e:
            logger.error(f"Error getting option chain for {ticker} {exp_date}: {str(e)}")
            return pd.DataFrame()