"""
Support Calculator Module

This module handles calculation of support levels using multiple methods.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.cluster import KMeans

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_providers.base import DataProvider
from modules import constants


logger = logging.getLogger(__name__)


class SupportCalculator:
    """
    Calculates multiple support levels for stocks using various technical analysis methods.
    
    Methods include:
    - Simple Moving Averages (200-day and 50-day)
    - 52-week low
    - Local minima detection
    - K-means clustering of price levels
    """
    
    def __init__(self, data_provider: DataProvider, support_buffer: float = constants.DEFAULT_SUPPORT_BUFFER):
        """
        Initialize the SupportCalculator.
        
        Args:
            data_provider: The data provider for fetching historical data
            support_buffer: Buffer percentage to apply to support levels (default 2%)
        """
        self.data_provider = data_provider
        self.support_buffer = support_buffer
    
    def calculate_support_levels(self, ticker: str) -> Dict[str, float]:
        """
        Calculate multiple support levels for a stock including near-term support.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing various support levels and current price:
            - sma_200: 200-day simple moving average
            - near_term_support: 50-day simple moving average
            - low_52w: 52-week low
            - local_support: Average of recent local minima
            - cluster_support: K-means cluster support level
            - final_support: Combined support with buffer applied
            - current_price: Current stock price
            
            Returns empty dict if calculation fails
        """
        try:
            # Get historical data from provider
            hist = self.data_provider.get_historical_data(ticker, period="1y")
            
            if len(hist) < constants.MIN_HISTORY_DAYS:
                logger.warning(f"{ticker}: Insufficient history ({len(hist)} days)")
                return {}
            
            # Method 1: Long-term SMA
            sma_200 = hist["Close"].rolling(constants.SMA_PERIOD_LONG).mean().iloc[-1]
            
            # Method 2: Near-term SMA
            near_term_support = hist["Close"].rolling(constants.SMA_PERIOD_SHORT).mean().iloc[-1]
            
            # Method 3: 52-week low
            low_52w = hist["Low"].min()
            
            # Method 4: Local minima detection
            local_support = self._calculate_local_support(hist["Low"].values, low_52w)
            
            # Method 5: K-means clustering
            support_cluster = self._calculate_kmeans_support(hist["Close"].values, ticker, low_52w)
            
            # Combined support with buffer
            avg_support = np.mean([sma_200, local_support, support_cluster])
            final_support = avg_support * (1 - self.support_buffer)
            
            current_price = hist["Close"].iloc[-1]
            
            # Calculate support distances
            support_diff_pct = ((current_price - final_support) / current_price * 100)
            near_term_diff_pct = ((current_price - near_term_support) / current_price * 100)
            
            # Log support components
            self._log_support_levels(
                ticker, sma_200, near_term_support, near_term_diff_pct, 
                low_52w, local_support, support_cluster, final_support, 
                current_price, support_diff_pct
            )
            
            return {
                "sma_200": sma_200,
                "near_term_support": near_term_support,
                "low_52w": low_52w,
                "local_support": local_support,
                "cluster_support": support_cluster,
                "final_support": final_support,
                "current_price": current_price,
            }
            
        except Exception as e:
            logger.error(f"Support calculation error for {ticker}: {str(e)}")
            return {}
    
    def _calculate_local_support(self, prices: np.ndarray, low_52w: float) -> float:
        """
        Calculate local support using local minima detection.
        
        Args:
            prices: Array of low prices
            low_52w: 52-week low as fallback
            
        Returns:
            Average of recent local minima or fallback value
        """
        local_mins_idx = signal.argrelextrema(
            prices, np.less_equal, order=constants.LOCAL_MINIMA_ORDER
        )[0]
        
        if len(local_mins_idx) > 0:
            recent_supports = prices[local_mins_idx[-3:]]  # Last 3 local minima
            return np.mean(recent_supports)
        else:
            # Fallback to 52-week low with buffer
            local_support = low_52w * (1 - self.support_buffer)
            logger.debug(f"No local minima found, using 52w low * {1 - self.support_buffer}")
            return local_support
    
    def _calculate_kmeans_support(self, closes: np.ndarray, ticker: str, low_52w: float) -> float:
        """
        Calculate support level using K-means clustering.
        
        Args:
            closes: Array of closing prices
            ticker: Stock ticker for logging
            low_52w: 52-week low as fallback
            
        Returns:
            Lowest cluster center or fallback value
        """
        try:
            closes_reshaped = closes.reshape(-1, 1)
            kmeans = KMeans(n_clusters=constants.KMEANS_CLUSTERS, random_state=42, n_init=10)
            kmeans.fit(closes_reshaped)
            support_cluster = min(kmeans.cluster_centers_)[0]
            return support_cluster
        except Exception as e:
            logger.error(f"K-means clustering error for {ticker}: {str(e)}")
            return low_52w
    
    def _log_support_levels(
        self,
        ticker: str,
        sma_200: float,
        near_term_support: float,
        near_term_diff_pct: float,
        low_52w: float,
        local_support: float,
        support_cluster: float,
        final_support: float,
        current_price: float,
        support_diff_pct: float
    ):
        """Log calculated support levels for debugging."""
        logger.debug(f"{ticker} Support Levels:")
        logger.debug(f"  {constants.SMA_PERIOD_LONG}-day SMA: ${sma_200:.2f}")
        logger.debug(
            f"  {constants.SMA_PERIOD_SHORT}-day SMA (near-term): ${near_term_support:.2f} "
            f"({near_term_diff_pct:.1f}% below current)"
        )
        logger.debug(f"  52-week low: ${low_52w:.2f}")
        logger.debug(f"  Local support: ${local_support:.2f}")
        logger.debug(f"  Cluster support: ${support_cluster:.2f}")
        logger.debug(f"  Final support ({int((1-self.support_buffer)*100)}%): ${final_support:.2f}")
        logger.debug(f"  Current price: ${current_price:.2f}")
        logger.debug(f"  Distance to final support: {support_diff_pct:.1f}%")