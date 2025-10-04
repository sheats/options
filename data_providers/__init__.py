"""Data provider abstraction for CSP scanner"""

from .base import DataProvider
from .yfinance_provider import YFinanceProvider

__all__ = ['DataProvider', 'YFinanceProvider']

# Conditional import for TastyTrade
try:
    from .tastytrade_provider import TastyTradeProvider
    __all__.append('TastyTradeProvider')
except ImportError:
    pass