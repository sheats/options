"""Cache provider abstraction for CSP scanner"""

from .base import CacheProvider
from .sqlite_provider import SQLiteCacheProvider

__all__ = ['CacheProvider', 'SQLiteCacheProvider']