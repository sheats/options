# Cache Provider Architecture

This directory contains the cache provider architecture for the CSP Scanner, designed to optimize performance by caching stock quality filter results.

## Architecture Overview

The cache provider architecture follows a similar pattern to the data provider architecture:

- **Abstract Base Class (`CacheProvider`)**: Defines the interface that all cache providers must implement
- **SQLite Implementation (`SQLiteCacheProvider`)**: Default implementation using SQLite for local caching
- **Extensible Design**: Easy to add new cache backends (Redis, Memcached, etc.)

## Key Features

1. **Self-Contained**: Each cache provider handles its own initialization, schema creation, and error handling
2. **Filter-Aware**: Cache keys include filter criteria (max_pe, min_return) to support different scanning parameters
3. **Time-Based Expiration**: Configurable cache lifetime with automatic expiration checking
4. **Optional Usage**: Scanner works with or without cache provider (use `--no-cache` flag)
5. **Statistics Support**: Built-in cache statistics for monitoring and debugging

## Interface Methods

### Core Methods

- `save_quality_stocks(ticker, exchange, stock_data, passed_filter, filter_criteria)`: Save stock quality check results
- `get_quality_stocks(exchange, filter_criteria)`: Retrieve cached stocks for given criteria
- `clear_cache()`: Remove all cached data
- `is_cache_valid(last_updated, lifetime_hours)`: Check if a cache entry is still valid

### Additional Methods (SQLite Provider)

- `get_cache_stats()`: Get cache statistics by exchange

## Usage

### Basic Usage in CSP Scanner

```python
from cache_providers import SQLiteCacheProvider

# Initialize cache provider
cache = SQLiteCacheProvider("cache.db", cache_lifetime_hours=24)

# Use in scanner
scanner = CSPScanner(
    data_provider=data_provider,
    cache_provider=cache,  # Optional
)
```

### Command Line Usage

```bash
# Run with cache (default)
python csp_scanner.py

# Run without cache
python csp_scanner.py --no-cache

# Clear cache before running
python csp_scanner.py --clear-cache

# Use custom cache database
python csp_scanner.py --cache-db my_cache.db
```

## Cache Schema (SQLite)

The SQLite implementation uses the following schema:

```sql
CREATE TABLE quality_stocks (
    ticker TEXT,
    exchange TEXT,
    market_cap REAL,
    pe_ratio REAL,
    forward_eps REAL,
    one_yr_return REAL,
    passed_filter INTEGER,
    max_pe REAL,
    min_return REAL,
    last_updated TIMESTAMP,
    PRIMARY KEY (ticker, exchange, max_pe, min_return)
)
```

## Extending with New Providers

To add a new cache provider (e.g., Redis):

1. Create a new file: `redis_provider.py`
2. Implement the `CacheProvider` interface
3. Add to `__init__.py` exports
4. Use in scanner with `cache_provider=RedisCacheProvider(...)`

Example structure:

```python
from .base import CacheProvider

class RedisCacheProvider(CacheProvider):
    def __init__(self, redis_url: str, **kwargs):
        self.redis = redis.from_url(redis_url)
        self.ttl = kwargs.get('cache_lifetime_hours', 24) * 3600
    
    def save_quality_stocks(self, ...):
        # Implementation
        pass
    
    # ... other methods
```

## Performance Benefits

- **Reduced API Calls**: Quality filter results are cached, avoiding repeated stock info lookups
- **Faster Scans**: Subsequent scans with same criteria use cached results
- **Configurable Lifetime**: Balance between freshness and performance (default: 24 hours)
- **Criteria-Specific**: Different filter criteria maintain separate cache entries

## Testing

Run the test script to verify cache functionality:

```bash
python test_cache_provider.py
```

This tests:
- Cache initialization
- Saving and retrieving data
- Filter criteria matching
- Cache expiration
- Cache clearing
- Statistics retrieval