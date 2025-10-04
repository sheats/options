# SQLite Cache Implementation for CSP Scanner

## Overview

The CSP Scanner now includes SQLite database caching to store quality stock data with a 24-hour cache lifetime. This significantly improves performance by reducing API calls for stock fundamental data.

## Features

1. **Automatic Cache Management**
   - Creates `csp_scanner_cache.db` automatically on first run
   - Caches stock quality metrics (market cap, P/E ratio, forward EPS, 1-year return)
   - 24-hour cache lifetime (configurable)
   - Respects different filter criteria for different exchanges

2. **Cache Structure**
   - Primary key: (ticker, exchange, max_pe, min_return)
   - Stores both passed and failed stocks to avoid repeated API calls
   - Indexed for fast lookups

3. **Command Line Options**
   - `--clear-cache`: Clear the cache before running scan
   - `--cache-db <path>`: Use custom cache database file

## Database Schema

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
);
```

## Usage Examples

```bash
# Normal usage - will use cache if available
python csp_scanner.py --exchange NASDAQ

# Clear cache and fetch fresh data
python csp_scanner.py --exchange NASDAQ --clear-cache

# Use a custom cache database
python csp_scanner.py --exchange SP500 --cache-db my_cache.db

# View cache statistics (shown in logs)
python csp_scanner.py --exchange NASDAQ --debug
```

## Cache Testing

Use the included test script to inspect cache contents:

```bash
# View cache statistics and contents
python test_cache.py

# Clear old cache entries (>24 hours)
python test_cache.py --clear-old
```

## Performance Benefits

- First run: Full API calls for all stocks
- Subsequent runs within 24 hours: Near-instant quality filtering
- Reduces API rate limiting issues
- Allows for more frequent scans without API throttling

## Implementation Details

1. **Cache Check Flow**
   - Check if all requested tickers exist in cache for the exchange
   - If cache is complete and fresh (<24 hours), use cached data
   - Otherwise, check individual tickers and fetch missing ones

2. **Filter Awareness**
   - Cache respects different P/E and return filters for NASDAQ vs SP500
   - Separate cache entries for different filter combinations
   - Ensures correct data is used for each exchange type

3. **Error Handling**
   - Graceful fallback if cache operations fail
   - Continues with API calls if cache is unavailable
   - Logs all cache-related errors for debugging

## Cache Maintenance

The cache is self-maintaining:
- Old entries are ignored after 24 hours
- Use `--clear-cache` to force a complete refresh
- Database file can be deleted manually if needed