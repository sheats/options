# Ticker Data Pre-fetching and Caching System

## Overview

The ticker update system pre-fetches comprehensive stock data and caches it for efficient retrieval. This dramatically improves the performance of the CSP scanner by reducing API calls during scanning operations.

## Features

### Comprehensive Data Collection

The `update_tickers.py` script fetches and caches:

- **Basic Information**: Company name, sector, industry, market cap, employees
- **Price Data**: Current price, 52-week high/low, moving averages (20, 50, 200-day)
- **Volume Data**: Average volume, relative volume, float shares
- **Valuation Metrics**: P/E ratios, PEG ratio, price-to-sales, price-to-book
- **Earnings & Dividends**: EPS, earnings dates, dividend yield, payout ratio
- **Financial Metrics**: Profit margins, ROE, ROA, debt/equity, current ratio
- **Risk Metrics**: Beta, historical volatility (10, 30, 60, 90-day, 1-year)
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, support/resistance
- **Returns**: 1-week, 1-month, 3-month, 6-month, 1-year, YTD returns
- **Options Data**: IV rank, put/call ratio, average implied volatility

### Performance Benefits

- **Without cache**: 5-10 seconds per ticker (multiple API calls)
- **With cache**: <0.1 seconds per ticker (single database query)
- **Typical speedup**: 50-100x faster scanning

## Usage

### Basic Commands

```bash
# Update S&P 500 stocks
python3 update_tickers.py --exchange SP500

# Update NASDAQ-100 stocks
python3 update_tickers.py --exchange NASDAQ

# Update top 500 NASDAQ stocks by market cap
python3 update_tickers.py --exchange NASDAQ_500

# Update all NASDAQ stocks with market cap > $1B
python3 update_tickers.py --exchange NASDAQ_ALL --min-market-cap 1
```

### Advanced Options

```bash
# Clear existing cache before updating
python3 update_tickers.py --exchange SP500 --clear-cache

# Use more concurrent workers (default: 5)
python3 update_tickers.py --exchange NASDAQ_500 --workers 10

# Adjust rate limiting (default: 0.1 seconds between requests)
python3 update_tickers.py --exchange SP500 --rate-limit 0.5

# Don't resume from last position (start fresh)
python3 update_tickers.py --exchange NASDAQ --no-resume

# Use custom cache database
python3 update_tickers.py --exchange SP500 --cache-db /path/to/cache.db
```

### Monitoring Cache Status

```bash
# Check overall cache status
python3 check_cache_status.py

# Check specific exchange
python3 check_cache_status.py --exchange NASDAQ

# Check specific ticker
python3 check_cache_status.py --ticker AAPL
```

## Integration with CSP Scanner

The CSP scanner automatically uses cached ticker data when available:

1. **Quality filtering**: Uses cached fundamental data (market cap, P/E, EPS, returns)
2. **IV rank lookup**: Uses cached IV rank instead of recalculating
3. **Current price**: Uses cached price data when fresh enough

To force the scanner to bypass cache:
```bash
python3 csp_scanner.py --no-cache
```

## Scheduling Automatic Updates

See [scheduler_setup.md](scheduler_setup.md) for detailed instructions on setting up daily automatic updates.

### Quick Cron Setup (macOS/Linux)

```bash
# Edit crontab
crontab -e

# Add daily update jobs
0 6 * * * cd /Users/sheats/options && python3 update_tickers.py --exchange SP500 >> logs/sp500.log 2>&1
0 7 * * * cd /Users/sheats/options && python3 update_tickers.py --exchange NASDAQ >> logs/nasdaq.log 2>&1
```

## Architecture

### Cache Provider Extensions

The `SQLiteCacheProvider` has been extended with new methods:

- `save_ticker_data()`: Stores comprehensive ticker data as JSON
- `get_ticker_data()`: Retrieves cached ticker data with age checking
- `get_cached_tickers()`: Lists all cached tickers
- `clear_ticker_data()`: Clears ticker data cache
- `get_ticker_cache_stats()`: Returns cache statistics

### Database Schema

```sql
-- New ticker_data table
CREATE TABLE ticker_data (
    ticker TEXT,
    exchange TEXT,
    data TEXT,  -- JSON blob with all ticker data
    last_updated TIMESTAMP,
    PRIMARY KEY (ticker, exchange)
);

-- Indexes for efficient lookups
CREATE INDEX idx_ticker_data_lookup ON ticker_data(exchange, last_updated);
CREATE INDEX idx_ticker_data_ticker ON ticker_data(ticker);
```

## Error Handling & Recovery

### Automatic Resume

The script automatically tracks progress and can resume from interruptions:

```bash
# If interrupted (Ctrl+C or system shutdown)
# Just run the same command again - it will skip already cached tickers
python3 update_tickers.py --exchange NASDAQ_500
```

### Retry Logic

- Failed tickers are retried up to 3 times with exponential backoff
- Rate limiting prevents API throttling
- Errors are logged but don't stop the entire process

### Graceful Shutdown

- Handles SIGINT (Ctrl+C) and SIGTERM signals
- Completes current ticker before shutting down
- Safe to interrupt at any time

## Best Practices

1. **Run during off-market hours** to get stable end-of-day data
2. **Stagger update times** for different exchanges to avoid API overload
3. **Monitor logs** regularly for errors
4. **Clear cache monthly** to ensure data freshness
5. **Use market cap filters** for NASDAQ_ALL to limit scope

## Troubleshooting

### Common Issues

1. **Rate limit errors**
   - Solution: Reduce workers (`--workers 3`) or increase delay (`--rate-limit 0.5`)

2. **Memory issues with large exchanges**
   - Solution: Use market cap filter (`--min-market-cap 5`)

3. **Stale data after market close**
   - Solution: Schedule updates after 5 PM ET when data is settled

4. **Database locked errors**
   - Solution: Ensure only one update process runs at a time

### Logs and Debugging

- Check `update_tickers.log` for detailed execution logs
- Use `--debug` flag with CSP scanner to see cache hit/miss info
- Monitor cache age with `check_cache_status.py`

## Performance Tuning

### Optimal Settings by Exchange

- **S&P 500**: 5-8 workers, 0.1s rate limit
- **NASDAQ-100**: 5 workers, 0.1s rate limit  
- **NASDAQ-500**: 5-10 workers, 0.15s rate limit
- **NASDAQ_ALL**: 3-5 workers, 0.2s rate limit, use market cap filter

### Cache Lifetime

Default cache lifetime is 24 hours. Adjust in `constants.py`:
```python
CACHE_LIFETIME_HOURS = 24  # Increase for less frequent updates
```

## Future Enhancements

Potential improvements:

1. **Differential updates**: Only update changed data
2. **Real-time updates**: Stream updates during market hours
3. **Distributed caching**: Share cache across multiple machines
4. **Data compression**: Compress JSON blobs for space efficiency
5. **Historical tracking**: Store historical snapshots for backtesting