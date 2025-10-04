# NASDAQ Stock Fetcher

A comprehensive solution for fetching NASDAQ stocks with multiple data sources, caching, and filtering capabilities.

## Features

- **Multiple Data Sources**:
  - NASDAQ FTP server (official source)
  - NASDAQ Trader API
  - Fallback mechanisms for reliability

- **Stock Universe Options**:
  - NASDAQ-100 (top 100 stocks)
  - NASDAQ-500 (top 500 by market cap)
  - All NASDAQ stocks (3000+ stocks)

- **Filtering & Enhancement**:
  - Filter by minimum market cap
  - Fetch detailed company information
  - Sector and industry classification
  - Financial metrics (P/E, volume, etc.)

- **Performance Features**:
  - Automatic caching (24-hour expiry)
  - Rate limiting to respect API limits
  - Progress bars for large operations
  - Parallel processing for speed

## Installation

The NASDAQ fetcher is integrated into the options project. Make sure you have all dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line with CSP Scanner

The NASDAQ fetcher is fully integrated with the CSP scanner:

```bash
# Scan NASDAQ-100 stocks (original behavior)
python csp_scanner.py --exchange NASDAQ

# Scan top 500 NASDAQ stocks by market cap
python csp_scanner.py --exchange NASDAQ_500

# Scan all NASDAQ stocks with >$10B market cap
python csp_scanner.py --exchange NASDAQ_ALL --min-market-cap 10

# Scan NASDAQ-500 with custom parameters
python csp_scanner.py --exchange NASDAQ_500 --min-iv-rank 15 --support-buffer 0.03
```

### Python API

```python
from nasdaq_fetcher import NASDAQFetcher, get_nasdaq_stocks

# Initialize fetcher
fetcher = NASDAQFetcher(cache_enabled=True)

# Get NASDAQ-100
nasdaq_100 = fetcher.get_nasdaq_100()

# Get NASDAQ-500 with company info
nasdaq_500 = fetcher.get_nasdaq_500(include_info=True)

# Get all NASDAQ stocks with market cap > $5B
large_caps = fetcher.get_all_nasdaq_stocks(
    min_market_cap=5e9,
    include_info=True
)

# Convenience function
top_200 = get_nasdaq_stocks(count=200, min_market_cap=10e9)
```

## Data Structure

When fetching with `include_info=True`, each stock includes:

```python
{
    'symbol': 'AAPL',
    'name': 'Apple Inc.',
    'exchange': 'NASDAQ',
    'market_cap': 3.2e12,
    'sector': 'Technology',
    'industry': 'Consumer Electronics',
    'pe_ratio': 32.5,
    'forward_pe': 28.7,
    'price': 195.50,
    'volume': 45000000,
    'avg_volume': 50000000,
    '52_week_high': 199.62,
    '52_week_low': 164.08,
    'dividend_yield': 0.0044,
    'beta': 1.29,
    'employees': 164000,
    'website': 'https://www.apple.com',
    'description': 'Apple Inc. designs, manufactures...'
}
```

## Cache Management

The fetcher automatically caches results for 24 hours to improve performance:

```python
# Cache is enabled by default
fetcher = NASDAQFetcher(cache_enabled=True)

# Clear cache when needed
fetcher.clear_cache()

# Disable cache for fresh data
fetcher = NASDAQFetcher(cache_enabled=False)
```

Cache files are stored in `.nasdaq_cache/` directory.

## Rate Limiting

The fetcher implements automatic rate limiting:
- 100ms delay between individual stock info requests
- 1 second delay between batches of 50 stocks
- Parallel processing with 10 threads for efficiency

## Example Scripts

Run the example script to see various usage patterns:

```bash
python example_nasdaq_fetch.py
```

## Performance Tips

1. **Use caching**: Keep cache enabled for repeated runs
2. **Filter early**: Use `min_market_cap` to reduce processing
3. **Skip info when not needed**: Set `include_info=False` for faster results
4. **Use NASDAQ_500**: Good balance between coverage and speed

## Integration with CSP Scanner

The CSP scanner now supports enhanced NASDAQ scanning:

1. **Progress Bars**: Automatically shown for >10 stocks
2. **Smart Filtering**: Quality filters adapted for NASDAQ stocks
3. **Efficient Caching**: Integrated with scanner's cache system

Example workflow:

```bash
# First run - fetches and caches NASDAQ-500
python csp_scanner.py --exchange NASDAQ_500 --min-iv-rank 20

# Second run - uses cached stock list (much faster)
python csp_scanner.py --exchange NASDAQ_500 --min-iv-rank 15

# Clear all caches when needed
python csp_scanner.py --exchange NASDAQ_500 --clear-cache
```

## Troubleshooting

### FTP Connection Issues
If FTP fails, the fetcher automatically falls back to the API source.

### Missing Stock Info
Some stocks may have limited info available. The fetcher handles this gracefully and returns available data.

### Rate Limiting
If you hit rate limits, the fetcher will log warnings but continue processing.

### Cache Issues
If you see stale data, clear the cache:
```python
fetcher.clear_cache()
```

## Future Enhancements

Potential improvements for the NASDAQ fetcher:
- Add more data sources (Finviz, Alpha Vantage)
- Support for other exchanges (NYSE, AMEX)
- Real-time price updates
- Historical market cap tracking
- ETF constituent fetching