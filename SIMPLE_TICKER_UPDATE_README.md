# Simple Ticker Update System

A simplified data collection and storage system that fetches comprehensive stock data and stores everything in SQLite for later querying.

## What It Does

- Fetches ALL available data for stocks (no filtering during fetch)
- Stores everything in SQLite database using INSERT OR REPLACE
- Generates optional hotlist based on cached data
- Simple, straightforward, no complex filtering logic

## Usage

### Basic Update
```bash
# Update S&P 500 stocks
python update_tickers_simple.py --exchange SP500

# Update NASDAQ-100
python update_tickers_simple.py --exchange NASDAQ

# Update top 500 NASDAQ by market cap
python update_tickers_simple.py --exchange NASDAQ_500

# Update ALL NASDAQ stocks (3500+)
python update_tickers_simple.py --exchange NASDAQ_ALL
```

### With Hotlist Generation
```bash
# Update and generate hotlist
python update_tickers_simple.py --exchange SP500 --hotlist

# Adjust worker threads for speed
python update_tickers_simple.py --exchange NASDAQ_500 --workers 10 --hotlist
```

## Data Stored

For each ticker, we store:
- Company info (name, sector, industry, website)
- Market data (market cap, shares, float)
- Price data (current, 52-week high/low, moving averages)
- Volume data
- Valuation metrics (P/E, PEG, P/S, P/B)
- Financials (revenue, margins, debt, cash)
- Growth metrics
- Dividends
- Ownership (institutional, insider)
- Risk metrics (beta, short ratio)
- Analyst ratings
- Technical indicators (RSI, MACD)
- Returns (1y, 6m, 3m, etc.)
- Next earnings date
- IV rank (calculated from historical volatility)

## Database Structure

Uses the cache provider's `ticker_data` table:
- `ticker`: Stock symbol (primary key with exchange)
- `exchange`: Exchange name
- `data`: JSON blob with all data
- `last_updated`: Timestamp

## Hotlist Logic

The hotlist is generated from cached data with simple filters:
- Market cap > $10B
- IV rank > 40%
- Score = IV rank × (1 + institutional ownership)

Top stocks are sorted by score and saved to CSV.

## Performance

- ~1.5 seconds per ticker with 5 workers
- ~100 stocks take ~1-2 minutes
- NASDAQ_ALL (3500+ stocks) takes ~30-45 minutes

## Daily Usage

```bash
# Morning update
python update_tickers_simple.py --exchange SP500 --hotlist

# Use with CSP scanner
python csp_scanner.py --exchange SP500
```

The CSP scanner will automatically use the cached data when available, making scans 50-100x faster than fetching live data.

## Notes

- No filtering during data fetch - stores everything
- Market cap filtering happens only in hotlist generation
- Uses yfinance for all data fetching
- Rate limited to avoid API throttling
- Handles errors gracefully - continues on failures