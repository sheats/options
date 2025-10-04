# CSP Hotlist Generation

The revised `update_tickers.py` script now focuses on creating a "hotlist" of stocks ideal for Cash-Secured Put (CSP) strategies.

## Hotlist Criteria

The hotlist identifies stocks that meet ALL of these strict criteria:

### Quality & Size
- **Market Cap**: > $50B (large, stable companies)
- **Forward EPS**: > $0 (profitable)

### Ownership & Stability
- **Institutional Ownership**: > 50% (institutional confidence)
- **Beta**: < 1.2 (lower volatility than market)
- **Debt/Equity**: < 1.0 (conservative leverage)

### Growth & Value
- **EPS Growth**: > 5% annually (growing earnings)
- **Dividend Yield**: > 1% (income component)
- **Analyst Rating**: < 2.5 (buy recommendation)

### Technical & Options
- **IV Rank**: > 50% (elevated premium)
- **RSI**: < 50 preferred (oversold entry)
- **MACD**: > 0 preferred (positive momentum)
- **1-Year Return**: Positive or strong momentum

## Hotlist Scoring

Each stock receives a score based on:
```
score = iv_rank × (2 - beta) × institutional_ownership × rsi_factor × momentum_factor
```

Where:
- Higher IV rank = higher score (more premium)
- Lower beta = higher score (more stability)
- Higher institutional ownership = higher score
- RSI < 50 = 1.0x multiplier, else 0.5x
- MACD > 0 = 1.2x multiplier, else 1.0x

## Usage

### Generate Hotlist Only (from existing cache)
```bash
python update_tickers.py --exchange SP500 --hotlist-only
```

### Update Data and Generate Hotlist
```bash
# S&P 500
python update_tickers.py --exchange SP500

# NASDAQ-500
python update_tickers.py --exchange NASDAQ_500

# Top 500 NASDAQ stocks by market cap (limited for performance)
python update_tickers.py --exchange NASDAQ_ALL
```

### Customize Hotlist Parameters
```bash
# Higher IV rank requirement
python update_tickers.py --exchange NASDAQ_500 --min-iv-rank 60

# For cron job (quiet mode)
python update_tickers.py --exchange SP500 --quiet
```

## Output

The script generates:
1. Console output showing top 3 stocks
2. CSV file: `hotlist_{exchange}_{date}.csv`
3. Database table: `hotlist` in cache

### Sample Output
```
CSP Hotlist for SP500 - Top Stocks for Cash-Secured Puts
====================================================================================================

Sample (Top 3):

AAPL: Score=85.2, IV_rank=60.5%, Beta=1.10
  Inst. Own: 62.3%, D/E: 0.85, RSI: 42.5
  Div Yield: 1.2%, 1Y Return: 15.3%, Price: $175.50

MSFT: Score=82.7, IV_rank=55.2%, Beta=0.95  
  Inst. Own: 71.5%, D/E: 0.65, RSI: 38.2
  Div Yield: 1.5%, 1Y Return: 22.1%, Price: $425.75

JPM: Score=78.3, IV_rank=52.8%, Beta=1.15
  Inst. Own: 68.9%, D/E: 0.92, RSI: 45.7  
  Div Yield: 2.8%, 1Y Return: 18.5%, Price: $165.25

Full hotlist with 23 stocks saved to: hotlist_SP500_2025-01-10.csv
```

## Integration with CSP Scanner

Use the hotlist as input for the CSP scanner:
```bash
# Extract tickers from hotlist
tickers=$(awk -F',' 'NR>1 {print $1}' hotlist_SP500_2025-01-10.csv | tr '\n' ',')

# Run CSP scanner on hotlist stocks
python csp_scanner.py --stocks "$tickers" --min-iv-rank 40
```

## Daily Workflow

1. **Morning (5 AM)**: Update ticker data and generate hotlist
   ```bash
   python update_tickers.py --exchange SP500 --quiet
   ```

2. **Market Hours**: Use hotlist for CSP scanning
   ```bash
   python csp_scanner.py --stocks "$(cat hotlist_SP500_*.csv | awk -F',' 'NR>1 {print $1}' | head -20 | tr '\n' ',')"
   ```

## Database Schema

The hotlist table stores:
- ticker, exchange (primary keys)
- score, iv_rank, beta, rsi_14
- institutional_ownership, debt_to_equity
- earnings_growth, dividend_yield
- recommendation_mean, market_cap_b
- full data as JSON
- last_updated timestamp

## Performance Notes

- NASDAQ_ALL is limited to top 500 by market cap for performance
- Hotlist generation takes ~1 second from cached data
- Full update takes 10-30 minutes depending on exchange