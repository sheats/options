# Cash-Secured Put (CSP) Scanner

A sophisticated Python tool for identifying optimal cash-secured put opportunities with minimal loss potential, designed for quantitative options income strategies.

## Project Structure

```
options/
├── csp_scanner.py           # Main scanner script (consolidated version)
├── modules/                 # Business logic modules
│   ├── __init__.py
│   ├── constants.py        # Tunable parameters
│   ├── stock_filter.py     # Stock quality filtering
│   ├── support_calculator.py # Technical support levels
│   ├── option_analyzer.py  # Option chain analysis
│   └── scoring.py          # Opportunity scoring
├── data_providers/          # Data provider abstraction
│   ├── __init__.py
│   ├── base.py             # Abstract base class
│   ├── yfinance_provider.py    # Yahoo Finance implementation
│   └── tastytrade_provider.py  # TastyTrade implementation (optional)
├── cache_providers/         # Cache provider abstraction
│   ├── __init__.py
│   ├── base.py             # Abstract base class
│   └── sqlite_provider.py  # SQLite cache implementation
├── nasdaq_fetcher.py       # NASDAQ stock universe fetcher
├── backtest_csp.py         # Backtesting tool
├── requirements.txt        # Python dependencies
├── tests/                  # Unit tests
├── archive/                # Historical versions
│   ├── csp_scanner_original.py
│   └── csp_scanner_refactored.py
└── README.md              # This file
```

## Features

### Core Features
- **Quality Stock Filtering**: Screens for large-cap stocks with strong fundamentals (market cap >$10B, P/E <40, positive forward EPS)
- **Multi-Method Support Calculation**: Combines 200-day SMA, 50-day near-term support, 52-week lows, local minima detection, and K-means clustering
- **Risk-Adjusted Scoring**: Prioritizes high probability trades with optimal risk/reward ratios
- **Earnings Avoidance**: Automatically skips options with earnings within 7 days of expiration
- **IV Rank Filtering**: Focuses on elevated IV environments (default >20% IV rank) for premium maximization
- **Greek-Based Selection**: Targets delta range of -0.45 to -0.05 for balanced risk/reward

### Advanced Features
- **Multiple Data Providers**: Support for Yahoo Finance and TastyTrade APIs
- **Caching System**: SQLite-based cache for improved performance and reduced API calls
- **NASDAQ Integration**: Access to NASDAQ-100, NASDAQ 500, and all NASDAQ stocks
- **Market Cap Filtering**: Filter stocks by minimum market capitalization
- **Modular Architecture**: Clean separation of concerns for easy customization
- **Progress Tracking**: Visual progress bars for large stock universe scans

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd options

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
# Scan default S&P 500 leaders
python csp_scanner.py

# Scan NASDAQ-100 stocks
python csp_scanner.py --exchange NASDAQ

# Scan NASDAQ 500 (larger universe)
python csp_scanner.py --exchange NASDAQ_500

# Scan all NASDAQ stocks with market cap > $5B
python csp_scanner.py --exchange NASDAQ_ALL --min-market-cap 5

# Scan specific stocks
python csp_scanner.py --stocks AAPL,MSFT,GOOGL,AMZN

# Limit to 4-week expirations with lower IV requirement
python csp_scanner.py --max-weeks 4 --min-iv-rank 15

# Wider support buffer for more opportunities
python csp_scanner.py --support-buffer 0.05

# Save results to CSV
python csp_scanner.py --output results.csv --top 50

# Debug mode for detailed filtering information
python csp_scanner.py --debug --stocks AAPL
```

### Command Line Arguments

#### Basic Options
- `--stocks`: Comma-separated list of tickers (overrides exchange selection)
- `--exchange`: Stock universe to scan (choices: SP500, NASDAQ, NASDAQ_500, NASDAQ_ALL; default: SP500)
- `--max-weeks`: Maximum weeks to expiration (default: 8)
- `--min-iv-rank`: Minimum IV rank percentage (default: 20)
- `--support-buffer`: Support level buffer percentage (default: 0.02)
- `--no-support-filter`: Disable support level filtering
- `--output`: Path to save CSV results
- `--top`: Number of top opportunities to display (default: 20)

#### Advanced Options
- `--provider`: Data provider to use (choices: yfinance, tastytrade; default: yfinance)
- `--tt-username`: TastyTrade username (required for tastytrade provider)
- `--tt-password`: TastyTrade password (required for tastytrade provider)
- `--date`: Analysis date in YYYY-MM-DD format (default: today)
- `--min-market-cap`: Minimum market cap filter in billions (e.g., 10 for $10B)
- `--debug`: Enable debug logging for detailed output

#### Cache Options
- `--no-cache`: Disable caching
- `--clear-cache`: Clear cache before running scan
- `--cache-db`: Path to cache database file (default: csp_scanner_cache.db)

## Output Columns

- **Ticker**: Stock symbol
- **Expiration**: Option expiration date
- **Days to Exp**: Days until expiration
- **Strike**: Put strike price
- **Premium**: Option premium (mid-price)
- **Delta**: Option delta (assignment probability indicator)
- **Theta**: Time decay value
- **Gamma**: Rate of delta change
- **Daily Premium**: Premium divided by days to expiration
- **Annualized ROC**: Return on collateral annualized
- **POP**: Probability of Profit (1 - |delta|)
- **Support**: Calculated support level (multi-method average)
- **Near Term Support**: 50-day moving average support
- **Current Price**: Current stock price
- **IV Rank**: Implied volatility rank (52-week)
- **Score**: Risk-adjusted opportunity score
- **Max Contracts**: Maximum contracts based on 5% position sizing

## Strategy Overview

The scanner implements a flexible CSP strategy that can be tuned from conservative to aggressive:

### Conservative Settings (Default)
- Target: 10-15% annualized returns
- Win rate: 80-90%
- Settings: `--min-iv-rank 20 --support-buffer 0.02`

### Moderate Settings
- Target: 15-25% annualized returns
- Win rate: 70-80%
- Settings: `--min-iv-rank 15 --support-buffer 0.03 --exchange NASDAQ_500`

### Aggressive Settings
- Target: 25-35% annualized returns
- Win rate: 60-70%
- Settings: `--min-iv-rank 10 --support-buffer 0.05 --no-support-filter --exchange NASDAQ_ALL`

Key principles:
1. Only sell puts on quality stocks you'd want to own
2. Strike prices below strong technical support levels
3. High IV environments for enhanced premiums
4. Short duration (1-8 weeks) for faster theta decay
5. Avoid earnings to reduce volatility risk

## Risk Management

- **Position Sizing**: Limit each position to 5% of account
- **Diversification**: Spread across uncorrelated sectors
- **Exit Strategy**: Close at 50% profit or roll at 21 DTE
- **Assignment**: If assigned, implement wheel strategy

## Disclaimer

This tool is for educational and informational purposes only. It is not financial advice. Options trading involves substantial risk and is not suitable for all investors. Always conduct your own research and consider consulting with a financial advisor.

## Configuration

### Tuning the Scanner

The scanner's behavior can be fine-tuned by modifying constants in `modules/constants.py`:

- **MIN_IV_RANK**: Increase for higher volatility opportunities
- **MIN_DELTA/MAX_DELTA**: Adjust delta range for risk tolerance
- **MIN_DAILY_PREMIUM**: Minimum daily premium requirement
- **MIN_ANNUALIZED_ROC**: Minimum return on collateral
- **MIN_MARKET_CAP**: Quality filter for stock selection
- **MAX_PE_RATIO**: Maximum P/E ratio filter

### Cache Configuration

The scanner includes an SQLite cache to improve performance:
- Cache lifetime: 24 hours (configurable)
- Stores quality filter results to reduce API calls
- Clear cache with `--clear-cache` flag
- Disable with `--no-cache` flag

## Testing

```bash
# Run all tests
python run_tests.py

# Run specific test module
pytest tests/test_stock_filter.py -v

# Run with coverage
pytest --cov=modules tests/
```

## Contributing

Pull requests welcome! Please:
1. Follow PEP 8 standards
2. Include appropriate error handling
3. Add tests for new functionality
4. Update documentation as needed

## License

MIT License - see LICENSE file for details