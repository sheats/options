# Cash-Secured Put (CSP) Scanner

A sophisticated Python tool for identifying optimal cash-secured put opportunities with minimal loss potential, designed for quantitative options income strategies.

## Project Structure

```
options/
├── csp_scanner.py           # Main scanner script
├── data_providers/          # Data provider abstraction
│   ├── __init__.py
│   ├── base.py             # Abstract base class
│   ├── yfinance_provider.py    # Yahoo Finance implementation
│   └── tastytrade_provider.py  # TastyTrade implementation (optional)
├── example.py              # Simple usage examples
├── backtest_csp.py         # Backtesting tool
├── requirements.txt        # Python dependencies
├── .flake8                # Code style configuration
└── archive/               # Old versions (for reference)
```

## Features

- **Quality Stock Filtering**: Screens for large-cap stocks with strong fundamentals (market cap >$50B, P/E <30, positive YoY returns)
- **Multi-Method Support Calculation**: Combines 200-day SMA, 52-week lows, local minima detection, and K-means clustering
- **Risk-Adjusted Scoring**: Prioritizes high probability trades with optimal risk/reward ratios
- **Earnings Avoidance**: Automatically skips options that span earnings dates
- **IV Rank Filtering**: Focuses on elevated IV environments (>50% IV rank) for premium maximization
- **Greek-Based Selection**: Targets delta range of -0.15 to -0.30 for ~70-85% win rate

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

# Scan specific stocks
python csp_scanner.py --stocks AAPL,MSFT,GOOGL,AMZN

# Limit to 4-week expirations
python csp_scanner.py --max-weeks 4

# Save results to CSV
python csp_scanner.py --output results.csv --top 50
```

### Command Line Arguments
- `--stocks`: Comma-separated list of tickers (default: S&P 500 large-cap leaders)
- `--max-weeks`: Maximum weeks to expiration (default: 8)
- `--output`: Path to save CSV results
- `--top`: Number of top opportunities to display (default: 20)

## Output Columns

- **Ticker**: Stock symbol
- **Expiration**: Option expiration date
- **Days to Exp**: Days until expiration
- **Strike**: Put strike price
- **Premium**: Option premium (mid-price)
- **Delta**: Option delta (assignment probability indicator)
- **Theta**: Time decay value
- **Daily Premium**: Premium divided by days to expiration
- **Annualized ROC**: Return on collateral annualized
- **POP**: Probability of Profit (1 - |delta|)
- **Support**: Calculated support level
- **Current Price**: Current stock price
- **IV Rank**: Implied volatility rank (52-week)
- **Score**: Risk-adjusted opportunity score

## Strategy Overview

The scanner implements a conservative CSP strategy targeting:
- 15-25% annualized returns
- <10% drawdowns
- 70-85% win rate

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

## Contributing

Pull requests welcome! Please ensure code follows PEP 8 standards and includes appropriate error handling.

## License

MIT License - see LICENSE file for details