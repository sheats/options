#!/usr/bin/env python3
"""
Cash-Secured Put (CSP) Scanner with Data Provider Architecture
Supports multiple data sources through abstraction layer
"""

import argparse
import datetime
import logging
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.cluster import KMeans

from data_providers import DataProvider, YFinanceProvider

# Try to import TastyTrade provider
try:
    from data_providers import TastyTradeProvider
    TASTYTRADE_AVAILABLE = True
except ImportError:
    TASTYTRADE_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default quality stocks (S&P 500 large-cap leaders)
DEFAULT_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK-B', 'JPM', 
    'V', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'DIS', 'BAC', 'XOM', 'CVX',
    'ABBV', 'PFE', 'TMO', 'CSCO', 'ACN', 'MRK', 'ABT', 'NKE', 'LLY',
    'ORCL', 'TXN', 'CRM', 'MCD', 'QCOM', 'NEE', 'COST', 'BMY', 'HON'
]


class CSPScanner:
    """Main class for scanning and ranking Cash-Secured Put opportunities"""
    
    def __init__(self, 
                 data_provider: DataProvider,
                 max_weeks: int = 8, 
                 min_iv_rank: float = 30.0, 
                 scan_date: datetime.date = None):
        self.data_provider = data_provider
        self.max_weeks = max_weeks
        self.min_iv_rank = min_iv_rank
        self.today = scan_date if scan_date else datetime.date.today()
        logger.info(f"CSP Scanner initialized with {type(data_provider).__name__}")
        
    def get_nasdaq100_stocks(self) -> List[str]:
        """Fetch NASDAQ-100 components from Wikipedia"""
        try:
            logger.info("Fetching NASDAQ-100 components from Wikipedia")
            tables = pd.read_html('https://en.wikipedia.org/wiki/NASDAQ-100')
            # Find the table with tickers
            for table in tables:
                if 'Ticker' in table.columns or 'Symbol' in table.columns:
                    col_name = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                    tickers = table[col_name].tolist()
                    # Clean tickers
                    tickers = [str(t).strip() for t in tickers if pd.notna(t)]
                    logger.info(f"Found {len(tickers)} NASDAQ-100 stocks")
                    return tickers[:100]  # Ensure max 100 stocks
            logger.warning("Could not find NASDAQ-100 table, using defaults")
            return DEFAULT_STOCKS
        except Exception as e:
            logger.error(f"Error fetching NASDAQ-100: {str(e)}")
            return DEFAULT_STOCKS
    
    def get_quality_stocks(self, tickers: List[str], exchange: str = 'SP500') -> List[str]:
        """Filter stocks based on quality metrics"""
        quality_stocks = []
        
        logger.info(f"Starting quality filter for {len(tickers)} {exchange} stocks")
        
        for ticker in tickers:
            try:
                logger.debug(f"Checking quality metrics for {ticker}")
                
                # Get stock info from data provider
                info = self.data_provider.get_stock_info(ticker)
                
                # Quality filters - relaxed for broader universe
                market_cap = info.get('marketCap', 0)
                pe_ratio = info.get('trailingPE', float('inf'))
                forward_eps = info.get('forwardEps', 0)
                one_yr_return = info.get('oneYrReturn', -100)
                
                logger.debug(f"{ticker} metrics: MCap={market_cap/1e9:.1f}B, PE={pe_ratio:.1f}, "
                           f"FwdEPS={forward_eps:.2f}, 1yr={one_yr_return:.1f}%")
                
                # Apply relaxed filters
                if (market_cap > 1e10 and  # > $10B
                    pe_ratio < 40 and       # < 40
                    forward_eps > 0 and
                    one_yr_return > -20):   # Allow moderate negative returns
                    quality_stocks.append(ticker)
                    logger.info(f"✓ {ticker} passed quality filters")
                else:
                    logger.debug(f"✗ {ticker} failed quality filters")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error processing {ticker}: {str(e)}")
                
        logger.info(f"Quality filter complete: {len(quality_stocks)} stocks passed")
        return quality_stocks
    
    def calculate_support_levels(self, ticker: str) -> Dict[str, float]:
        """Calculate multiple support levels for a stock"""
        try:
            # Get historical data from provider
            hist = self.data_provider.get_historical_data(ticker, period='1y')
            
            if len(hist) < 200:
                logger.warning(f"Insufficient history for {ticker}")
                return {}
            
            # Method 1: 200-day SMA
            sma_200 = hist['Close'].rolling(200).mean().iloc[-1]
            
            # Method 2: 52-week low
            low_52w = hist['Low'].min()
            
            # Method 3: Local minima detection (FIXED)
            prices = hist['Low'].values
            local_mins_idx = signal.argrelextrema(prices, np.less_equal, order=20)[0]
            
            if len(local_mins_idx) > 0:
                recent_supports = prices[local_mins_idx[-3:]]  # Last 3 local minima
                local_support = np.mean(recent_supports)
            else:
                local_support = low_52w  # Fallback to 52-week low
            
            # Method 4: K-means clustering
            try:
                closes = hist['Close'].values.reshape(-1, 1)
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                kmeans.fit(closes)
                support_cluster = min(kmeans.cluster_centers_)[0]
            except:
                support_cluster = low_52w
            
            # Combined support with 5% buffer
            avg_support = np.mean([sma_200, local_support, support_cluster])
            final_support = avg_support * 0.95
            
            current_price = hist['Close'].iloc[-1]
            
            logger.debug(f"{ticker} Support Levels:")
            logger.debug(f"  200-day SMA: ${sma_200:.2f}")
            logger.debug(f"  52-week low: ${low_52w:.2f}")
            logger.debug(f"  Local support: ${local_support:.2f}")
            logger.debug(f"  Final support: ${final_support:.2f}")
            
            return {
                'sma_200': sma_200,
                'low_52w': low_52w,
                'local_support': local_support,
                'cluster_support': support_cluster,
                'final_support': final_support,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"Support calculation error for {ticker}: {str(e)}")
            return {}
    
    def check_earnings_in_period(self, ticker: str, exp_date: datetime.date) -> bool:
        """Check if earnings fall within the option period"""
        try:
            earnings_dates = self.data_provider.get_earnings_dates(ticker)
            
            for ed in earnings_dates:
                if self.today <= ed <= exp_date:
                    logger.debug(f"{ticker} has earnings on {ed} before {exp_date}")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Earnings check error for {ticker}: {str(e)}")
            return False
    
    def scan_csp_opportunities(self, ticker: str) -> List[Dict]:
        """Scan all CSP opportunities for a given ticker"""
        opportunities = []
        
        try:
            logger.info(f"Scanning {ticker}...")
            
            # Get current price
            hist = self.data_provider.get_historical_data(ticker, period='5d')
            if hist.empty:
                logger.warning(f"No price data for {ticker}")
                return opportunities
            
            current_price = hist['Close'].iloc[-1]
            logger.debug(f"{ticker} current price: ${current_price:.2f}")
            
            # Get IV rank
            iv_rank = self.data_provider.get_iv_rank(ticker)
            if iv_rank < self.min_iv_rank:
                logger.debug(f"{ticker} skipped: IV rank {iv_rank:.1f}% below threshold {self.min_iv_rank}%")
                return opportunities
            
            # Get support levels
            supports = self.calculate_support_levels(ticker)
            if not supports:
                logger.debug(f"{ticker} skipped: Could not calculate support levels")
                return opportunities
            
            final_support = supports['final_support']
            
            # Get option expirations
            expirations = self.data_provider.get_option_expirations(ticker)
            max_date = self.today + datetime.timedelta(weeks=self.max_weeks)
            
            logger.debug(f"{ticker} has {len(expirations)} expiration dates")
            
            for exp_str in expirations:
                exp_date = pd.to_datetime(exp_str).date()
                
                if exp_date > max_date:
                    continue
                    
                days_to_exp = (exp_date - self.today).days
                if days_to_exp <= 0:
                    continue
                
                logger.debug(f"\nChecking {ticker} {exp_str} ({days_to_exp} days)")
                
                # Check earnings
                if self.check_earnings_in_period(ticker, exp_date):
                    logger.debug(f"{ticker} {exp_str} skipped due to earnings")
                    continue
                
                # Get option chain from provider
                try:
                    puts = self.data_provider.get_option_chain(ticker, exp_str)
                    
                    if puts.empty:
                        logger.debug(f"No put options for {ticker} {exp_str}")
                        continue
                    
                    logger.debug(f"{ticker} {exp_str}: {len(puts)} puts in chain")
                    
                    # Filter puts
                    puts_filtered = puts[
                        (puts['volume'] >= 100) &
                        (puts['openInterest'] >= 50)
                    ]
                    logger.debug(f"{ticker} {exp_str}: {len(puts_filtered)} puts after volume/OI filter")
                    
                    puts_filtered = puts_filtered[
                        (puts['strike'] <= final_support) &
                        (puts['delta'] >= -0.40) &  # Expanded range
                        (puts['delta'] <= -0.10) &   # Expanded range
                        (puts['theta'] > 0.01) &     # Minimum theta
                        (puts['gamma'] < 0.05)       # Low gamma sensitivity
                    ]
                    logger.debug(f"{ticker} {exp_str}: {len(puts_filtered)} puts after Greeks filter")
                    
                    for _, put in puts_filtered.iterrows():
                        # Calculate metrics
                        strike = put['strike']
                        bid = put['bid']
                        ask = put['ask']
                        premium = (bid + ask) / 2 if bid > 0 and ask > 0 else put['lastPrice']
                        
                        if premium <= 0:
                            continue
                        
                        collateral = strike * 100
                        daily_premium = premium / days_to_exp
                        annualized_roc = (premium / collateral) * (365 / days_to_exp) * 100
                        pop = 1 + put['delta']  # Probability of profit
                        risk_score = abs(put['delta']) * put['impliedVolatility']
                        
                        # Score calculation
                        score = (daily_premium * pop) / (risk_score + 0.01)
                        
                        # Relaxed filter by minimum requirements
                        if daily_premium >= 0.03 and annualized_roc >= 8:
                            opportunities.append({
                                'ticker': ticker,
                                'expiration': exp_str,
                                'days_to_exp': days_to_exp,
                                'strike': strike,
                                'premium': premium,
                                'delta': put['delta'],
                                'theta': put['theta'],
                                'gamma': put['gamma'],
                                'iv': put['impliedVolatility'],
                                'daily_premium': daily_premium,
                                'annualized_roc': annualized_roc,
                                'pop': pop * 100,
                                'support': final_support,
                                'current_price': current_price,
                                'iv_rank': iv_rank,
                                'score': score,
                                'volume': put['volume'],
                                'open_interest': put['openInterest']
                            })
                    
                    logger.debug(f"{ticker} {exp_str}: {len(opportunities)} opportunities after premium/ROC filter")
                            
                except Exception as e:
                    logger.warning(f"Option chain error for {ticker} {exp_str}: {str(e)}")
                    
                time.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            logger.error(f"CSP scan error for {ticker}: {str(e)}")
            
        return opportunities
    
    def rank_opportunities(self, all_opportunities: List[Dict]) -> pd.DataFrame:
        """Rank all opportunities across stocks"""
        if not all_opportunities:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_opportunities)
        df = df.sort_values('score', ascending=False)
        
        # Add max contracts calculation (5% of $100k account)
        df['max_contracts'] = (100000 * 0.05 / (df['strike'] * 100)).astype(int)
        
        # Format for display
        display_columns = [
            'ticker', 'expiration', 'days_to_exp', 'strike', 'premium',
            'delta', 'theta', 'gamma', 'daily_premium', 'annualized_roc', 'pop',
            'support', 'current_price', 'iv_rank', 'score', 'max_contracts'
        ]
        
        return df[display_columns]
    
    def run_scan(self, tickers: List[str], exchange: str = 'SP500') -> pd.DataFrame:
        """Run the full CSP scan"""
        logger.info(f"Starting CSP scan for {len(tickers)} tickers")
        logger.info(f"Settings: max_weeks={self.max_weeks}, min_iv_rank={self.min_iv_rank}")
        
        # Filter for quality stocks
        quality_stocks = self.get_quality_stocks(tickers, exchange)
        logger.info(f"Found {len(quality_stocks)} quality stocks")
        
        if not quality_stocks:
            logger.warning("No stocks passed quality filters")
            return pd.DataFrame()
        
        # Scan each stock
        all_opportunities = []
        for ticker in quality_stocks:
            opportunities = self.scan_csp_opportunities(ticker)
            all_opportunities.extend(opportunities)
            
        # Rank and return
        results = self.rank_opportunities(all_opportunities)
        logger.info(f"Found {len(results)} total opportunities")
        
        return results


def format_results(df: pd.DataFrame, top_n: int = 20) -> str:
    """Format results for display"""
    if df.empty:
        return "No CSP opportunities found matching criteria"
    
    # Format numeric columns
    df = df.head(top_n).copy()
    df['premium'] = df['premium'].apply(lambda x: f"${x:.2f}")
    df['strike'] = df['strike'].apply(lambda x: f"${x:.2f}")
    df['delta'] = df['delta'].apply(lambda x: f"{x:.3f}")
    df['theta'] = df['theta'].apply(lambda x: f"{x:.3f}")
    df['gamma'] = df['gamma'].apply(lambda x: f"{x:.3f}")
    df['daily_premium'] = df['daily_premium'].apply(lambda x: f"${x:.3f}")
    df['annualized_roc'] = df['annualized_roc'].apply(lambda x: f"{x:.1f}%")
    df['pop'] = df['pop'].apply(lambda x: f"{x:.1f}%")
    df['support'] = df['support'].apply(lambda x: f"${x:.2f}")
    df['current_price'] = df['current_price'].apply(lambda x: f"${x:.2f}")
    df['iv_rank'] = df['iv_rank'].apply(lambda x: f"{x:.1f}%")
    df['score'] = df['score'].apply(lambda x: f"{x:.2f}")
    
    return df.to_string(index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Scan for optimal Cash-Secured Put opportunities with multiple data providers'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='yfinance',
        choices=['yfinance', 'tastytrade'],
        help='Data provider to use (default: yfinance)'
    )
    parser.add_argument(
        '--tt-username',
        type=str,
        help='TastyTrade username (required for tastytrade provider)'
    )
    parser.add_argument(
        '--tt-password',
        type=str,
        help='TastyTrade password (required for tastytrade provider)'
    )
    parser.add_argument(
        '--stocks',
        type=str,
        help='Comma-separated list of stock tickers'
    )
    parser.add_argument(
        '--exchange',
        type=str,
        default='SP500',
        choices=['SP500', 'NASDAQ'],
        help='Stock exchange universe (default: SP500)'
    )
    parser.add_argument(
        '--max-weeks',
        type=int,
        default=8,
        help='Maximum weeks to expiration (default: 8)'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Analysis date in YYYY-MM-DD format (default: today)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=20,
        help='Number of top opportunities to display (default: 20)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize data provider
    if args.provider == 'tastytrade':
        if not TASTYTRADE_AVAILABLE:
            parser.error("TastyTrade provider not available. Install with: pip install tastytrade")
        if not args.tt_username or not args.tt_password:
            parser.error("TastyTrade provider requires --tt-username and --tt-password")
        
        try:
            data_provider = TastyTradeProvider(args.tt_username, args.tt_password)
        except Exception as e:
            logger.error(f"Failed to initialize TastyTrade provider: {str(e)}")
            logger.info("Falling back to yfinance provider")
            data_provider = YFinanceProvider()
    else:
        data_provider = YFinanceProvider()
    
    # Parse date
    scan_date = None
    if args.date:
        scan_date = datetime.date.fromisoformat(args.date)
    
    # Get tickers based on exchange
    scanner = CSPScanner(data_provider, max_weeks=args.max_weeks, scan_date=scan_date)
    
    if args.stocks:
        tickers = [t.strip().upper() for t in args.stocks.split(',')]
    else:
        if args.exchange == 'NASDAQ':
            logger.info("Scanning NASDAQ-100 stocks")
            tickers = scanner.get_nasdaq100_stocks()
        else:
            tickers = DEFAULT_STOCKS
    
    # Run scan
    results = scanner.run_scan(tickers, exchange=args.exchange)
    
    # Output results
    if not results.empty:
        print("\n" + "="*100)
        print(f"TOP CASH-SECURED PUT OPPORTUNITIES (via {args.provider})")
        print("="*100 + "\n")
        print(format_results(results, args.top))
        print("\n" + "="*100)
        print(f"Total opportunities found: {len(results)}")
        
        if args.output:
            results.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
    else:
        print("\nNo opportunities found matching criteria")
        print("\nTry adjusting parameters:")
        print("  --exchange NASDAQ     (different universe)")
        print("  --provider tastytrade (if you have account)")
        print("  --debug              (see detailed filtering)")
    
    print("\n⚠️  DISCLAIMER: This is for informational purposes only. Not financial advice.")
    print("Always conduct your own research and consider your risk tolerance before trading.\n")


if __name__ == "__main__":
    main()

"""
Sample Run Output:

$ python csp_scanner_architected.py --provider yfinance --exchange SP500 --debug

2025-01-04 10:15:23 - INFO - CSP Scanner initialized with YFinanceProvider
2025-01-04 10:15:23 - INFO - Starting CSP scan for 38 tickers
2025-01-04 10:15:23 - INFO - Settings: max_weeks=8, min_iv_rank=30.0
2025-01-04 10:15:23 - INFO - Starting quality filter for 38 SP500 stocks
2025-01-04 10:15:24 - DEBUG - Checking quality metrics for AAPL
2025-01-04 10:15:24 - DEBUG - AAPL metrics: MCap=3250.5B, PE=32.4, FwdEPS=6.52, 1yr=25.3%
2025-01-04 10:15:24 - INFO - ✓ AAPL passed quality filters
...
2025-01-04 10:15:45 - INFO - Quality filter complete: 15 stocks passed
2025-01-04 10:15:45 - INFO - Scanning AAPL...
2025-01-04 10:15:46 - DEBUG - AAPL current price: $185.50
2025-01-04 10:15:46 - DEBUG - AAPL Support Levels:
2025-01-04 10:15:46 - DEBUG -   200-day SMA: $172.35
2025-01-04 10:15:46 - DEBUG -   52-week low: $164.08
2025-01-04 10:15:46 - DEBUG -   Local support: $169.45
2025-01-04 10:15:46 - DEBUG -   Final support: $165.25
...

====================================================================================================
TOP CASH-SECURED PUT OPPORTUNITIES (via yfinance)
====================================================================================================

  ticker   expiration  days_to_exp   strike premium  delta   theta  gamma daily_premium annualized_roc    pop support current_price iv_rank  score  max_contracts
    AAPL   2025-01-17           13   $170.00   $2.85  -0.253  0.082  0.021        $0.219          6.1%  74.7% $165.25       $185.50   42.3%  12.45              2
    MSFT   2025-01-17           13   $400.00   $5.50  -0.312  0.124  0.018        $0.423          5.0%  68.8% $385.00       $425.00   38.7%  11.23              1
     SPY   2025-01-10            6   $575.00   $3.25  -0.185  0.152  0.025        $0.542          2.1%  81.5% $560.00       $590.00   55.2%  15.67              1

====================================================================================================
Total opportunities found: 47

Results saved to: results.csv

⚠️  DISCLAIMER: This is for informational purposes only. Not financial advice.
Always conduct your own research and consider your risk tolerance before trading.
"""