#!/usr/bin/env python3
"""
Cash-Secured Put (CSP) Scanner for Options Income Strategies - Fixed Version
Works with available yfinance data (no Greeks required)
"""

import argparse
import datetime
import logging
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import signal
from sklearn.cluster import KMeans

# Suppress yfinance warnings
warnings.filterwarnings('ignore')

# Configure logging with DEBUG level for troubleshooting
logging.basicConfig(
    level=logging.DEBUG,
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
    
    def __init__(self, max_weeks: int = 8, min_iv_rank: float = 30.0):
        self.max_weeks = max_weeks
        self.min_iv_rank = min_iv_rank  # Lowered from 50 to 30 for more results
        self.today = datetime.date.today()
        logger.info(f"CSP Scanner initialized: max_weeks={max_weeks}, min_iv_rank={min_iv_rank}")
        
    def get_quality_stocks(self, tickers: List[str]) -> List[str]:
        """Filter stocks based on quality metrics"""
        quality_stocks = []
        
        logger.info(f"Starting quality filter for {len(tickers)} stocks")
        
        for ticker in tickers:
            try:
                logger.debug(f"\n{'='*50}")
                logger.debug(f"Checking quality metrics for {ticker}")
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Log all available info
                logger.debug(f"{ticker} info keys: {list(info.keys())[:20]}...")
                
                # Quality filters with more lenient thresholds
                market_cap = info.get('marketCap', 0)
                pe_ratio = info.get('trailingPE', float('inf'))
                forward_eps = info.get('forwardEps', 0)
                
                # Calculate 1-year return
                hist = stock.history(period='1y')
                if len(hist) > 0:
                    one_yr_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                    current_price = hist['Close'].iloc[-1]
                else:
                    one_yr_return = -100
                    current_price = 0
                
                logger.debug(f"{ticker} Quality Metrics:")
                logger.debug(f"  Market Cap: ${market_cap/1e9:.1f}B (min: $50B)")
                logger.debug(f"  P/E Ratio: {pe_ratio:.1f} (max: 35)")  # Relaxed from 30
                logger.debug(f"  Forward EPS: ${forward_eps:.2f} (min: >0)")
                logger.debug(f"  1-Year Return: {one_yr_return:.1f}% (min: >-10%)")  # Relaxed
                logger.debug(f"  Current Price: ${current_price:.2f}")
                
                # Apply relaxed filters
                if (market_cap > 3e10 and  # Lowered to $30B
                    pe_ratio < 35 and      # Raised from 30
                    forward_eps > 0 and
                    one_yr_return > -10):   # Allow small negative returns
                    quality_stocks.append(ticker)
                    logger.info(f"✓ {ticker} PASSED quality filters")
                else:
                    failed_reasons = []
                    if market_cap <= 3e10:
                        failed_reasons.append(f"Market cap too low: ${market_cap/1e9:.1f}B")
                    if pe_ratio >= 35:
                        failed_reasons.append(f"P/E too high: {pe_ratio:.1f}")
                    if forward_eps <= 0:
                        failed_reasons.append(f"Negative forward EPS: ${forward_eps:.2f}")
                    if one_yr_return <= -10:
                        failed_reasons.append(f"Poor 1yr return: {one_yr_return:.1f}%")
                    
                    logger.info(f"✗ {ticker} FAILED quality filters: {', '.join(failed_reasons)}")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error processing {ticker}: {str(e)}")
                
        logger.info(f"\nQuality filter complete: {len(quality_stocks)}/{len(tickers)} stocks passed")
        return quality_stocks
    
    def calculate_iv_rank(self, ticker: str) -> float:
        """Calculate IV rank from historical volatility"""
        try:
            logger.debug(f"Calculating IV rank for {ticker}")
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')
            
            # Calculate rolling 30-day volatility as proxy for IV
            returns = hist['Close'].pct_change()
            rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
            
            current_vol = rolling_vol.iloc[-1]
            min_vol = rolling_vol.min()
            max_vol = rolling_vol.max()
            
            if max_vol > min_vol:
                iv_rank = (current_vol - min_vol) / (max_vol - min_vol) * 100
            else:
                iv_rank = 50  # Default if no variation
            
            logger.debug(f"{ticker} IV Rank: {iv_rank:.1f}% (current: {current_vol:.1f}%, min: {min_vol:.1f}%, max: {max_vol:.1f}%)")
            return iv_rank
            
        except Exception as e:
            logger.warning(f"IV rank calculation error for {ticker}: {str(e)}")
            return 0
    
    def calculate_support_levels(self, ticker: str) -> Dict[str, float]:
        """Calculate multiple support levels for a stock"""
        try:
            logger.debug(f"Calculating support levels for {ticker}")
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')
            
            if len(hist) < 200:
                logger.warning(f"Insufficient history for {ticker} ({len(hist)} days)")
                return {}
            
            # Method 1: 200-day SMA
            sma_200 = hist['Close'].rolling(200).mean().iloc[-1]
            
            # Method 2: 52-week low
            low_52w = hist['Low'].min()
            
            # Method 3: Local minima detection
            prices = hist['Low'].values
            local_mins_idx = signal.argrelextrema(prices, np.less, order=20)[0]
            
            if len(local_mins_idx) > 0:
                recent_supports = prices[local_mins_idx[-3:]]  # Last 3 local minima
                local_support = np.mean(recent_supports)
            else:
                local_support = low_52w
            
            # Method 4: K-means clustering
            try:
                closes = hist['Close'].values.reshape(-1, 1)
                kmeans = KMeans(n_clusters=3, random_state=42)
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
            logger.debug(f"  Cluster support: ${support_cluster:.2f}")
            logger.debug(f"  Final support (95%): ${final_support:.2f}")
            logger.debug(f"  Current price: ${current_price:.2f}")
            logger.debug(f"  Support % below: {((current_price - final_support) / current_price * 100):.1f}%")
            
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
    
    def check_earnings_date(self, ticker: str, exp_date: datetime.date) -> bool:
        """Check if earnings fall within the option period"""
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            if calendar is not None and 'Earnings Date' in calendar:
                earnings_dates = calendar['Earnings Date']
                if isinstance(earnings_dates, pd.Timestamp):
                    earnings_dates = [earnings_dates]
                
                for ed in earnings_dates:
                    if self.today <= ed.date() <= exp_date:
                        logger.debug(f"{ticker} has earnings on {ed.date()} before {exp_date}")
                        return True
            
        except Exception as e:
            logger.debug(f"Earnings check error for {ticker}: {str(e)}")
            
        return False
    
    def scan_csp_opportunities(self, ticker: str) -> List[Dict]:
        """Scan all CSP opportunities for a given ticker"""
        opportunities = []
        
        try:
            logger.info(f"\nScanning CSP opportunities for {ticker}")
            stock = yf.Ticker(ticker)
            
            # Get current price
            hist = stock.history(period='1d')
            if hist.empty:
                logger.warning(f"No price data for {ticker}")
                return opportunities
                
            current_price = hist['Close'].iloc[-1]
            logger.debug(f"{ticker} current price: ${current_price:.2f}")
            
            # Get IV rank
            iv_rank = self.calculate_iv_rank(ticker)
            if iv_rank < self.min_iv_rank:
                logger.info(f"{ticker} IV rank {iv_rank:.1f}% below threshold {self.min_iv_rank}%")
                return opportunities
            
            # Get support levels
            supports = self.calculate_support_levels(ticker)
            if not supports:
                logger.warning(f"Could not calculate supports for {ticker}")
                return opportunities
            
            final_support = supports['final_support']
            
            # Get option expirations
            expirations = stock.options
            logger.debug(f"{ticker} has {len(expirations)} expiration dates")
            
            max_date = self.today + datetime.timedelta(weeks=self.max_weeks)
            
            for exp_str in expirations:
                exp_date = pd.to_datetime(exp_str).date()
                
                if exp_date > max_date:
                    continue
                    
                days_to_exp = (exp_date - self.today).days
                if days_to_exp <= 0:
                    continue
                
                logger.debug(f"\nChecking {ticker} {exp_str} ({days_to_exp} days)")
                
                # Check earnings
                if self.check_earnings_date(ticker, exp_date):
                    logger.info(f"Skipping {ticker} {exp_str} due to earnings")
                    continue
                
                # Get option chain
                try:
                    opt_chain = stock.option_chain(exp_str)
                    puts = opt_chain.puts
                    
                    logger.debug(f"Found {len(puts)} puts for {ticker} {exp_str}")
                    logger.debug(f"Put columns: {puts.columns.tolist()}")
                    
                    # Since we don't have Greeks, use moneyness-based filtering
                    # Target strikes 15-25% below current price (similar to -0.15 to -0.30 delta)
                    target_strike_high = current_price * 0.85
                    target_strike_low = current_price * 0.75
                    
                    # Initial filters
                    filtered_puts = puts[
                        (puts['volume'] >= 50) &  # Lowered from 100
                        (puts['openInterest'] >= 25) &  # Lowered from 50
                        (puts['strike'] <= target_strike_high) &
                        (puts['strike'] >= target_strike_low) &
                        (puts['strike'] <= final_support * 1.05)  # Allow strikes slightly above support
                    ]
                    
                    logger.debug(f"After filtering: {len(filtered_puts)} puts remain")
                    logger.debug(f"  Volume filter: {len(puts[puts['volume'] >= 50])}")
                    logger.debug(f"  OI filter: {len(puts[puts['openInterest'] >= 25])}")
                    logger.debug(f"  Strike range: ${target_strike_low:.2f} - ${target_strike_high:.2f}")
                    logger.debug(f"  Support filter: strikes <= ${final_support * 1.05:.2f}")
                    
                    for idx, put in filtered_puts.iterrows():
                        # Calculate metrics
                        strike = put['strike']
                        bid = put['bid']
                        ask = put['ask']
                        
                        # Skip if no bid/ask
                        if bid == 0 or ask == 0:
                            continue
                            
                        premium = (bid + ask) / 2
                        
                        if premium <= 0:
                            continue
                        
                        # Estimate delta based on moneyness
                        moneyness = strike / current_price
                        estimated_delta = -0.5 * (1 - moneyness)  # Rough approximation
                        
                        collateral = strike * 100
                        daily_premium = premium / days_to_exp
                        annualized_roc = (premium / collateral) * (365 / days_to_exp) * 100
                        
                        # Estimate probability of profit
                        distance_from_current = (current_price - strike) / current_price
                        pop = min(0.85, 0.5 + distance_from_current * 2) * 100  # Rough estimate
                        
                        # Score calculation (without Greeks)
                        iv_score = put['impliedVolatility'] if put['impliedVolatility'] > 0 else 0.3
                        risk_score = (1 - moneyness) * iv_score
                        score = (daily_premium * pop / 100) / (risk_score + 0.01)
                        
                        # Filter by minimum requirements
                        if daily_premium >= 0.03 and annualized_roc >= 8:  # Lowered thresholds
                            opp = {
                                'ticker': ticker,
                                'expiration': exp_str,
                                'days_to_exp': days_to_exp,
                                'strike': strike,
                                'premium': premium,
                                'bid': bid,
                                'ask': ask,
                                'moneyness': moneyness,
                                'estimated_delta': estimated_delta,
                                'iv': put['impliedVolatility'],
                                'daily_premium': daily_premium,
                                'annualized_roc': annualized_roc,
                                'pop': pop,
                                'support': final_support,
                                'current_price': current_price,
                                'iv_rank': iv_rank,
                                'score': score,
                                'volume': put['volume'],
                                'open_interest': put['openInterest']
                            }
                            opportunities.append(opp)
                            logger.debug(f"  Added: ${strike:.2f} strike, ${premium:.2f} premium, {annualized_roc:.1f}% ROC")
                            
                except Exception as e:
                    logger.warning(f"Option chain error for {ticker} {exp_str}: {str(e)}")
                    
                time.sleep(0.5)  # Rate limiting
            
            logger.info(f"Found {len(opportunities)} opportunities for {ticker}")
            
        except Exception as e:
            logger.error(f"CSP scan error for {ticker}: {str(e)}")
            
        return opportunities
    
    def rank_opportunities(self, all_opportunities: List[Dict]) -> pd.DataFrame:
        """Rank all opportunities across stocks"""
        if not all_opportunities:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_opportunities)
        df = df.sort_values('score', ascending=False)
        
        # Format for display
        display_columns = [
            'ticker', 'expiration', 'days_to_exp', 'strike', 'premium',
            'bid', 'ask', 'moneyness', 'iv', 'daily_premium', 'annualized_roc', 
            'pop', 'support', 'current_price', 'iv_rank', 'score'
        ]
        
        return df[display_columns]
    
    def run_scan(self, tickers: List[str]) -> pd.DataFrame:
        """Run the full CSP scan"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting CSP scan for {len(tickers)} tickers")
        logger.info(f"Settings: max_weeks={self.max_weeks}, min_iv_rank={self.min_iv_rank}")
        logger.info(f"{'='*60}\n")
        
        # Filter for quality stocks
        quality_stocks = self.get_quality_stocks(tickers)
        logger.info(f"\nFound {len(quality_stocks)} quality stocks: {', '.join(quality_stocks)}")
        
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
        logger.info(f"\n{'='*60}")
        logger.info(f"Found {len(results)} total opportunities")
        logger.info(f"{'='*60}")
        
        return results


def format_results(df: pd.DataFrame, top_n: int = 20) -> str:
    """Format results for display"""
    if df.empty:
        return "No CSP opportunities found matching criteria"
    
    # Format numeric columns
    df = df.head(top_n).copy()
    df['premium'] = df['premium'].apply(lambda x: f"${x:.2f}")
    df['strike'] = df['strike'].apply(lambda x: f"${x:.2f}")
    df['bid/ask'] = df.apply(lambda x: f"${x['bid']:.2f}/${x['ask']:.2f}", axis=1)
    df['moneyness'] = df['moneyness'].apply(lambda x: f"{x:.1%}")
    df['iv'] = df['iv'].apply(lambda x: f"{x:.1%}")
    df['daily_premium'] = df['daily_premium'].apply(lambda x: f"${x:.3f}")
    df['annualized_roc'] = df['annualized_roc'].apply(lambda x: f"{x:.1f}%")
    df['pop'] = df['pop'].apply(lambda x: f"{x:.1f}%")
    df['support'] = df['support'].apply(lambda x: f"${x:.2f}")
    df['current_price'] = df['current_price'].apply(lambda x: f"${x:.2f}")
    df['iv_rank'] = df['iv_rank'].apply(lambda x: f"{x:.1f}%")
    df['score'] = df['score'].apply(lambda x: f"{x:.2f}")
    
    # Select columns for display
    display_cols = ['ticker', 'expiration', 'days_to_exp', 'strike', 'bid/ask', 
                   'daily_premium', 'annualized_roc', 'pop', 'iv_rank', 'score']
    
    return df[display_cols].to_string(index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Scan for optimal Cash-Secured Put opportunities (Fixed Version)'
    )
    parser.add_argument(
        '--stocks',
        type=str,
        help='Comma-separated list of stock tickers (default: S&P 500 leaders)'
    )
    parser.add_argument(
        '--max-weeks',
        type=int,
        default=8,
        help='Maximum weeks to expiration (default: 8)'
    )
    parser.add_argument(
        '--min-iv-rank',
        type=int,
        default=30,
        help='Minimum IV rank percentage (default: 30)'
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
    if not args.debug:
        logging.getLogger().setLevel(logging.INFO)
    
    # Get tickers
    if args.stocks:
        tickers = [t.strip().upper() for t in args.stocks.split(',')]
    else:
        tickers = DEFAULT_STOCKS
    
    # Run scan
    scanner = CSPScanner(max_weeks=args.max_weeks, min_iv_rank=args.min_iv_rank)
    results = scanner.run_scan(tickers)
    
    # Output results
    if not results.empty:
        print("\n" + "="*100)
        print("TOP CASH-SECURED PUT OPPORTUNITIES")
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
        print("  --min-iv-rank 20  (lower IV requirement)")
        print("  --max-weeks 12    (longer expirations)")
        print("  --stocks AAPL,MSFT,GOOGL  (specific high-volume stocks)")
    
    print("\n⚠️  DISCLAIMER: This is for informational purposes only. Not financial advice.")
    print("Always conduct your own research and consider your risk tolerance before trading.\n")


if __name__ == "__main__":
    main()