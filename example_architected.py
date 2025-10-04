#!/usr/bin/env python3
"""
Example usage of the architected CSP scanner with different data providers
"""

import logging
from datetime import date

from csp_scanner_architected import CSPScanner
from data_providers import YFinanceProvider

# Optional: Uncomment if you have tastytrade installed and credentials
# from data_providers import TastyTradeProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_yfinance_scan():
    """Example using yfinance data provider"""
    print("\n" + "="*60)
    print("Example 1: YFinance Provider Scan")
    print("="*60)
    
    # Initialize yfinance provider
    provider = YFinanceProvider()
    
    # Create scanner
    scanner = CSPScanner(
        data_provider=provider,
        max_weeks=4,
        min_iv_rank=25.0  # Lower threshold for example
    )
    
    # Scan specific stocks
    test_stocks = ['AAPL', 'MSFT', 'SPY', 'QQQ', 'JPM']
    results = scanner.run_scan(test_stocks)
    
    if not results.empty:
        print(f"\nFound {len(results)} opportunities")
        print("\nTop 3 by score:")
        for idx, row in results.head(3).iterrows():
            print(f"\n{row['ticker']} ${row['strike']:.2f} Put")
            print(f"  Expiration: {row['expiration'][:10]} ({row['days_to_exp']} days)")
            print(f"  Premium: ${row['premium']:.2f}")
            print(f"  Delta: {row['delta']:.3f}")
            print(f"  ROC: {row['annualized_roc']:.1f}% annualized")
            print(f"  Score: {row['score']:.2f}")
    else:
        print("\nNo opportunities found")


def example_tastytrade_scan():
    """Example using TastyTrade data provider (requires credentials)"""
    print("\n" + "="*60)
    print("Example 2: TastyTrade Provider Scan")
    print("="*60)
    
    try:
        # Initialize TastyTrade provider (requires credentials)
        # Replace with your actual credentials
        provider = TastyTradeProvider(
            username="your_username",
            password="your_password"
        )
        
        # Create scanner
        scanner = CSPScanner(
            data_provider=provider,
            max_weeks=6,
            min_iv_rank=30.0
        )
        
        # Scan liquid ETFs
        test_stocks = ['SPY', 'QQQ', 'IWM', 'DIA', 'TLT']
        results = scanner.run_scan(test_stocks)
        
        if not results.empty:
            print(f"\nFound {len(results)} opportunities with native Greeks")
            # TastyTrade provides more accurate Greeks
            print("\nMost accurate delta opportunity:")
            best = results.iloc[0]
            print(f"{best['ticker']} ${best['strike']:.2f} Put")
            print(f"  Native Delta: {best['delta']:.4f}")
            print(f"  Native Theta: {best['theta']:.4f}")
            print(f"  Native Gamma: {best['gamma']:.4f}")
        
    except Exception as e:
        print(f"TastyTrade example failed: {str(e)}")
        print("Make sure to install tastytrade: pip install tastytrade")
        print("And provide valid credentials")


def example_custom_date_scan():
    """Example scanning for a specific date"""
    print("\n" + "="*60)
    print("Example 3: Historical Date Scan")
    print("="*60)
    
    # Scan as if it were a different date
    provider = YFinanceProvider()
    
    # Create scanner for specific date
    scanner = CSPScanner(
        data_provider=provider,
        max_weeks=2,
        min_iv_rank=20.0,
        scan_date=date(2024, 12, 1)  # Historical date
    )
    
    print(f"Scanning as of: {scanner.today}")
    
    # Scan high-volume stocks
    test_stocks = ['AAPL', 'TSLA', 'NVDA']
    results = scanner.run_scan(test_stocks)
    
    print(f"Found {len(results)} opportunities for historical date")


def example_nasdaq_universe():
    """Example scanning NASDAQ-100 universe"""
    print("\n" + "="*60)
    print("Example 4: NASDAQ-100 Universe Scan")
    print("="*60)
    
    provider = YFinanceProvider()
    scanner = CSPScanner(
        data_provider=provider,
        max_weeks=8,
        min_iv_rank=30.0
    )
    
    # Get NASDAQ-100 stocks
    nasdaq_stocks = scanner.get_nasdaq100_stocks()
    print(f"Scanning {len(nasdaq_stocks)} NASDAQ-100 stocks...")
    
    # Scan first 10 for speed
    results = scanner.run_scan(nasdaq_stocks[:10], exchange='NASDAQ')
    
    if not results.empty:
        print(f"\nTop NASDAQ opportunities by IV rank:")
        top_iv = results.nlargest(3, 'iv_rank')
        for _, row in top_iv.iterrows():
            print(f"{row['ticker']}: IV Rank {row['iv_rank']:.1f}%, "
                  f"Score {row['score']:.2f}")


def main():
    """Run all examples"""
    print("CSP Scanner Architecture Examples")
    print("=================================")
    
    # Run examples
    example_yfinance_scan()
    
    # Uncomment if you have TastyTrade credentials
    # example_tastytrade_scan()
    
    example_custom_date_scan()
    example_nasdaq_universe()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("\nKey features demonstrated:")
    print("- Multiple data provider support")
    print("- Historical date analysis")
    print("- Different stock universes")
    print("- Native vs computed Greeks")
    print("="*60)


if __name__ == "__main__":
    main()