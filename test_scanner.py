#!/usr/bin/env python3
"""
Quick test of the fixed CSP scanner
"""

from csp_scanner_fixed import CSPScanner
import logging

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("Testing Fixed CSP Scanner")
    print("="*60)
    
    # Test with high-liquidity stocks
    test_stocks = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'JPM']
    
    # Initialize scanner with relaxed parameters
    scanner = CSPScanner(
        max_weeks=4,      # Near-term options
        min_iv_rank=20.0  # Lower IV requirement for testing
    )
    
    print(f"\nScanning {len(test_stocks)} stocks with relaxed parameters:")
    print(f"  Max weeks: {scanner.max_weeks}")
    print(f"  Min IV rank: {scanner.min_iv_rank}%")
    print(f"  Stocks: {', '.join(test_stocks)}")
    
    # Run scan
    results = scanner.run_scan(test_stocks)
    
    if not results.empty:
        print(f"\n✓ SUCCESS: Found {len(results)} opportunities!")
        print("\nTop 5 opportunities:")
        print("-"*60)
        
        # Show simplified results
        top_5 = results.head(5)
        for idx, row in top_5.iterrows():
            print(f"\n{row['ticker']} ${row['strike']:.2f} Put (exp: {row['expiration'][:10]})")
            print(f"  Premium: ${row['premium']:.2f} (bid: ${row['bid']:.2f}, ask: ${row['ask']:.2f})")
            print(f"  ROC: {row['annualized_roc']:.1f}% annualized")
            print(f"  Days: {row['days_to_exp']}, Daily: ${row['daily_premium']:.3f}")
            print(f"  Score: {row['score']:.2f}")
    else:
        print("\n✗ No opportunities found.")
        print("\nPossible reasons:")
        print("- Low volatility environment")
        print("- Stocks don't meet quality criteria") 
        print("- No liquid options in target strike range")
        print("\nTry running: python debug_options_data.py")


if __name__ == "__main__":
    main()