#!/usr/bin/env python3
"""
Example script demonstrating how to use the NASDAQ fetcher

This script shows various ways to fetch NASDAQ stocks using the NASDAQFetcher class.
"""

from nasdaq_fetcher import NASDAQFetcher, get_nasdaq_stocks
import pandas as pd
from datetime import datetime


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")


def main():
    print("NASDAQ Stock Fetcher - Example Usage")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize fetcher
    fetcher = NASDAQFetcher(cache_enabled=True)
    
    # Example 1: Get NASDAQ-100 stocks
    print_section("Example 1: Fetching NASDAQ-100 Stocks")
    nasdaq_100 = fetcher.get_nasdaq_100()
    print(f"Found {len(nasdaq_100)} NASDAQ-100 stocks")
    print(f"First 10: {nasdaq_100[:10]}")
    
    # Example 2: Get NASDAQ 500 stocks
    print_section("Example 2: Fetching NASDAQ 500 Stocks")
    nasdaq_500 = fetcher.get_nasdaq_500(include_info=False)
    print(f"Found {len(nasdaq_500)} NASDAQ 500 stocks")
    print(f"First 5 stocks:")
    for i, stock in enumerate(nasdaq_500[:5]):
        print(f"  {i+1}. {stock['symbol']}: {stock['name']}")
    
    # Example 3: Get all NASDAQ stocks with market cap filter
    print_section("Example 3: Fetching Large-Cap NASDAQ Stocks (>$50B)")
    large_caps = fetcher.get_all_nasdaq_stocks(
        min_market_cap=50e9,  # $50 billion
        include_info=True
    )
    print(f"Found {len(large_caps)} large-cap NASDAQ stocks")
    
    if large_caps:
        # Convert to DataFrame for better display
        df = pd.DataFrame(large_caps)
        df['market_cap_billions'] = df['market_cap'] / 1e9
        
        print("\nTop 10 by market cap:")
        top_10 = df.nlargest(10, 'market_cap')[['symbol', 'name', 'market_cap_billions', 'sector']]
        for idx, row in top_10.iterrows():
            print(f"  {row['symbol']:6} - {row['name'][:40]:40} ${row['market_cap_billions']:>8.1f}B  {row['sector']}")
    
    # Example 4: Using convenience function
    print_section("Example 4: Using Convenience Function")
    mid_caps = get_nasdaq_stocks(
        count=200,  # Get top 200 stocks
        min_market_cap=10e9,  # Minimum $10B market cap
        include_info=False  # Just symbols for speed
    )
    print(f"Found {len(mid_caps)} mid-to-large cap stocks")
    print(f"Symbols 50-60: {[s['symbol'] for s in mid_caps[50:60]]}")
    
    # Example 5: Fetch with different sources
    print_section("Example 5: Testing Different Data Sources")
    
    # Try FTP source
    try:
        print("Testing FTP source...")
        stocks_ftp = fetcher._fetch_from_source("ftp")
        print(f"  FTP: Found {len(stocks_ftp)} stocks")
    except Exception as e:
        print(f"  FTP: Failed - {e}")
    
    # Try API source
    try:
        print("Testing API source...")
        stocks_api = fetcher._fetch_from_source("api")
        print(f"  API: Found {len(stocks_api)} stocks")
    except Exception as e:
        print(f"  API: Failed - {e}")
    
    # Example 6: Clear cache
    print_section("Example 6: Cache Management")
    print("Cache location:", fetcher.CACHE_DIR if hasattr(fetcher, 'CACHE_DIR') else "Default cache dir")
    print("Clearing cache...")
    fetcher.clear_cache()
    print("Cache cleared successfully")
    
    # Example 7: Integration with CSP Scanner
    print_section("Example 7: CSP Scanner Integration")
    print("To use with CSP scanner:")
    print("  python csp_scanner.py --exchange NASDAQ_500")
    print("  python csp_scanner.py --exchange NASDAQ_ALL --min-market-cap 5")
    print("  python csp_scanner.py --exchange NASDAQ_500 --min-iv-rank 15")
    
    print("\n" + "="*60)
    print("Example complete!")


if __name__ == "__main__":
    main()