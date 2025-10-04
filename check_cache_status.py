#!/usr/bin/env python3
"""
Cache Status Monitor

This script displays the current status of the ticker data cache,
including statistics, age, and coverage for each exchange.
"""

import argparse
import datetime
import sys
from tabulate import tabulate

from cache_providers import SQLiteCacheProvider
from modules import constants


def format_age(hours):
    """Format age in hours to human-readable string"""
    if hours < 1:
        return f"{int(hours * 60)} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        days = hours / 24
        return f"{days:.1f} days"


def main():
    parser = argparse.ArgumentParser(description="Check ticker cache status")
    parser.add_argument(
        "--cache-db",
        type=str,
        default=constants.CACHE_DB_FILE,
        help=f"Cache database file (default: {constants.CACHE_DB_FILE})"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Check specific ticker data"
    )
    parser.add_argument(
        "--exchange",
        type=str,
        help="Filter by exchange"
    )
    
    args = parser.parse_args()
    
    # Initialize cache provider
    cache = SQLiteCacheProvider(args.cache_db)
    
    print("=" * 60)
    print("Ticker Data Cache Status")
    print("=" * 60)
    print(f"Cache file: {args.cache_db}")
    print(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check specific ticker if requested
    if args.ticker:
        ticker_data = cache.get_ticker_data(args.ticker.upper())
        if ticker_data:
            print(f"\nData for {args.ticker.upper()}:")
            print(f"  Last updated: {ticker_data.get('last_updated', 'Unknown')}")
            print(f"  Company: {ticker_data.get('company_name', 'Unknown')}")
            print(f"  Sector: {ticker_data.get('sector', 'Unknown')}")
            print(f"  Market Cap: ${ticker_data.get('market_cap', 0)/1e9:.2f}B")
            print(f"  Price: ${ticker_data.get('current_price', 0):.2f}")
            print(f"  P/E Ratio: {ticker_data.get('trailing_pe', 'N/A')}")
            print(f"  IV Rank: {ticker_data.get('iv_rank', 0):.1f}%")
            print(f"  1Y Return: {ticker_data.get('return_1y', 0):.1f}%")
        else:
            print(f"\nNo cached data found for {args.ticker.upper()}")
        print()
    
    # Get ticker cache statistics
    ticker_stats = cache.get_ticker_cache_stats()
    
    if ticker_stats:
        print("Ticker Data Cache Summary:")
        print()
        
        # Prepare table data
        table_data = []
        total_tickers = 0
        
        for exchange, stats in ticker_stats.items():
            if args.exchange and exchange != args.exchange:
                continue
                
            total_tickers += stats['ticker_count']
            
            # Determine status based on age
            newest_age = stats['newest_hours']
            if newest_age < 24:
                status = "✅ Fresh"
            elif newest_age < 48:
                status = "⚠️  Aging"
            else:
                status = "❌ Stale"
            
            table_data.append([
                exchange,
                stats['ticker_count'],
                format_age(stats['newest_hours']),
                format_age(stats['oldest_hours']),
                status
            ])
        
        # Display table
        headers = ["Exchange", "Tickers", "Newest", "Oldest", "Status"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        if not args.exchange:
            print(f"\nTotal tickers cached: {total_tickers}")
    else:
        print("No ticker data found in cache")
    
    # Get quality stock cache statistics
    print("\n" + "=" * 60)
    print("Quality Stocks Cache Summary:")
    print()
    
    quality_stats = cache.get_cache_stats()
    
    if quality_stats:
        table_data = []
        
        for exchange, stats in quality_stats.items():
            if args.exchange and exchange != args.exchange:
                continue
                
            table_data.append([
                exchange,
                stats['total_stocks'],
                stats['passed_filter'],
                f"{stats['passed_filter']/stats['total_stocks']*100:.1f}%" if stats['total_stocks'] > 0 else "0%",
                format_age(stats['age_hours']),
                "✅" if stats['is_valid'] else "❌"
            ])
        
        headers = ["Exchange", "Total", "Passed", "Pass %", "Age", "Valid"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print("No quality stock data found in cache")
    
    # Show cached tickers for specific exchange
    if args.exchange:
        print(f"\n" + "=" * 60)
        print(f"Cached tickers for {args.exchange}:")
        print()
        
        tickers = cache.get_cached_tickers(args.exchange)
        if tickers:
            # Display in columns
            columns = 10
            for i in range(0, len(tickers), columns):
                print("  " + "  ".join(tickers[i:i+columns]))
        else:
            print(f"No tickers cached for {args.exchange}")
    
    print("\n" + "=" * 60)
    print("Cache Recommendations:")
    print()
    
    # Provide recommendations based on cache state
    recommendations = []
    
    if not ticker_stats:
        recommendations.append("• No ticker data cached - run update_tickers.py to populate cache")
    else:
        for exchange, stats in ticker_stats.items():
            if stats['newest_hours'] > 48:
                recommendations.append(f"• {exchange} data is {format_age(stats['newest_hours'])} old - consider updating")
    
    if not quality_stats:
        recommendations.append("• No quality stock data - run csp_scanner.py to populate")
    
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("✅ Cache is up to date!")
    
    print()


if __name__ == "__main__":
    main()