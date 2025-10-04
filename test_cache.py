#!/usr/bin/env python3
"""Test script to verify SQLite caching functionality"""

import sqlite3
import datetime
import os
import sys

CACHE_DB_FILE = "csp_scanner_cache.db"

def test_cache_db():
    """Test the cache database functionality"""
    print(f"Testing cache database: {CACHE_DB_FILE}")
    
    if not os.path.exists(CACHE_DB_FILE):
        print("Cache database does not exist yet. Run the scanner to create it.")
        return
    
    try:
        conn = sqlite3.connect(CACHE_DB_FILE)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"\nTables in database: {[t[0] for t in tables]}")
        
        # Check quality_stocks table structure
        cursor.execute("PRAGMA table_info(quality_stocks);")
        columns = cursor.fetchall()
        print("\nquality_stocks table structure:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        # Check cached data
        cursor.execute("""
            SELECT exchange, COUNT(*) as count, 
                   MIN(last_updated) as oldest, 
                   MAX(last_updated) as newest
            FROM quality_stocks
            GROUP BY exchange
        """)
        
        print("\nCached data summary:")
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                exchange, count, oldest, newest = row
                print(f"  {exchange}: {count} stocks")
                print(f"    Oldest: {oldest}")
                print(f"    Newest: {newest}")
        else:
            print("  No cached data found")
        
        # Check specific stocks
        cursor.execute("""
            SELECT ticker, exchange, market_cap/1e9 as mcap_b, pe_ratio, 
                   one_yr_return, passed_filter, last_updated
            FROM quality_stocks
            WHERE passed_filter = 1
            ORDER BY last_updated DESC
            LIMIT 10
        """)
        
        print("\nRecent quality stocks (passed filter):")
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                ticker, exchange, mcap, pe, ret, passed, updated = row
                print(f"  {ticker} ({exchange}): MCap=${mcap:.1f}B, PE={pe:.1f}, "
                      f"1yr={ret:.1f}%, Updated: {updated}")
        else:
            print("  No stocks that passed filters")
        
        conn.close()
        
    except Exception as e:
        print(f"Error testing cache: {str(e)}")

def clear_old_cache():
    """Clear cache entries older than 24 hours"""
    if not os.path.exists(CACHE_DB_FILE):
        print("No cache database to clear")
        return
        
    try:
        conn = sqlite3.connect(CACHE_DB_FILE)
        cursor = conn.cursor()
        
        cutoff = datetime.datetime.now() - datetime.timedelta(hours=24)
        cursor.execute("""
            DELETE FROM quality_stocks 
            WHERE last_updated < ?
        """, (cutoff,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"Deleted {deleted} old cache entries")
        
    except Exception as e:
        print(f"Error clearing old cache: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--clear-old":
        clear_old_cache()
    else:
        test_cache_db()