# Daily Ticker Update Scheduler Setup

This document provides instructions for setting up daily automatic updates of ticker data using cron (macOS/Linux) or Task Scheduler (Windows).

## Overview

The `update_tickers.py` script pre-fetches comprehensive stock data for all configured exchanges and stores it in a cache. This significantly speeds up the CSP scanner by reducing API calls during scanning.

## macOS/Linux Setup (using cron)

1. **Open Terminal and edit your crontab:**
   ```bash
   crontab -e
   ```

2. **Add the following cron jobs to run at different times to avoid API overload:**

   ```bash
   # Update S&P 500 stocks daily at 6:00 AM
   0 6 * * * cd /Users/sheats/options && /usr/bin/python3 update_tickers.py --exchange SP500 >> /Users/sheats/options/logs/update_sp500.log 2>&1

   # Update NASDAQ-100 stocks daily at 7:00 AM
   0 7 * * * cd /Users/sheats/options && /usr/bin/python3 update_tickers.py --exchange NASDAQ >> /Users/sheats/options/logs/update_nasdaq100.log 2>&1

   # Update NASDAQ-500 stocks daily at 8:00 AM (weekdays only)
   0 8 * * 1-5 cd /Users/sheats/options && /usr/bin/python3 update_tickers.py --exchange NASDAQ_500 >> /Users/sheats/options/logs/update_nasdaq500.log 2>&1

   # Update all NASDAQ stocks with >$1B market cap on Saturdays at 6:00 AM
   0 6 * * 6 cd /Users/sheats/options && /usr/bin/python3 update_tickers.py --exchange NASDAQ_ALL --min-market-cap 1 >> /Users/sheats/options/logs/update_nasdaq_all.log 2>&1
   ```

3. **Create logs directory:**
   ```bash
   mkdir -p /Users/sheats/options/logs
   ```

4. **Verify cron jobs:**
   ```bash
   crontab -l
   ```

## Windows Setup (using Task Scheduler)

1. **Open Task Scheduler:**
   - Press Win+R, type `taskschd.msc`, press Enter

2. **Create a new task for each exchange:**

   ### S&P 500 Daily Update:
   - Click "Create Task"
   - Name: "Update S&P 500 Ticker Data"
   - Triggers: Daily at 6:00 AM
   - Actions: 
     - Program: `C:\Python\python.exe` (adjust path)
     - Arguments: `C:\Users\sheats\options\update_tickers.py --exchange SP500`
     - Start in: `C:\Users\sheats\options`

   ### NASDAQ-100 Daily Update:
   - Create similar task for NASDAQ at 7:00 AM

   ### NASDAQ-500 Weekday Update:
   - Create similar task for NASDAQ_500 at 8:00 AM, Mon-Fri only

   ### NASDAQ All Weekly Update:
   - Create similar task for NASDAQ_ALL on Saturdays at 6:00 AM

## Manual Update Commands

You can also run updates manually:

```bash
# Update specific exchange
python3 update_tickers.py --exchange SP500

# Update with custom market cap filter (in billions)
python3 update_tickers.py --exchange NASDAQ_ALL --min-market-cap 10

# Clear cache and start fresh
python3 update_tickers.py --exchange SP500 --clear-cache

# Use more workers for faster updates (default is 5)
python3 update_tickers.py --exchange NASDAQ_500 --workers 10

# Disable resume (start from beginning even if partially cached)
python3 update_tickers.py --exchange SP500 --no-resume
```

## Monitoring

1. **Check update logs:**
   ```bash
   # View latest S&P 500 update log
   tail -f /Users/sheats/options/logs/update_sp500.log
   
   # Check for errors across all logs
   grep ERROR /Users/sheats/options/logs/update_*.log
   ```

2. **View cache statistics:**
   ```bash
   # Create a simple script to check cache stats
   cat > check_cache_stats.py << 'EOF'
   from cache_providers import SQLiteCacheProvider
   from modules import constants
   
   cache = SQLiteCacheProvider(constants.CACHE_DB_FILE)
   stats = cache.get_ticker_cache_stats()
   
   if stats:
       print("Ticker Cache Statistics:")
       for exchange, data in stats.items():
           print(f"\n{exchange}:")
           print(f"  Tickers cached: {data['ticker_count']}")
           print(f"  Oldest data: {data['oldest_hours']:.1f} hours ago")
           print(f"  Newest data: {data['newest_hours']:.1f} hours ago")
   else:
       print("No cached ticker data found")
   EOF
   
   python3 check_cache_stats.py
   ```

## Best Practices

1. **Stagger update times** to avoid overwhelming the API
2. **Run updates during off-market hours** (early morning is ideal)
3. **Monitor logs regularly** for errors
4. **Clear cache monthly** to ensure data freshness
5. **Adjust worker count** based on your system and API limits

## Troubleshooting

### Common Issues:

1. **"Permission denied" errors:**
   ```bash
   chmod +x /Users/sheats/options/update_tickers.py
   ```

2. **Python not found:**
   - Use full path to python: `/usr/bin/python3` or `which python3`

3. **Rate limiting errors:**
   - Reduce workers: `--workers 3`
   - Increase rate limit delay: `--rate-limit 0.5`

4. **Memory issues with NASDAQ_ALL:**
   - Use market cap filter: `--min-market-cap 5`
   - Process in smaller batches

### Recovery from Interruptions:

The script automatically resumes from where it left off:
```bash
# If interrupted, just run again - it will skip already cached tickers
python3 update_tickers.py --exchange NASDAQ_500
```

## Integration with CSP Scanner

Once ticker data is cached, the CSP scanner will automatically use it:

```bash
# Scanner will use cached data when available
python3 csp_scanner.py --exchange SP500

# Force fresh data fetch (bypass cache)
python3 csp_scanner.py --exchange SP500 --no-cache
```

## Performance Benefits

- **Before caching:** ~5-10 seconds per ticker (API calls for each metric)
- **After caching:** <0.1 seconds per ticker (database lookups)
- **Typical speedup:** 50-100x faster scanning

## Maintenance

### Weekly maintenance tasks:
1. Check log file sizes and rotate if needed
2. Verify all scheduled jobs are running
3. Review error logs

### Monthly maintenance tasks:
1. Clear old cache data: `python3 update_tickers.py --clear-cache`
2. Update all exchanges with fresh data
3. Check disk space usage of cache database