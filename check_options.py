#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Check JPM options in detail
ticker = 'JPM'
stock = yf.Ticker(ticker)
current_price = stock.history(period='1d')['Close'].iloc[-1]
print(f'{ticker} current price: ${current_price:.2f}')

# Get option expirations
expirations = stock.options
print(f'\nAvailable expirations: {len(expirations)}')
print('First 5 expirations:', expirations[:5])

# Check the first expiration in detail
if expirations:
    exp_str = expirations[0]
    opt_chain = stock.option_chain(exp_str)
    puts = opt_chain.puts
    print(f'\nExpiration {exp_str}:')
    print(f'Total puts: {len(puts)}')
    
    # Show puts with various filters
    print('\nPuts with volume >= 100:')
    high_volume = puts[puts['volume'] >= 100]
    print(f'Count: {len(high_volume)}')
    
    print('\nPuts with delta between -0.30 and -0.15:')
    delta_filtered = puts[(puts['delta'] >= -0.30) & (puts['delta'] <= -0.15)]
    print(f'Count: {len(delta_filtered)}')
    if len(delta_filtered) > 0:
        print('\nSample puts:')
        print(delta_filtered[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'delta', 'impliedVolatility']].head())
    
    # Check support level
    print(f'\nCurrent price: ${current_price:.2f}')
    print(f'95% of 200-day SMA would be around: ${current_price * 0.85:.2f} (estimate)')