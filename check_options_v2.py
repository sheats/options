#!/usr/bin/env python3
import yfinance as yf
import pandas as pd

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
    print(f'Columns available: {list(puts.columns)}')
    
    # Show a sample put
    if len(puts) > 0:
        print('\nSample put option:')
        print(puts.iloc[0])