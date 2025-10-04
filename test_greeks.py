#!/usr/bin/env python3
import yfinance as yf
import pandas as pd

ticker = 'JPM'
stock = yf.Ticker(ticker)

# Try different methods to get Greeks
exp_str = stock.options[0]
print(f"Testing {ticker} options for {exp_str}")

# Method 1: Standard option_chain
opt_chain = stock.option_chain(exp_str)
puts = opt_chain.puts
print(f"\nMethod 1 - option_chain columns: {list(puts.columns)}")

# Method 2: Check if there's a different API
try:
    # Sometimes yfinance stores additional data in the option object
    print(f"\nOption chain object attributes: {dir(opt_chain)}")
except:
    pass

# Method 3: Try to access raw data
print("\nChecking if raw option data contains Greeks...")
# Get one put option's raw data
if len(puts) > 0:
    sample_symbol = puts.iloc[0]['contractSymbol']
    try:
        opt = yf.Ticker(sample_symbol)
        info = opt.info
        print(f"Option ticker info keys: {list(info.keys())[:20]}...")
    except Exception as e:
        print(f"Failed to get option info: {e}")