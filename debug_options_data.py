#!/usr/bin/env python3
"""
Debug script to show what option data is available from yfinance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def debug_option_chain(ticker='AAPL'):
    """Show all available data from yfinance option chains"""
    
    print(f"\n{'='*60}")
    print(f"Debugging Option Data for {ticker}")
    print(f"{'='*60}\n")
    
    # Get stock info
    stock = yf.Ticker(ticker)
    info = stock.info
    
    print("Stock Info:")
    print(f"  Current Price: ${info.get('currentPrice', 'N/A')}")
    print(f"  Market Cap: ${info.get('marketCap', 0)/1e9:.1f}B")
    print(f"  P/E Ratio: {info.get('trailingPE', 'N/A')}")
    print(f"  52 Week Range: ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}")
    
    # Get current price from history
    hist = stock.history(period='5d')
    current_price = hist['Close'].iloc[-1]
    print(f"  Latest Close: ${current_price:.2f}")
    
    # Get options expirations
    expirations = stock.options
    print(f"\nAvailable Expirations: {len(expirations)}")
    print(f"  Next 5: {expirations[:5]}")
    
    # Get first expiration chain
    if expirations:
        exp_date = expirations[0]
        print(f"\nExamining option chain for {exp_date}:")
        
        opt_chain = stock.option_chain(exp_date)
        puts = opt_chain.puts
        
        print(f"\nPut Option Columns:")
        for col in puts.columns:
            print(f"  - {col}")
        
        print(f"\nTotal Puts: {len(puts)}")
        
        # Show sample put data
        print("\nSample Put Data (first 5 rows):")
        sample = puts.head()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(sample)
        
        # Check for Greeks
        print("\nGreeks Availability:")
        has_greeks = {
            'delta': 'delta' in puts.columns,
            'gamma': 'gamma' in puts.columns,
            'theta': 'theta' in puts.columns,
            'vega': 'vega' in puts.columns,
            'rho': 'rho' in puts.columns
        }
        for greek, available in has_greeks.items():
            print(f"  {greek}: {'✓ Available' if available else '✗ Not Available'}")
        
        # Show IV distribution
        print("\nImplied Volatility Distribution:")
        if 'impliedVolatility' in puts.columns:
            iv_stats = puts['impliedVolatility'].describe()
            print(iv_stats)
        
        # Show strike distribution relative to current price
        print("\nStrike Distribution:")
        strikes = puts['strike'].values
        otm_15_pct = strikes[strikes <= current_price * 0.85]
        otm_20_pct = strikes[strikes <= current_price * 0.80]
        otm_25_pct = strikes[strikes <= current_price * 0.75]
        
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Strikes 15% OTM or more: {len(otm_15_pct)} (max: ${otm_15_pct.max():.2f} if any)")
        print(f"  Strikes 20% OTM or more: {len(otm_20_pct)} (max: ${otm_20_pct.max():.2f} if any)")
        print(f"  Strikes 25% OTM or more: {len(otm_25_pct)} (max: ${otm_25_pct.max():.2f} if any)")
        
        # Show liquidity
        print("\nLiquidity Analysis:")
        liquid_puts = puts[(puts['volume'] > 0) & (puts['openInterest'] > 0)]
        print(f"  Puts with volume > 0: {len(liquid_puts)}")
        print(f"  Puts with OI > 0: {len(puts[puts['openInterest'] > 0])}")
        print(f"  Puts with both: {len(liquid_puts)}")
        
        # Calendar info
        print("\nEarnings Calendar:")
        try:
            calendar = stock.calendar
            if calendar is not None:
                print(f"  Next Earnings: {calendar.get('Earnings Date', 'Unknown')}")
            else:
                print("  No earnings data available")
        except:
            print("  Error fetching calendar")


def check_multiple_stocks():
    """Check option availability for multiple stocks"""
    test_stocks = ['AAPL', 'MSFT', 'JPM', 'SPY', 'QQQ']
    
    print(f"\n{'='*60}")
    print("Checking Multiple Stocks for Option Data")
    print(f"{'='*60}\n")
    
    for ticker in test_stocks:
        try:
            stock = yf.Ticker(ticker)
            exps = stock.options
            
            if exps:
                chain = stock.option_chain(exps[0])
                puts = chain.puts
                
                liquid_puts = puts[(puts['volume'] >= 50) & (puts['openInterest'] >= 50)]
                
                print(f"{ticker}:")
                print(f"  Expirations: {len(exps)}")
                print(f"  First exp puts: {len(puts)}")
                print(f"  Liquid puts: {len(liquid_puts)}")
                print(f"  Has Greeks: {'Yes' if 'delta' in puts.columns else 'No'}")
                
        except Exception as e:
            print(f"{ticker}: Error - {str(e)}")


if __name__ == "__main__":
    # Debug single stock in detail
    debug_option_chain('AAPL')
    
    # Check multiple stocks
    check_multiple_stocks()
    
    print("\n" + "="*60)
    print("Debug complete. Key findings:")
    print("- yfinance does NOT provide Greeks (delta, theta, etc.)")
    print("- Available data: strike, bid, ask, volume, OI, IV")
    print("- Must use alternative methods to estimate option value")
    print("="*60)