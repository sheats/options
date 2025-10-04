#!/usr/bin/env python3
"""
Simple Ticker Data Updater - Just fetch and store everything

This script fetches comprehensive stock data and stores it in SQLite database.
No filtering during fetch - just get everything and store it.
"""

import argparse
import datetime
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from cache_providers import SQLiteCacheProvider
from nasdaq_fetcher import NASDAQFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_ticker_data(ticker: str) -> Optional[Dict]:
    """Fetch all available data for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info:
            return None
            
        # Get everything we can
        data = {
            "ticker": ticker,
            "last_updated": datetime.datetime.now().isoformat(),
            
            # Company info
            "name": info.get("longName", info.get("shortName", ticker)),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "website": info.get("website", ""),
            "employees": info.get("fullTimeEmployees", 0),
            
            # Market data
            "market_cap": info.get("marketCap", 0),
            "enterprise_value": info.get("enterpriseValue", 0),
            "shares_outstanding": info.get("sharesOutstanding", 0),
            "float_shares": info.get("floatShares", 0),
            
            # Price data
            "current_price": info.get("regularMarketPrice", info.get("previousClose", 0)),
            "previous_close": info.get("previousClose", 0),
            "52_week_high": info.get("fiftyTwoWeekHigh", 0),
            "52_week_low": info.get("fiftyTwoWeekLow", 0),
            "50_day_avg": info.get("fiftyDayAverage", 0),
            "200_day_avg": info.get("twoHundredDayAverage", 0),
            
            # Volume
            "volume": info.get("regularMarketVolume", info.get("volume", 0)),
            "avg_volume": info.get("averageVolume", 0),
            "avg_volume_10d": info.get("averageVolume10days", 0),
            
            # Valuation
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "price_to_book": info.get("priceToBook"),
            
            # Financials
            "revenue": info.get("totalRevenue", 0),
            "revenue_per_share": info.get("revenuePerShare", 0),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "gross_margin": info.get("grossMargins"),
            "ebitda": info.get("ebitda", 0),
            "net_income": info.get("netIncomeToCommon", 0),
            "trailing_eps": info.get("trailingEps", 0),
            "forward_eps": info.get("forwardEps", 0),
            
            # Balance sheet
            "total_cash": info.get("totalCash", 0),
            "total_debt": info.get("totalDebt", 0),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "return_on_assets": info.get("returnOnAssets"),
            "return_on_equity": info.get("returnOnEquity"),
            
            # Growth
            "earnings_growth": info.get("earningsGrowth"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
            
            # Dividends
            "dividend_yield": info.get("dividendYield", 0),
            "dividend_rate": info.get("dividendRate", 0),
            "payout_ratio": info.get("payoutRatio", 0),
            
            # Ownership
            "institutional_ownership": info.get("heldPercentInstitutions", 0),
            "insider_ownership": info.get("heldPercentInsiders", 0),
            
            # Risk
            "beta": info.get("beta"),
            "short_ratio": info.get("shortRatio"),
            
            # Analyst
            "recommendation_mean": info.get("recommendationMean"),
            "recommendation_key": info.get("recommendationKey", ""),
            "target_mean": info.get("targetMeanPrice"),
            "number_of_analysts": info.get("numberOfAnalystOpinions", 0),
        }
        
        # Get historical data for technicals
        try:
            hist = stock.history(period="3mo")
            if len(hist) >= 14:
                # Calculate RSI
                close = hist["Close"]
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                data["rsi_14"] = rsi.iloc[-1] if not rsi.empty else None
                
                # Calculate MACD
                exp1 = close.ewm(span=12, adjust=False).mean()
                exp2 = close.ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                data["macd"] = macd.iloc[-1] if not macd.empty else None
                
                # Calculate returns
                if len(hist) > 0:
                    # 1 year return
                    year_ago_hist = stock.history(period="1y")
                    if len(year_ago_hist) > 0:
                        data["return_1y"] = ((close.iloc[-1] / year_ago_hist["Close"].iloc[0]) - 1) * 100
                    
                    # 30 day volatility
                    returns = close.pct_change().dropna()
                    data["volatility_30d"] = returns.rolling(30).std().iloc[-1] * (252 ** 0.5) * 100 if len(returns) >= 30 else None
                    
        except Exception as e:
            logger.debug(f"Error calculating technicals for {ticker}: {e}")
        
        # Get next earnings date
        try:
            earnings = stock.earnings_dates
            if earnings is not None and not earnings.empty:
                future_earnings = earnings[earnings.index > datetime.datetime.now()]
                if not future_earnings.empty:
                    data["next_earnings_date"] = future_earnings.index[0].isoformat()
        except Exception:
            pass
            
        # Calculate IV rank proxy from volatility
        try:
            hist_1y = stock.history(period="1y")
            if len(hist_1y) > 30:
                returns = hist_1y["Close"].pct_change().dropna()
                
                # Rolling 30-day volatilities over the year
                vol_series = returns.rolling(30).std() * (252 ** 0.5) * 100
                vol_series = vol_series.dropna()
                
                if len(vol_series) > 0:
                    current_vol = vol_series.iloc[-1]
                    vol_min = vol_series.min()
                    vol_max = vol_series.max()
                    
                    if vol_max > vol_min:
                        iv_rank = ((current_vol - vol_min) / (vol_max - vol_min)) * 100
                        data["iv_rank"] = min(100, max(0, iv_rank))
                    else:
                        data["iv_rank"] = 50.0
        except Exception:
            pass
            
        return data
        
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {str(e)}")
        return None


def get_exchange_tickers(exchange: str) -> List[str]:
    """Get all tickers for an exchange"""
    if exchange == "SP500":
        # Use a hardcoded list of major S&P 500 stocks for now
        # You can expand this or find a better data source
        sp500_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "JPM", "JNJ", "V", "UNH", "HD", "PG", "MA", "DIS", "PYPL", "BAC",
            "ADBE", "NFLX", "XOM", "CMCSA", "VZ", "PFE", "TMO", "CSCO", "ABT",
            "NKE", "CVX", "PEP", "ABBV", "ACN", "AVGO", "CRM", "COST", "WMT",
            "MCD", "T", "DHR", "NEE", "BMY", "QCOM", "UPS", "TXN", "RTX", "LIN",
            "AMGN", "HON", "ORCL", "PM", "UNP", "SBUX", "IBM", "GS", "CAT", "GE",
            "MMM", "INTU", "AMD", "BA", "LMT", "MDT", "BLK", "CVS", "AMT", "GILD",
            "MO", "AXP", "C", "SCHW", "WFC", "ZTS", "TMUS", "BKNG", "MDLZ", "ADP",
            "CME", "CI", "SYK", "TGT", "SPGI", "LRCX", "BDX", "VRTX", "CCI", "DE",
            "ISRG", "MU", "EQIX", "PNC", "ANTM", "APD", "CL", "TJX", "MS", "FIS",
            "ETN", "DUK", "BSX", "HUM", "REGN", "SHW", "FCX", "AMAT", "SO", "ITW"
        ]
        logger.info(f"Using {len(sp500_stocks)} major S&P 500 stocks")
        return sp500_stocks
            
    elif exchange in ["NASDAQ", "NASDAQ_500", "NASDAQ_ALL"]:
        fetcher = NASDAQFetcher()
        
        if exchange == "NASDAQ":
            return fetcher.get_nasdaq_100()
        elif exchange == "NASDAQ_500":
            stocks = fetcher.get_nasdaq_500(include_info=False)
            return [s["symbol"] for s in stocks]
        else:  # NASDAQ_ALL
            stocks = fetcher.get_all_nasdaq_stocks(include_info=False)
            return [s["symbol"] for s in stocks]
    
    logger.error(f"Unknown exchange: {exchange}")
    return []


def update_tickers(exchange: str, cache_provider: SQLiteCacheProvider, max_workers: int = 5):
    """Update all tickers for an exchange"""
    
    # Get tickers
    logger.info(f"Fetching {exchange} ticker list...")
    tickers = get_exchange_tickers(exchange)
    
    if not tickers:
        logger.error("No tickers found")
        return
        
    logger.info(f"Updating data for {len(tickers)} tickers...")
    
    # Track progress
    successful = 0
    failed = 0
    
    with tqdm(total=len(tickers), desc=f"Updating {exchange}") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(fetch_ticker_data, ticker): ticker
                for ticker in tickers
            }
            
            # Process results
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                
                try:
                    data = future.result()
                    if data:
                        # Save to database
                        cache_provider.save_ticker_data(ticker, exchange, data)
                        successful += 1
                    else:
                        failed += 1
                        logger.debug(f"No data for {ticker}")
                        
                except Exception as e:
                    failed += 1
                    logger.error(f"Error processing {ticker}: {e}")
                
                pbar.update(1)
                pbar.set_postfix({"Success": successful, "Failed": failed})
                
                # Rate limiting
                time.sleep(0.1)
    
    logger.info(f"\nUpdate complete: {successful} successful, {failed} failed")
    

def generate_hotlist(exchange: str, cache_provider: SQLiteCacheProvider):
    """Generate a simple hotlist from cached data"""
    logger.info(f"\nGenerating hotlist for {exchange}...")
    
    # Get all cached tickers
    cached_tickers = cache_provider.get_cached_tickers(exchange)
    if not cached_tickers:
        logger.warning("No cached data found")
        return
        
    hotlist = []
    
    for ticker in cached_tickers:
        data = cache_provider.get_ticker_data(ticker)
        if not data:
            continue
            
        # Basic filters for hotlist
        market_cap = data.get("market_cap", 0)
        iv_rank = data.get("iv_rank", 0)
        institutional_ownership = data.get("institutional_ownership", 0)
        
        # Skip small caps
        if market_cap < 10e9:  # $10B minimum for hotlist
            continue
            
        # Skip low IV
        if iv_rank < 40:
            continue
            
        # Calculate simple score
        score = iv_rank * (1 + institutional_ownership)
        
        hotlist.append({
            "ticker": ticker,
            "name": data.get("name", ""),
            "score": score,
            "market_cap_b": market_cap / 1e9,
            "iv_rank": iv_rank,
            "institutional_ownership": institutional_ownership * 100,
            "price": data.get("current_price", 0),
        })
    
    # Sort by score
    hotlist.sort(key=lambda x: x["score"], reverse=True)
    
    # Display top 10
    print(f"\n{'='*80}")
    print(f"Top 10 Hotlist for {exchange}")
    print(f"{'='*80}")
    print(f"{'Ticker':<8} {'Name':<30} {'Score':<8} {'MCap($B)':<10} {'IV Rank':<8} {'Inst%':<8} {'Price':<8}")
    print("-" * 80)
    
    for stock in hotlist[:10]:
        print(f"{stock['ticker']:<8} {stock['name'][:30]:<30} {stock['score']:<8.1f} "
              f"{stock['market_cap_b']:<10.1f} {stock['iv_rank']:<8.1f} "
              f"{stock['institutional_ownership']:<8.1f} ${stock['price']:<7.2f}")
    
    # Save to CSV
    if hotlist:
        df = pd.DataFrame(hotlist)
        filename = f"hotlist_{exchange}_{datetime.date.today()}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"\nFull hotlist saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Update ticker data in database")
    parser.add_argument(
        "--exchange",
        type=str,
        default="SP500",
        choices=["SP500", "NASDAQ", "NASDAQ_500", "NASDAQ_ALL"],
        help="Exchange to update"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent workers"
    )
    parser.add_argument(
        "--hotlist",
        action="store_true",
        help="Generate hotlist after update"
    )
    parser.add_argument(
        "--cache-db",
        type=str,
        default="csp_scanner_cache.db",
        help="Cache database file"
    )
    
    args = parser.parse_args()
    
    # Initialize cache provider
    cache_provider = SQLiteCacheProvider(args.cache_db)
    
    # Update tickers
    update_tickers(args.exchange, cache_provider, args.workers)
    
    # Generate hotlist if requested
    if args.hotlist:
        generate_hotlist(args.exchange, cache_provider)


if __name__ == "__main__":
    main()