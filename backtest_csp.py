#!/usr/bin/env python3
"""
Simple CSP Strategy Backtester
Tests historical performance of the CSP selection criteria
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSPBacktester:
    """Backtest CSP strategy performance"""

    def __init__(self, start_date="2023-01-01", end_date="2024-01-01"):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.results = []

    def simulate_csp_trade(self, ticker, trade_date, strike, premium, exp_date):
        """Simulate a single CSP trade outcome"""
        stock = yf.Ticker(ticker)

        # Get price on expiration
        exp_prices = stock.history(start=exp_date, end=exp_date + timedelta(days=5))
        if exp_prices.empty:
            return None

        exp_price = exp_prices["Close"].iloc[0]

        # Calculate P&L
        if exp_price >= strike:
            # Option expires worthless - keep full premium
            profit = premium
            status = "expired"
        else:
            # Assigned - calculate loss
            profit = premium - (strike - exp_price)
            status = "assigned"

        # Calculate returns
        collateral = strike * 100
        trade_return = (profit * 100) / collateral * 100  # Percentage return
        days = (exp_date - trade_date).days
        annualized_return = trade_return * 365 / days if days > 0 else 0

        return {
            "ticker": ticker,
            "trade_date": trade_date,
            "exp_date": exp_date,
            "strike": strike,
            "premium": premium,
            "exp_price": exp_price,
            "profit_per_share": profit,
            "status": status,
            "trade_return": trade_return,
            "annualized_return": annualized_return,
            "days": days,
        }

    def backtest_ticker(self, ticker, num_trades=10):
        """Backtest CSP strategy on a single ticker"""
        logger.info(f"Backtesting {ticker}")

        # Generate random trade dates
        date_range = pd.date_range(self.start_date, self.end_date, freq="D")
        trade_dates = np.random.choice(
            date_range, size=min(num_trades, len(date_range)), replace=False
        )

        for trade_date in sorted(trade_dates):
            try:
                stock = yf.Ticker(ticker)
                # Convert numpy datetime64 to pandas Timestamp
                trade_date_ts = pd.Timestamp(trade_date)
                hist = stock.history(
                    start=trade_date_ts - timedelta(days=30), end=trade_date_ts
                )

                if hist.empty:
                    continue

                current_price = hist["Close"].iloc[-1]

                # Simulate strike selection (20-25% OTM)
                strike = round(current_price * 0.77, 2)

                # Simulate premium (1.5-2.5% of strike for 30 days)
                iv_factor = np.random.uniform(0.015, 0.025)
                premium = round(strike * iv_factor, 2)

                # Expiration 28-35 days out
                days_to_exp = np.random.randint(28, 36)
                exp_date = trade_date_ts + timedelta(days=days_to_exp)

                # Simulate trade
                result = self.simulate_csp_trade(
                    ticker, trade_date_ts, strike, premium, exp_date
                )
                if result:
                    self.results.append(result)

            except Exception as e:
                logger.warning(f"Error simulating trade: {str(e)}")

    def calculate_statistics(self):
        """Calculate backtest performance statistics"""
        if not self.results:
            return None

        df = pd.DataFrame(self.results)

        # Win rate
        win_rate = (df["status"] == "expired").mean() * 100

        # Returns
        avg_return = df["trade_return"].mean()
        avg_annualized = df["annualized_return"].mean()

        # Risk metrics
        max_loss = df["trade_return"].min()
        sharpe = (
            (avg_annualized / df["annualized_return"].std() * np.sqrt(252))
            if df["annualized_return"].std() > 0
            else 0
        )

        # By status
        expired_trades = df[df["status"] == "expired"]
        assigned_trades = df[df["status"] == "assigned"]

        stats = {
            "total_trades": len(df),
            "win_rate": win_rate,
            "avg_trade_return": avg_return,
            "avg_annualized_return": avg_annualized,
            "max_loss_pct": max_loss,
            "sharpe_ratio": sharpe,
            "expired_count": len(expired_trades),
            "assigned_count": len(assigned_trades),
            "avg_days": df["days"].mean(),
        }

        if len(assigned_trades) > 0:
            stats["avg_loss_on_assignment"] = assigned_trades["trade_return"].mean()

        return stats

    def run_backtest(self, tickers, trades_per_ticker=10):
        """Run full backtest on multiple tickers"""
        print(f"\nRunning CSP Strategy Backtest")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print("=" * 60)

        for ticker in tickers:
            self.backtest_ticker(ticker, trades_per_ticker)

        stats = self.calculate_statistics()

        if stats:
            print("\nBACKTEST RESULTS")
            print("-" * 60)
            print(f"Total Trades: {stats['total_trades']}")
            print(f"Win Rate: {stats['win_rate']:.1f}%")
            print(f"Average Trade Return: {stats['avg_trade_return']:.2f}%")
            print(f"Average Annualized Return: {stats['avg_annualized_return']:.1f}%")
            print(f"Max Loss: {stats['max_loss_pct']:.2f}%")
            print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
            print(
                f"Expired/Assigned: {stats['expired_count']}/{stats['assigned_count']}"
            )

            if "avg_loss_on_assignment" in stats:
                print(f"Avg Loss on Assignment: {stats['avg_loss_on_assignment']:.2f}%")

            # Generate trade log
            df = pd.DataFrame(self.results)
            df = df.sort_values("trade_date")

            print("\n\nSAMPLE TRADES (First 10)")
            print("-" * 60)
            sample = df.head(10)[
                ["ticker", "trade_date", "strike", "premium", "status", "trade_return"]
            ]
            sample["trade_date"] = sample["trade_date"].dt.date
            sample["trade_return"] = sample["trade_return"].apply(lambda x: f"{x:.2f}%")
            print(sample.to_string(index=False))

        else:
            print("\nNo trades simulated")

        print("\n" + "=" * 60)
        print("Note: This is a simplified backtest with random parameters.")
        print(
            "Real results will vary based on actual market conditions and selection criteria."
        )


def main():
    # Test stocks
    test_tickers = ["AAPL", "MSFT", "JNJ", "WMT", "JPM"]

    # Run backtest
    backtester = CSPBacktester(start_date="2023-01-01", end_date="2024-01-01")

    backtester.run_backtest(test_tickers, trades_per_ticker=20)

    print("\n⚠️  This backtest uses simplified assumptions.")
    print("Actual CSP scanner uses more sophisticated selection criteria.")


if __name__ == "__main__":
    main()
