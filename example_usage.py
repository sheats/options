#!/usr/bin/env python3
"""
Example usage of the CSP Scanner
Demonstrates scanning a small set of high-quality stocks
"""

import sys

import pandas as pd

from csp_scanner import CSPScanner

# Example high-quality tickers
EXAMPLE_STOCKS = ["AAPL", "MSFT", "JNJ", "PG", "WMT"]


def main():
    print("CSP Scanner Example - Scanning 5 Quality Stocks")
    print("=" * 60)

    # Initialize scanner with conservative parameters
    scanner = CSPScanner(
        max_weeks=4,  # Focus on near-term options
        min_iv_rank=50.0,  # Only elevated IV environments
    )

    # Run the scan
    print("\nScanning stocks:", ", ".join(EXAMPLE_STOCKS))
    print("This may take a minute...\n")

    results = scanner.run_scan(EXAMPLE_STOCKS)

    if not results.empty:
        # Show top 10 opportunities
        print("\nTOP 10 CSP OPPORTUNITIES")
        print("-" * 60)

        top_10 = results.head(10).copy()

        # Create a simplified view
        simplified = pd.DataFrame(
            {
                "Stock": top_10["ticker"],
                "Exp": top_10["expiration"].str[:10],
                "Strike": top_10["strike"].apply(lambda x: f"${x:.0f}"),
                "Premium": top_10["premium"].apply(lambda x: f"${x:.2f}"),
                "Ann ROC": top_10["annualized_roc"].apply(lambda x: f"{x:.1f}%"),
                "Win %": top_10["pop"].apply(lambda x: f"{x:.0f}%"),
                "Score": top_10["score"].apply(lambda x: f"{x:.1f}"),
            }
        )

        print(simplified.to_string(index=False))

        # Show summary statistics
        print("\n\nSUMMARY STATISTICS")
        print("-" * 60)
        print(f"Total opportunities found: {len(results)}")
        print(f"Average annualized ROC: {results['annualized_roc'].mean():.1f}%")
        print(f"Average win probability: {results['pop'].mean():.1f}%")
        print(f"Best opportunity score: {results['score'].max():.2f}")

        # Show example position sizing
        print("\n\nPOSITION SIZING EXAMPLE ($100K ACCOUNT)")
        print("-" * 60)
        best_opp = results.iloc[0]
        position_size = best_opp["strike"] * 100
        num_contracts = int(100000 * 0.05 / position_size)  # 5% of account

        print(f"Best opportunity: {best_opp['ticker']} ${best_opp['strike']:.2f} Put")
        print(f"Collateral per contract: ${position_size:,.2f}")
        print(f"Max contracts (5% rule): {num_contracts}")
        print(
            f"Total premium collected: ${best_opp['premium'] * num_contracts * 100:,.2f}"
        )

    else:
        print("\nNo opportunities found. This could be due to:")
        print("- Low IV environment (IV rank < 50%)")
        print("- Stocks not meeting quality criteria")
        print("- No strikes below calculated support levels")
        print("- Upcoming earnings for all stocks")

    print("\n" + "=" * 60)
    print("Remember: This is not financial advice!")
    print("Always do your own research before trading.")


if __name__ == "__main__":
    main()
