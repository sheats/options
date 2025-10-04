# CSP Scanner Refactoring Guide

## Overview

The CSP Scanner has been refactored into a modular architecture with comprehensive test coverage. This guide explains the new structure and how to use it.

## Module Structure

```
options/
├── modules/                    # Business logic modules
│   ├── __init__.py
│   ├── constants.py           # All tunable constants
│   ├── stock_filter.py        # Stock quality filtering
│   ├── support_calculator.py  # Support level calculations
│   ├── option_analyzer.py     # Option chain analysis
│   └── scoring.py             # Scoring and ranking
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── conftest.py           # Shared pytest fixtures
│   ├── test_stock_filter.py
│   ├── test_support_calculator.py
│   ├── test_option_analyzer.py
│   └── test_scoring.py
├── data_providers/           # Data provider interfaces (existing)
├── cache_providers/          # Cache provider interfaces (existing)
├── csp_scanner_refactored.py # Refactored scanner using modules
├── pytest.ini               # Pytest configuration
└── run_tests.py            # Test runner script
```

## Key Changes

### 1. Extracted Modules

- **constants.py**: All tunable parameters in one place
  - Scanner settings (weeks, IV rank, etc.)
  - Quality filters (market cap, P/E, etc.)
  - Option filters (volume, Greeks, etc.)
  - Support calculations parameters

- **stock_filter.py**: `StockFilter` class
  - Quality stock filtering logic
  - Exchange-specific criteria (NASDAQ vs SP500)
  - Cache integration

- **support_calculator.py**: `SupportCalculator` class
  - Multiple support level methods (SMA, local minima, K-means)
  - Near-term and long-term support
  - Configurable buffer application

- **option_analyzer.py**: `OptionAnalyzer` class
  - Option chain filtering
  - Greeks-based analysis
  - Earnings date checking
  - Opportunity creation

- **scoring.py**: `ScoringEngine` class
  - Score calculation algorithm
  - Opportunity ranking
  - Position sizing
  - Results formatting

### 2. Dependency Injection

All modules use dependency injection for data and cache providers:

```python
# Example usage
data_provider = YFinanceProvider()
cache_provider = SQLiteCacheProvider("cache.db", 24)

stock_filter = StockFilter(data_provider, cache_provider)
support_calc = SupportCalculator(data_provider, buffer=0.02)
option_analyzer = OptionAnalyzer(data_provider, max_weeks=8)
scoring_engine = ScoringEngine()
```

### 3. Comprehensive Tests

Each module has a corresponding test file with:
- Unit tests for all public methods
- Edge case testing
- Mock data providers for isolation
- Parametrized tests for different scenarios

## Running Tests

### Install test dependencies:
```bash
pip install -r requirements.txt
```

### Run all tests:
```bash
pytest
```

### Run with coverage:
```bash
pytest --cov=modules --cov-report=html
# or use the helper script:
python run_tests.py --cov
```

### Run specific test file:
```bash
pytest tests/test_stock_filter.py
```

### Run specific test:
```bash
pytest tests/test_stock_filter.py::TestStockFilter::test_get_quality_stocks_sp500
```

## Using the Refactored Scanner

The refactored scanner (`csp_scanner_refactored.py`) works identically to the original:

```bash
# Run with default settings
python csp_scanner_refactored.py

# Run with custom parameters
python csp_scanner_refactored.py --exchange NASDAQ --min-iv-rank 15 --support-buffer 0.03

# Run with specific stocks
python csp_scanner_refactored.py --stocks AAPL,MSFT,GOOGL --debug
```

## Tuning the Scanner

All tunable parameters are now in `modules/constants.py`:

1. **Quality Filters**:
   - `MIN_MARKET_CAP`: Minimum market capitalization
   - `MAX_PE_RATIO`: Maximum P/E ratio
   - `MIN_FORWARD_EPS`: Minimum forward earnings
   - `MIN_ONE_YEAR_RETURN`: Minimum 1-year return

2. **Option Filters**:
   - `MIN_VOLUME`: Minimum option volume
   - `MIN_OPEN_INTEREST`: Minimum open interest
   - `MIN_DELTA` / `MAX_DELTA`: Delta range
   - `MIN_THETA`: Minimum theta decay
   - `MAX_GAMMA`: Maximum gamma risk

3. **Premium Requirements**:
   - `MIN_DAILY_PREMIUM`: Minimum daily premium in dollars
   - `MIN_ANNUALIZED_ROC`: Minimum annualized return

4. **Support Calculations**:
   - `SMA_PERIOD_LONG`: Long-term SMA period (200)
   - `SMA_PERIOD_SHORT`: Near-term SMA period (50)
   - `LOCAL_MINIMA_ORDER`: Window for local minima detection
   - `KMEANS_CLUSTERS`: Number of price clusters

## Backwards Compatibility

The refactored scanner maintains full backwards compatibility:
- Same command-line interface
- Same output format
- Same data and cache provider interfaces
- Original `csp_scanner.py` remains unchanged

## Benefits of Refactoring

1. **Maintainability**: Clear separation of concerns
2. **Testability**: Each module can be tested in isolation
3. **Reusability**: Modules can be used independently
4. **Configurability**: All constants in one place
5. **Extensibility**: Easy to add new features or providers

## Future Enhancements

The modular structure makes it easy to add:
- New scoring algorithms
- Additional technical indicators
- Alternative data providers
- Real-time monitoring capabilities
- Portfolio integration
- Risk management features