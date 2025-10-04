#!/usr/bin/env python3
"""
Simple test runner script for CSP Scanner tests.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --cov              # Run with coverage
    python run_tests.py tests/test_stock_filter.py  # Run specific test file
"""

import subprocess
import sys


def main():
    """Run pytest with appropriate arguments."""
    cmd = ["pytest"]
    
    # Add coverage if requested
    if "--cov" in sys.argv:
        cmd.extend(["--cov=modules", "--cov-report=term-missing", "--cov-report=html"])
        sys.argv.remove("--cov")
    
    # Add any additional arguments
    cmd.extend(sys.argv[1:])
    
    # Run pytest
    result = subprocess.run(cmd)
    
    # Exit with pytest's return code
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()