#!/usr/bin/env python3
"""
Simple database prompt script for exploring ticker data
"""

import sqlite3
import sys
import os

def main():
    # Default database file
    db_file = "csp_scanner_cache.db"
    
    # Allow specifying database file as argument
    if len(sys.argv) > 1:
        db_file = sys.argv[1]
    
    # Check if database exists
    if not os.path.exists(db_file):
        print(f"Error: Database file '{db_file}' not found")
        sys.exit(1)
    
    print(f"Opening {db_file}...")
    print("Type '.help' for SQLite commands or '.exit' to quit\n")
    
    # Open SQLite prompt
    os.execvp("sqlite3", ["sqlite3", db_file])

if __name__ == "__main__":
    main()