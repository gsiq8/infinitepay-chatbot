#!/usr/bin/env python3
"""
Quick script to run the InfinitePay scraper
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from infinitepay_scraper import main

if __name__ == "__main__":
    main()
