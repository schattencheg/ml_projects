"""
Simple demo script to run the ML project without interactive input
"""
import sys
import os
# Add src directory to the path so imports work correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import main

if __name__ == "__main__":
    main()