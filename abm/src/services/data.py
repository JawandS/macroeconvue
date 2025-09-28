"""
MacroEconVue ABM Simulation
Manage the different files and outputs
"""

# Dependencies
import os
import pandas as pd
import src.config as config

# Get the path to data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Output file paths
DAILY_TRANSACTIONS_FILE = os.path.join(DATA_DIR, config.OUTPUT_DAILY_TRANSACTIONS)
MONTHLY_SUMMARY_FILE = os.path.join(DATA_DIR, config.OUTPUT_MONTHLY_SUMMARY)
HOUSEHOLD_BALANCES_FILE = os.path.join(DATA_DIR, config.OUTPUT_HOUSEHOLD_BALANCES)

def save_daily_transactions(transactions):
    """Save daily transactions to CSV file"""
    # TODO: implement
    pass