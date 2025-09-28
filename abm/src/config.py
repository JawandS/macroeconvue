"""
Configuration parameters for MacroEconVue ABM Simulation
"""

# Simulation time parameters
TOTAL_DAYS = 5
DAYS_PER_MONTH = 5
DAYS_PER_WEEK = 2
BURN_IN_MONTHS = 12  # Required for year-over-year inflation calculation
BURN_IN_TARGET_INFLATION = 0.02  # π* during burn-in (annualized)

# Central bank parameters
TARGET_INFLATION = 0.02  # π* = 2% annual
NATURAL_RATE = 0.03  # r* = 3% annual natural rate
TAYLOR_RULE_PHI_PI = 1.5  # φ_π coefficient on inflation gap

# Firm parameters (Zero Intelligence)
ALPHA_Q = 0.7  # Weight on lagged demand in production
SIGMA_Q = 0.1  # Standard deviation of production shocks
ALPHA_UP = 0.05  # Price markup when stockout (5%)
BETA_DOWN = 0.03  # Price markdown when excess inventory (3%)

# Service pricing parameters
SIGMA_S = 0.01  # Standard deviation of monthly service price shocks

# Household parameters
NUM_HOUSEHOLDS = 100
TRANSACTION_HISTORY_LENGTH = 30  # H = days of transaction history for LLM

# Goods and services categories
SERVICES = [
    "Housing",
    "Utilities",
    "Communications",
    "Transport",
    "Healthcare",
    "Basic Food"
]

GOODS = [
    "Dining",
    "Entertainment",
    "Apparel",
    "Electronics",
    "Home Goods",
    "Travel"
]

# CPI weights (must sum to 1.0)
CPI_WEIGHTS = {
    # Services
    "Housing": 0.25,
    "Utilities": 0.08,
    "Communications": 0.03,
    "Transport": 0.12,
    "Healthcare": 0.08,
    "Basic Food": 0.14,
    # Goods
    "Dining": 0.10,
    "Entertainment": 0.05,
    "Apparel": 0.06,
    "Electronics": 0.04,
    "Home Goods": 0.03,
    "Travel": 0.02
}

# Validation checks
assert abs(sum(CPI_WEIGHTS.values()) - 1.0) < 1e-10, "CPI weights must sum to 1.0"
assert set(CPI_WEIGHTS.keys()) == set(SERVICES + GOODS), "CPI weights must cover all categories"

# Output file names
OUTPUT_DAILY_TRANSACTIONS = "daily_transactions.csv"
OUTPUT_MONTHLY_SUMMARY = "monthly_summary.csv"
OUTPUT_HOUSEHOLD_BALANCES = "household_balances.csv"

# Simulation validation bounds
MAX_ANNUAL_INFLATION = 0.10  # 10% cap
MIN_ANNUAL_INFLATION = -0.05  # -5% floor (mild deflation allowed)
MAX_HOUSEHOLD_DEBT_RATIO = 5.0  # Max debt as multiple of monthly income
MAX_HOUSEHOLD_SAVINGS_RATIO = 20.0  # Max savings as multiple of monthly income
