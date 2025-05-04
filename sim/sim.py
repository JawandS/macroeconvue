import random
import numpy as np
import pandas as pd

class Household:
    """
    Represents a household agent that can be employed or unemployed,
    earns income, and decides on consumption and savings.
    """
    def __init__(self, id, preference, base_income, goods_categories):
        """
        Args:
            id (int): Unique identifier for the household.
            preference (float): Fraction of disposable income allocated to consumption [0,1].
            base_income (float): Income received when employed.
            goods_categories (dict): Dict mapping category to share of consumption.
        """
        self.id = id
        self.preference = preference
        self.base_income = base_income
        self.goods_categories = goods_categories
        self.employed = False
        self.savings = 0.0

    def step(self, interest_rate):
        """
        Perform one period update: determine employment, earn income,
        allocate between consumption and savings, generate transactions.

        Args:
            interest_rate (float): Current interest rate applied to savings.

        Returns:
            list of dict: Transactions for this household this period.
        """
        # Determine employment status (could use a Markov process)
        self.employed = random.choice([True, False])
        income = self.base_income if self.employed else 0.0
        # Add interest on previous savings
        income += self.savings * interest_rate
        # Split income into consumption and savings
        consumption_budget = income * self.preference
        self.savings = income - consumption_budget
        # Generate spend by category
        transactions = []
        for category, share in self.goods_categories.items():
            spend = consumption_budget * share
            transactions.append({
                'household_id': self.id,
                'category': category,
                'spend': spend
            })
        return transactions

class Firm:
    """
    Represents a firm producing a class of goods.
    """
    def __init__(self, category, production_capacity):
        self.category = category
        self.production_capacity = production_capacity

    def supply(self):
        """
        Returns total goods supplied this period.
        """
        return self.production_capacity

class CentralBank:
    """
    Sets policy interest rate according to a Taylor rule.
    """
    def __init__(self, inflation_target=0.02, phi_pi=1.5, phi_y=0.5, neutral_rate=0.01):
        self.inflation_target = inflation_target
        self.phi_pi = phi_pi
        self.phi_y = phi_y
        self.neutral_rate = neutral_rate
        self.interest_rate = neutral_rate

    def update_rate(self, inflation, output_gap=0.0):
        """
        Update policy rate based on inflation and output gap.
        """
        self.interest_rate = (
            self.neutral_rate
            + self.phi_pi * (inflation - self.inflation_target)
            + self.phi_y * output_gap
        )

class GoodsMarket:
    """
    Aggregates demand and supply, sets prices by matching function.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.prices = {}

    def update_prices(self, demand, supply):
        """
        Update prices based on aggregate demand and supply.
        """
        for cat, d in demand.items():
            s = supply.get(cat, 0)
            self.prices[cat] = (d / s) ** self.alpha if s > 0 else float('inf')
        return self.prices

class Simulation:
    """
    Orchestrates the agent-based model simulation and saves CSV outputs.
    """
    def __init__(
        self,
        num_households,
        household_pref,
        base_income,
        goods_categories,
        firm_configs,
        taylor_params,
        periods
    ):
        self.goods_categories = goods_categories
        self.households = [
            Household(i, household_pref, base_income, goods_categories)
            for i in range(num_households)
        ]
        self.firms = [Firm(cat, cap) for cat, cap in firm_configs.items()]
        self.central_bank = CentralBank(**taylor_params)
        self.market = GoodsMarket(alpha=1.0)
        self.periods = periods
        # Storage for CSV
        self.period_data = []

    def run(self):
        prev_price_index = None
        # Pre-define header fields for CSV
        categories = list(self.goods_categories.keys())
        household_ids = [hh.id for hh in self.households]

        for t in range(self.periods):
            agg_demand = {}
            # initialize household-category spend
            hh_cat_spend = {(hid, cat): 0.0 for hid in household_ids for cat in categories}

            # Each household generates transactions
            for hh in self.households:
                txns = hh.step(self.central_bank.interest_rate)
                for txn in txns:
                    hid = txn['household_id']
                    cat = txn['category']
                    spend = txn['spend']
                    agg_demand[cat] = agg_demand.get(cat, 0) + spend
                    hh_cat_spend[(hid, cat)] += spend

            # Firms supply goods
            agg_supply = {firm.category: firm.supply() for firm in self.firms}
            # Update prices
            prices = self.market.update_prices(agg_demand, agg_supply)
            price_index = np.mean(list(prices.values()))
            # Compute inflation
            if prev_price_index is None:
                inflation = 0.0
            else:
                inflation = (price_index - prev_price_index) / prev_price_index
            prev_price_index = price_index

            # Build CSV row
            row = {'period': t, 'inflation': inflation}
            for hid in household_ids:
                for cat in categories:
                    row[f'hh_{hid}_{cat}'] = hh_cat_spend[(hid, cat)]
            self.period_data.append(row)

            # Update policy rate
            self.central_bank.update_rate(inflation)

        # Export to CSV
        df = pd.DataFrame(self.period_data)
        df.to_csv('simulation_transactions.csv', index=False)
        return self.period_data

if __name__ == "__main__":
    # Example parameters
    goods_categories = {
        'housing': 0.4,
        'healthcare': 0.2,
        'transportation': 0.1,
        'durable': 0.15,
        'nondurable': 0.15
    }
    firm_configs = {cat: 1000 for cat in goods_categories}
    taylor_params = {
        'inflation_target': 0.02,
        'phi_pi': 1.5,
        'phi_y': 0.5,
        'neutral_rate': 0.01
    }
    sim = Simulation(
        num_households=50,
        household_pref=0.6,
        base_income=100.0,
        goods_categories=goods_categories,
        firm_configs=firm_configs,
        taylor_params=taylor_params,
        periods=100
    )
    sim.run()
    print("Simulation complete. Data saved to simulation_transactions.csv")
