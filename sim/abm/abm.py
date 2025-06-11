import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import random

class Household:
    """
    Represents a household agent in the ABM.
    
    Attributes
    ----------
    id : int
        Unique household identifier.
    psi : float
        Marginal propensity to consume.
    alpha : np.ndarray
        Preference weights for categories.
    min_consumption : np.ndarray
        Minimum quantity for each consumption category.
    savings : float
        Current savings (can be negative).
    unemployment_prob : float
        Probability of being unemployed in a given week.
    wage : float
        Weekly wage if employed.
    """
    def __init__(
        self,
        id: int,
        psi: float,
        alpha: np.ndarray,
        min_consumption: np.ndarray,
        unemployment_prob: float,
        wage: float,
        initial_savings: float = 0.0,
    ):
        self.id: int = id
        self.psi: float = psi
        self.alpha: np.ndarray = alpha.copy()
        self.min_consumption: np.ndarray = min_consumption.copy()
        self.savings: float = initial_savings
        self.unemployment_prob: float = unemployment_prob
        self.wage: float = wage
        self.income: float = 0.0

    def realize_income(self) -> float:
        """
        Realizes income for the current week.

        Returns
        -------
        float
            Income received in this week.
        """
        employed = np.random.rand() > self.unemployment_prob
        self.income = self.wage if employed else 0.0
        return self.income

    def compute_baseline_consumption(self, prices: np.ndarray) -> float:
        """
        Computes mandatory baseline consumption expenditure.

        Parameters
        ----------
        prices : np.ndarray
            Current prices per category.

        Returns
        -------
        float
            Total baseline consumption cost.
        """
        return float(np.dot(self.min_consumption, prices))

    def update_wealth_after_baseline(
        self, prev_rate: float, baseline_cost: float
    ) -> float:
        """
        Applies interest and income, subtracts baseline consumption.

        Parameters
        ----------
        prev_rate : float
            Current annual policy rate.
        baseline_cost : float
            Cost of baseline consumption.

        Returns
        -------
        float
            Intermediate wealth after interest, income, and baseline consumption.
        """
        self.savings = self.savings * (1 + prev_rate / 52) + self.income - baseline_cost
        return self.savings

    def compute_discretionary_budget(self, rate: float, lambda_: float) -> float:
        """
        Computes budget available for discretionary consumption.

        Parameters
        ----------
        rate : float
            Annualized policy rate.
        lambda_ : float
            Rate sensitivity parameter.

        Returns
        -------
        float
            Discretionary consumption budget.
        """
        B = max(self.savings, 0.0)
        C = (self.psi * B) / (1 + lambda_ * (rate / 52))
        return max(C, 0.0)

    def allocate_consumption(
        self, C: float, prices: np.ndarray
    ) -> np.ndarray:
        """
        Allocates discretionary consumption across categories by preference and price.

        Parameters
        ----------
        C : float
            Total discretionary consumption budget.
        prices : np.ndarray
            Current prices for each category.

        Returns
        -------
        np.ndarray
            Discretionary spending (in currency units) for each category.
        """
        weights = self.alpha / prices
        norm_weights = weights / np.sum(weights)
        allocation = C * norm_weights
        return allocation

    def finalize_savings(self, discretionary_spending: float) -> float:
        """
        Updates savings after discretionary consumption.

        Parameters
        ----------
        discretionary_spending : float
            Amount spent on discretionary consumption.

        Returns
        -------
        float
            Updated savings.
        """
        self.savings -= discretionary_spending
        return self.savings

class Marketplace:
    """
    Represents a marketplace for a single consumption category.

    Attributes
    ----------
    category_id : int
        Index of the consumption category.
    delta_up : float
        Upward price adjustment rate.
    delta_down : float
        Downward price adjustment rate.
    """
    def __init__(
        self,
        category_id: int,
        initial_price: float,
        delta_up: float,
        delta_down: float,
        initial_inventory: int,
    ):
        self.category_id: int = category_id
        self.price: float = initial_price
        self.delta_up: float = delta_up
        self.delta_down: float = delta_down
        self.inventory: float = float(initial_inventory)
        self.total_sold: float = 0.0
        self.sales: List[Tuple[int, float, float]] = []  # (household_id, quantity, price)

    def replenish_inventory(self, new_quantity: float):
        """
        Replenishes the inventory at the start of the week.

        Parameters
        ----------
        new_quantity : float
            Quantity to add to inventory.
        """
        self.inventory = float(new_quantity)
        self.total_sold = 0.0
        self.sales.clear()

    def offer_goods(
        self,
        household_demands: List[Tuple[int, float]],
        households: Dict[int, Household],
        max_rounds: int = 3,
    ):
        """
        Sells goods to households according to matching/adjustment rules.

        Parameters
        ----------
        household_demands : List[Tuple[int, float]]
            List of (household_id, desired quantity) tuples.
        households : Dict[int, Household]
            Mapping from household IDs to Household objects.
        max_rounds : int
            Maximum rounds without sales before market closes for the week.
        """
        # Shuffle to randomize order
        indices = list(range(len(household_demands)))
        random.shuffle(indices)
        no_sale_rounds = 0
        price = self.price
        sales_this_pass = 0
        n = len(household_demands)
        start_index = 0

        while no_sale_rounds < max_rounds and price > 0 and self.inventory > 0:
            sales_this_pass = 0
            for idx in indices[start_index:]:
                hid, desired_qty = household_demands[idx]
                if desired_qty <= 0:
                    continue
                h = households[hid]
                max_affordable_qty = min(
                    desired_qty, self.inventory, (h.savings // price)
                )
                if max_affordable_qty > 0:
                    # Accept the transaction
                    transaction_qty = max_affordable_qty
                    transaction_cost = transaction_qty * price
                    h.savings -= transaction_cost  # update savings now for realism
                    self.inventory -= transaction_qty
                    self.total_sold += transaction_qty
                    self.sales.append((hid, transaction_qty, price))
                    price *= (1 + self.delta_up)
                    sales_this_pass += 1
                else:
                    # No purchase: decrease price for next offer
                    price = max(0, price * (1 - self.delta_down))
                if self.inventory <= 0 or price == 0:
                    break
            if sales_this_pass == 0:
                no_sale_rounds += 1
            else:
                no_sale_rounds = 0
            # Prepare for next pass: everyone with unsatisfied demand may try again
            indices = [i for i in indices if household_demands[i][1] > 0 and self.inventory > 0]
            if len(indices) == 0:
                break
        self.price = price
        # Remaining stock discarded

    def get_sales_records(self) -> List[Tuple[int, float, float]]:
        """
        Returns sales records for this marketplace for the current week.

        Returns
        -------
        List[Tuple[int, float, float]]
            List of (household_id, quantity, price).
        """
        return self.sales.copy()

class CentralBank:
    """
    Central bank agent handling inflation and the Taylor-rule-based policy rate.

    Attributes
    ----------
    neutral_rate : float
        Long-run (neutral) policy rate.
    taylor_phi : float
        Taylor rule inflation gap coefficient.
    inflation_target : float
        Target (annualized) inflation.
    """
    def __init__(
        self,
        neutral_rate: float,
        taylor_phi: float,
        inflation_target: float,
    ):
        self.neutral_rate: float = neutral_rate
        self.taylor_phi: float = taylor_phi
        self.inflation_target: float = inflation_target
        self.rate: float = neutral_rate
        self.last_inflation: float = inflation_target

    def compute_inflation(self, price_history: List[float]) -> float:
        """
        Computes the four-week moving average inflation rate.

        Parameters
        ----------
        price_history : List[float]
            List of weekly aggregate price indices.

        Returns
        -------
        float
            Four-week average inflation rate (annualized).
        """
        if len(price_history) < 5:
            return self.inflation_target  # Use target until sufficient history
        pi = 0.0
        for k in range(1, 5):
            pt = price_history[-k]
            pt_prev = price_history[-k-1]
            pi += (pt - pt_prev) / pt_prev
        pi /= 4
        self.last_inflation = pi
        return pi

    def update_policy_rate(self, inflation: float):
        """
        Updates the policy rate using the Taylor rule.

        Parameters
        ----------
        inflation : float
            Latest measured inflation (annualized).
        """
        self.rate = self.neutral_rate + self.taylor_phi * (inflation - self.inflation_target)

class ABMSimulation:
    """
    Main class for running the agent-based macroeconomic simulation.
    """
    def __init__(
        self,
        num_households: int = 1000,
        num_categories: int = 4,
        weeks: int = 52,
        unemployment_prob: float = 0.05,
        wage: float = 800,
        min_consumption: List[float] = [20, 30, 10, 50],
        psi: float = 0.8,
        lambda_: float = 10.0,
        alpha: List[float] = [0.15, 0.45, 0.10, 0.30],
        neutral_rate: float = 0.02,
        taylor_phi: float = 1.5,
        inflation_target: float = 0.02,
        delta_up: float = 0.02,
        delta_down: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        Initializes the simulation with all parameters.

        Parameters
        ----------
        num_households : int
            Number of households.
        num_categories : int
            Number of consumption categories (should be 4 for default).
        weeks : int
            Number of weeks to simulate.
        unemployment_prob : float
            Probability of being unemployed in any week.
        wage : float
            Weekly wage if employed.
        min_consumption : List[float]
            Minimum quantity of each category.
        psi : float
            Marginal propensity to consume.
        lambda_ : float
            Interest rate dampening parameter.
        alpha : List[float]
            Preference weights for categories.
        neutral_rate : float
            Long-run (neutral) policy rate.
        taylor_phi : float
            Taylor rule inflation gap coefficient.
        inflation_target : float
            Target inflation.
        delta_up : float
            Upward price adjustment.
        delta_down : float
            Downward price adjustment.
        seed : Optional[int]
            Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.num_households = num_households
        self.num_categories = num_categories
        self.weeks = weeks
        self.households: Dict[int, Household] = {
            i: Household(
                id=i,
                psi=psi,
                alpha=np.array(alpha),
                min_consumption=np.array(min_consumption),
                unemployment_prob=unemployment_prob,
                wage=wage,
            )
            for i in range(num_households)
        }
        self.marketplaces: Dict[int, Marketplace] = {
            i: Marketplace(
                category_id=i,
                initial_price=1.0,
                delta_up=delta_up,
                delta_down=delta_down,
                initial_inventory=1,  # will be set below
            )
            for i in range(num_categories)
        }
        self.central_bank = CentralBank(
            neutral_rate=neutral_rate,
            taylor_phi=taylor_phi,
            inflation_target=inflation_target,
        )
        self.lambda_ = lambda_
        self.price_history: List[float] = [1.0]  # For price index
        self.transaction_records: List[Dict] = []
        self.aggregated_records: List[Dict] = []
        self.category_summary: List[Dict] = []

    def run(self):
        """
        Runs the full simulation over the configured time horizon.
        """
        # Compute expected demand for week 0 for each category
        expected_baseline = np.array([
            self.num_households * m for m in self.households[0].min_consumption
        ])
        # Add average expected discretionary demand (approximate as mean psi*B across all, assume B=0 at t=0)
        expected_demand = expected_baseline  # No initial discretionary demand since B=0 at t=0

        # Set initial inventories for each marketplace
        for i in range(self.num_categories):
            mean = expected_demand[i]
            std = 0.05 * mean
            init_inventory = int(np.random.normal(loc=mean, scale=std))
            self.marketplaces[i].replenish_inventory(max(init_inventory, 0))

        # Set initial prices, policy rate
        for i in range(self.num_categories):
            self.marketplaces[i].price = 1.0
        self.central_bank.rate = self.central_bank.neutral_rate

        for t in range(self.weeks):
            print(f"Week {t+1}/{self.weeks}")

            # 1. Households realize income, compute baseline consumption, update wealth
            prices = np.array([self.marketplaces[i].price for i in range(self.num_categories)])
            for h in self.households.values():
                h.realize_income()
            for h in self.households.values():
                baseline_cost = h.compute_baseline_consumption(prices)
                h.update_wealth_after_baseline(self.central_bank.rate, baseline_cost)

            # 2. Compute discretionary budgets and planned purchases
            planned_discretionary_spending = {}
            planned_discretionary_qty = {}
            for h in self.households.values():
                C = h.compute_discretionary_budget(self.central_bank.rate, self.lambda_)
                spending = h.allocate_consumption(C, prices)
                planned_discretionary_spending[h.id] = spending
                # Calculate desired quantity for each good
                qty = spending / prices
                planned_discretionary_qty[h.id] = qty

            # 3. Aggregate all household category demands for each marketplace
            all_household_qty_by_category = {
                i: [(hid, planned_discretionary_qty[hid][i]) for hid in planned_discretionary_qty]
                for i in range(self.num_categories)
            }

            # 4. Marketplaces replenish inventory
            for i in range(self.num_categories):
                mean = np.sum([h.min_consumption[i] for h in self.households.values()]) \
                    + np.sum([planned_discretionary_qty[h.id][i] for h in self.households.values()])
                
                std = 0.05 * mean
                inventory = int(np.random.normal(loc=mean, scale=std))
                self.marketplaces[i].replenish_inventory(max(inventory, 0))

            # 5. Marketplaces match and transact with households
            for i in range(self.num_categories):
                self.marketplaces[i].offer_goods(
                    all_household_qty_by_category[i],
                    self.households,
                )

            # 6. Gather transaction records for output
            for i in range(self.num_categories):
                category_sales = self.marketplaces[i].get_sales_records()
                for hid, qty, price in category_sales:
                    record = {
                        'household_id': hid,
                        'week': t,
                        'category': i,
                        'quantity': qty,
                        'price': price,
                        'savings_post': self.households[hid].savings,
                    }
                    self.transaction_records.append(record)

            # 7. Update prices for next week (done inside Marketplace via price attribute)

            # 8. Central bank computes inflation, updates rate monthly
            avg_price = np.mean([self.marketplaces[i].price for i in range(self.num_categories)])
            self.price_history.append(avg_price)
            if (t+1) % 4 == 0:
                inflation = self.central_bank.compute_inflation(self.price_history)
                self.central_bank.update_policy_rate(inflation)
            else:
                inflation = self.central_bank.last_inflation

            # 9. Collect aggregates
            agg_record = {
                'week': t,
                'avg_price': avg_price,
                'inflation': inflation,
                'policy_rate': self.central_bank.rate,
            }
            for i in range(self.num_categories):
                agg_record[f'price_{i}'] = self.marketplaces[i].price
                agg_record[f'quantity_{i}'] = self.marketplaces[i].total_sold
            self.aggregated_records.append(agg_record)

            # 10. Category summary for the week
            for i in range(self.num_categories):
                summary = {
                    'week': t,
                    'category': i,
                    'avg_price': self.marketplaces[i].price,
                    'total_quantity': self.marketplaces[i].total_sold,
                }
                self.category_summary.append(summary)

    def write_outputs(self, transaction_file: str, aggregate_file: str, category_file: str):
        """
        Writes all simulation outputs to CSV files.

        Parameters
        ----------
        transaction_file : str
            Path to transaction data CSV.
        aggregate_file : str
            Path to aggregate series CSV.
        category_file : str
            Path to category summary CSV.
        """
        transactions_df = pd.DataFrame(self.transaction_records)
        aggregates_df = pd.DataFrame(self.aggregated_records)
        category_df = pd.DataFrame(self.category_summary)
        transactions_df.to_csv(transaction_file, index=False)
        aggregates_df.to_csv(aggregate_file, index=False)
        category_df.to_csv(category_file, index=False)

if __name__ == "__main__":
    sim = ABMSimulation(num_households=10, seed=42)
    sim.run()
    sim.write_outputs('transactions.csv', 'aggregates.csv', 'category_summary.csv')
