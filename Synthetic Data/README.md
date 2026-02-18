# Synthetic Data Generation

This section of our project details the creation of a high-fidelity synthetic dataset designed to mimic the financial rhythms of a real-world organization. This data serves as the training foundation for our machine learning models, providing a controlled yet realistic environment to test cash flow forecasting.

## Temporal Foundation

We established the structural skeleton of our ledger by creating a 96-week period (approximately 22 months) starting on April 1, 2024. All data points are anchored to a Friday frequency to align with standard weekly business cycles, such as payroll and bank settlements.

## Revenue Modeling

To simulate a diversified income stream, our model layers three distinct revenue profiles:
* Revenue A (Predictable): Stable weekly receipts with minimal variability.
* Revenue B (Semi-Monthly): Stable receipts that trigger twice a month, following a mid-month and month-end cadence.
* Revenue C (Volatile): Highly erratic weekly receipts modeled using a log-normal distribution to simulate "lumpy" cash inflows.

## Expense Structure

Expenses are divided into compensation and non-compensation categories to reflect realistic corporate outflows:
* Compensation
	* Biweekly Payroll: Includes a fixed base plus variable overtime (OT) modeled with a Gamma distribution.
	* Deductions: Biweekly remittances for payroll taxes (30% base rate) and pension contributions (10% base rate).
	* Monthly Obligations: Statutory payments such as Employer Health Tax (EHT) and benefit premiums, which trigger on the business day of each month based on accumulated payroll. 
* Non-Compensation
	* Operating Expenses (OpEx): A combination of predictable fixed costs and variable costs that fluctuate with business activity.
	* Capital Expenditures (CapEx): Modeled as infrequent "random shock" events representing large-scale investments or emergency repairs.

## Cash Balance Calculation

The final ledger calculates a Weekly Ending Balance by starting with an initial capital of $1,000,000. Our model then applies a cumulative sum of all weekly inflows and outflows to determine the organization's real-time liquidity position throughout the 96-week simulation.