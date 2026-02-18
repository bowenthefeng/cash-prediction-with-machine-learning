# Synthetic Data Generation

This section of our project details the creation of a high-fidelity synthetic dataset designed to mimic the financial rhythms of a real-world organization. This data serves as the training foundation for our machine learning models, providing a controlled yet realistic environment to test cash flow forecasting.

## Temporal Foundation

We established the structural skeleton of our ledger by creating a 96-week period (approximately 22 months) starting on April 1, 2024. All data points are anchored to a Friday frequency to align with standard weekly business cycles, such as payroll and bank settlements.

```
weeks = 96
dates = pd.date_range(start='2024-04-01', periods=weeks, freq='W-FRI')
df = pd.DataFrame(index=range(weeks))
df['Week'] = dates
```

## Revenue Modeling

To simulate a diversified income stream, our model layers three distinct revenue profiles:

#### Revenue A (Predictable): Stable weekly receipts with minimal variability.

```
df['Revenue A'] = np.round(np.random.normal(200000, 20000, size=weeks).clip(min=150000), 2)
```

#### Revenue B (Semi-Monthly): Stable receipts that trigger twice a month, following a mid-month and month-end cadence.

```
df['Revenue B'] = np.round(0.0, 2)
semi_monthly = False

for i in range(1, len(df)):

    if i == len(df) - 1:
        semi_monthly = True
    elif df.loc[i-1, 'Week'].day < 15 and df.loc[i, 'Week'].day >= 15:
        semi_monthly = True
    elif df.loc[i-1, 'Week'].month != df.loc[i, 'Week'].month:
        semi_monthly = True

    if semi_monthly == True:
        df.at[i, 'Revenue B'] = np.round(np.random.normal(200000, 20000, size=1).clip(min=150000), 2)
        semi_monthly = False
```

#### Revenue C (Volatile): Highly erratic weekly receipts modeled using a log-normal distribution to simulate "lumpy" cash inflows.

```
df['Revenue C'] = np.round(np.random.lognormal(mean=11, sigma=0.8, size=weeks), 2)
```

## Expense Structure

Expenses are divided into compensation and non-compensation categories to reflect realistic corporate outflows:

#### Compensation

By including different payment schedules, like biweekly payroll, payroll taxes and pension, monthly benefits and other government payments such as employer health tax (EHT), we can accurately see how these various costs hit the cash balance at different times.

##### Biweekly Payroll: Includes a fixed base plus variable overtime (OT) modeled with a Gamma distribution.

```
PayrollBase = 200000
PayrollOT = np.random.gamma(shape=2, scale=10000, size=weeks)
PayrollTaxBaseRate = 0.3
PensionBaseRate = 0.1
variability = np.random.uniform(0.95, 1.05, size=weeks) * (df.index.to_numpy() % 2)
df['Payroll'] = -np.round((1 - PayrollTaxBaseRate) * (PayrollBase + PayrollOT) * variability, 2)
```

##### Deductions: Biweekly remittances for payroll taxes (30% base rate) and pension contributions (10% base rate).

```
variability = np.random.uniform(0.95, 1.05, size=weeks) * (df.index.to_numpy() % 2)
df['Payroll Taxes'] = -np.round(PayrollTaxBaseRate * (PayrollBase + PayrollOT) * variability, 2)
variability = np.random.uniform(0.95, 1.05, size=weeks) * (df.index.to_numpy() % 2)
df['Pension Contribution'] = -np.round(PensionBaseRate * (PayrollBase + PayrollOT) * variability, 2)
```

##### Monthly Obligations: Statutory payments such as Employer Health Tax (EHT) and benefit premiums, which trigger on the business day of each month based on accumulated payroll. 

```
BenefitsBaseRate = 0.08
EHTBaseRate = 0.02
running_payroll_sum = 0
df['Benefits Contribution'] = np.round(-0.0, 2)
df['EHT'] = np.round(-0.0, 2)
running_payroll_sum = df.loc[0, 'Payroll']
monthly = False

for i in range(1, len(df)):
    current_payroll = df.loc[i, 'Payroll']
    running_payroll_sum += current_payroll

    if i == len(df) - 1:
        monthly = True
    elif df.loc[i-1, 'Week'].month != df.loc[i, 'Week'].month:
        monthly = True

    if monthly == True:
        variability = np.random.uniform(0.95, 1.05)
        df.at[i, 'Benefits Contribution'] = np.round(BenefitsBaseRate * variability * running_payroll_sum, 2)
        variability = np.random.uniform(0.95, 1.05)
        df.at[i, 'EHT'] = np.round(EHTBaseRate * variability * running_payroll_sum, 2)
        running_payroll_sum = 0
        monthly = False
```

#### Non-Compensation

For the non-compensation portion of our model, we can focus on categories that represent the "keeping the lights on" costs of a business. These typically fall into Operating Expenses (OpEx) and Capital Expenditures (CapEx).

##### Operating Expenses (OpEx): A combination of predictable fixed costs and variable costs that fluctuate with business activity.

```
df['Fixed OpEx'] = -np.round(np.random.normal(150000, 10000, weeks).clip(min=100000), 2)
df['Variable OpEx'] = -np.round(np.random.lognormal(mean=11, sigma=0.8, size=weeks), 2)
```

##### Capital Expenditures (CapEx): Modeled as infrequent "random shock" events representing large-scale investments or emergency repairs.

```
shock = 1 if np.random.random() < 0.05 else 0
df['CapEx'] = -np.round(shock * np.random.uniform(200000, 800000, size=weeks).clip(min=100000), 2)
```

## Cash Balance Calculation

The final ledger calculates a Weekly Ending Balance by starting with an initial capital of $1,000,000. Our model then applies a cumulative sum of all weekly inflows and outflows to determine the organization's real-time liquidity position throughout the 96-week simulation.

```
beginning_balance = 1000000
df['Ending Balance'] = np.round(beginning_balance + df.iloc[:, 1:].cumsum().sum(axis=1), 2)
```
