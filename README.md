# Cash Prediction with Machine Learning

## Introduction

This project focuses on developing a machine learning model to forecast corporate cash flow. The goal is to provide a robust tool for predicting liquidity and helping organizations manage their financial health more effectively.

## Synthetic Data Generation

The first step of our project was the creation of a comprehensive synthetic dataset designed to mimic the weekly cash flow of a real-world organization. Instead of using simple random numbers, we built a data engine that reflects the actual timing and complexity of corporate finance.

Key features of this dataset include:
* Diverse Revenue Streams: Multiple inflows with varying levels of predictability.
* Complex Payroll Cycles: Biweekly payroll inclusive of overtime variability.
* Realistic Tax Obligations: Scheduled federal and provincial tax remittances that adjust based on payroll activity.
* Operating & Capital Expenses: Both recurring monthly costs and irregular, large-scale investments.

By starting with this high-fidelity synthetic data, we can train and test our machine learning models in a controlled environment that still captures the nuances and "lumpiness" of a professional ledger.

## Modeling