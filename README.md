# Quantative Research

A Quantitative research project simulating real tasks performed by the Quantative Research (QR) team at JPMC . the project covers commodity pricing, contract valuation, credit risk modelling, and FICO score quantization.

## Table of contents
- project overview
-  Tasks
    - Task 1 - natural gas price estimation
    - Task 2 - storage contract pricing
    - Task 3 - loan default prediction
    - Task 4 - FICO Score Quantization

## Project Overview

This project shows the workflow of a quantative research tasks across four distinct problem areas:

1. Commodity Pricing - estimate natural gas prices at any date using historical market data and a time series model.
2. Contract Valuation - price a natural gas storage contract by modelling all cash flows (purchases, sales, storage costs, fees).
3. Credit Risk Modelling - predict the probablity of a lloan default using ML and calculate expected loss.
4. FICO Quantatization - find optimal FICO score bucket boundries using MSE minimisation and log-likelihood maximisation via dynamic programming.

## Tasks
## Task 1 - Natural gas price estimatoor

**context:** the trading desk needs a model to estimate the purchase price of natural gas at any date - both historically and upto 1 year into the future.

**Approach:**
- Lead monthly end-of-month natural gas price data (oct 2020 - sep 2024)
- Identified 2 signal in the data: a long term upward trend and a repeating seasonal cycle (higher in winter, lower in summer).
- Built a **Fourier regression model** combining a linear trend with annual and semi-annual harmonic terms:
```
price(t) = β₀ + β₁·t + β₂·sin(2πt/12) + β₃·cos(2πt/12)
                      + β₄·sin(4πt/12) + β₅·cos(4πt/12)
```
- Fitted via Ordinary Least Squares (numpy `lstsq`).
- The model extrapolates gracefully because sinusoidal terms repeat the seasonal pattern indefinitely without diverging.

**Results:**
- R² = **0.936** — model explains 93.6% of price variance.
- RMSE = **$0.19/MMBtu** (average error less than 2%).
- Core function: `estimate_price(date) → float`

**Why not ARIMA or ML?** With only 48 data points, complex models
overfit. The parsimonious 6-parameter Fourier model is both accurate
and extrapolates sensibly

### Task 2 — Storage Contract Pricing

**Context:** A client wants to buy gas in summer, store it, and sell in
winter to profit from seasonal price differences. The desk needs a
model to price the contract fairly.

**Approach:**
Using the price model from Task 1, all cash flows are computed:
```
Contract Value = Revenue from gas sales
               − Cost of gas purchases
               − Injection fees ($/MMBtu)
               − Withdrawal fees ($/MMBtu)
               − Transport costs ($/trip)
               − Storage rent ($/month × months held)
```

Key constraints enforced:
- `injection_rate` caps volume purchasable per date.
- `max_storage` caps total volume held — excess injections are skipped.
- Withdrawal cannot exceed current stored volume.

**Core function:**
```python
price_contract(
    injection_dates, withdrawal_dates,
    injection_rate, withdrawal_rate,
    max_storage, storage_cost_per_month,
    injection_cost_per_unit, withdrawal_cost_per_unit,
    transport_cost_per_trip
) → float  # contract value in $
```

A `price_overrides` dict allows the desk to substitute live market
quotes for any date, bypassing the model estimate.

### Task 3 — Loan Default Prediction & Expected Loss

**Context:** The retail banking arm is experiencing higher-than-expected
loan defaults. The risk team needs a model to estimate the probability
of default (PD) and expected loss for any given loan.

**Dataset:** 10,000 personal loans with features: FICO score, income,
total debt, loan amount, credit lines outstanding, years employed.
Overall default rate: ~17%.

**Approach:**
Three models trained and compared:
| Model | ROC-AUC | Notes |
|---|---|---|
| Logistic Regression | ~0.90 | Interpretable, Basel-compliant |
| Random Forest | ~0.93 | Captures nonlinear interactions |
| Gradient Boosting | ~0.94 | Highest accuracy |

Feature engineering adds economically meaningful ratios:
- `debt_to_income` = total_debt / income
- `loan_to_income` = loan_amount / income
- `debt_service_ratio` = total_debt / loan_amount

**Expected Loss formula (Basel II/III standard):**
```
Expected Loss = PD × LGD × EAD
             = model_output × 0.90 × loan_amt_outstanding
```
where LGD = 1 − Recovery Rate = 1 − 0.10 = **0.90**

**Core function:**
```python
expected_loss(
    credit_lines_outstanding, loan_amt_outstanding,
    total_debt_outstanding, income,
    years_employed, fico_score
) → dict  # {'probability_of_default', 'expected_loss', 'risk_tier'}

**Key finding:** `fico_score` and `debt_to_income` are the two strongest
predictors of default.

### Task 4 — FICO Score Quantization

**Context:** Charlie's ML model requires categorical inputs, but FICO
scores are continuous integers (300–850). She needs an optimal mapping
from FICO scores to a fixed number of rating buckets, where rating 1 =
best credit and rating N = worst.

**Two methods implemented:**

**Method A — MSE Minimisation (1-D k-means):**
Treats bucketing as an approximation problem. Finds centroids that
minimise the total squared error between each score and its bucket's
representative value. Fast and intuitive, but ignores default patterns.

**Method B — Log-Likelihood Maximisation (Dynamic Programming):**
Credit-risk aware. Maximises the likelihood of observing the actual
default pattern given the bucketing:
```
LL(bucket) = k·ln(k/n) + (n-k)·ln(1 − k/n)
```

Solved with DP in O(F²·K) time, where F = unique FICO values, K = buckets.
Places boundaries where default rates change most sharply — maximally
informative for risk modelling.

**Core functions:**
```python
build_fico_rating_map(fico_scores, default_flags, n_buckets, method)
get_rating(fico_score, rating_map_info) → int
get_pd_from_rating(fico_score, rating_map_info, ...) → float
```

**Why DP over brute force?** Brute force is O(F^K) — infeasible for
large datasets. DP reduces this by recognising that the optimal K-bucket
solution is built from optimal (K-1)-bucket sub-solutions.

## Repository Structure
```
quantitative-research/
│
├── data/
│   ├── Nat_Gas.csv                  # Monthly natural gas prices
│   ├── Loan_Data.csv                # 10,000 personal loans
│   └── cx_loan_data.csv             # Mortgage book with FICO scores
│
├── notebooks/
│   ├── task1_gas_price_model.ipynb  # Natural gas price estimation
│   ├── task2_contract_pricing.ipynb # Storage contract valuation
│   ├── task3_loan_default.ipynb     # Loan default & expected loss
│   └── task4_fico_quantization.ipynb# FICO score bucketing
│
├── output/
│   ├── nat_gas_analysis.png         # Price model charts
│   ├── loan_default_analysis.png    # Model comparison charts
│   └── fico_quantization.png        # Bucket boundary charts
│
├── .gitignore
└── README.md
```