---
name: portfolio-analysis
description: Use this skill when the user asks about portfolio performance, risk metrics, holdings, sector allocation, or stress testing. Covers VaR, Sharpe ratio, beta, drawdown, and concentration analysis.
---

# Portfolio Analysis Workflow

## Step 1: Gather Holdings
Call `get_portfolio_positions` to see current holdings and unrealized P&L.

## Step 2: Assess Performance
Call `get_portfolio_performance` with the relevant period (1d, 1w, 1m, 3m, 6m, 1y, ytd).

## Step 3: Calculate Risk
Call `calculate_portfolio_risk` to get VaR, Sharpe ratio, beta, volatility, and max drawdown.

## Step 4: Check Concentration
Call `get_sector_exposure` to identify any sector concentration above 30%.

## Step 5: Stress Test (if risk is elevated)
If volatility is high or concentration is flagged, call `run_stress_test` with an appropriate scenario:
- `market_crash_2008` — broad market stress
- `rate_hike_300bps` — interest rate sensitivity
- `tech_selloff_30pct` — tech-heavy portfolios
- `oil_shock` — energy exposure
- `pandemic_lockdown` — macro disruption

## Presenting Results
- Lead with the portfolio's total value and overall return
- Highlight any risk metrics outside normal bounds
- Compare Sharpe ratio against the benchmark (>1 is good, >2 is excellent)
- Flag any compliance or concentration issues
