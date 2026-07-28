---
name: market-research
description: Use this skill when the user asks about stock prices, market conditions, historical trends, or wants to screen stocks by criteria like sector, valuation, or dividend yield.
---

# Market Research Workflow

## For a Single Stock Quote
Call `get_stock_quote` with the ticker symbol.

## For Market Overview
Call `get_market_summary` (no parameters) to get S&P 500, DJIA, NASDAQ, and Russell 2000 levels.

## For Historical Trends
Call `get_historical_prices` with ticker, date_from, date_to, and interval (daily/weekly/monthly).

## For Stock Screening
Call `screen_stocks` with filters:
- `sector` — Technology, Healthcare, Financial Services, etc.
- `min_market_cap` — minimum market cap in dollars
- `max_pe_ratio` — maximum P/E ratio
- `min_dividend_yield` — minimum dividend yield percentage
- `sort_by` — market_cap, pe_ratio, dividend_yield, volume, price

## For Deep Stock Analysis
Call `analyze_stock` with the ticker to get valuation, technicals, support/resistance, and sector comparison.

## Presenting Results
- Compare individual stocks against the broader market indices
- Note daily change direction and volume
- Provide context (e.g., "up 2.3% vs S&P 500 up 0.8%")
