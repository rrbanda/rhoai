---
name: trade-evaluation
description: Use this skill when the user wants to check trade compliance, submit an order, or verify regulatory status before trading. Covers pre-trade checks, order placement, and KYC/AML status.
---

# Trade Evaluation Workflow

## Step 1: Pre-Trade Compliance Check
ALWAYS call `check_compliance` before any trade to validate:
- Position concentration limits
- Restricted securities list
- Cash reserve requirements
- Sector exposure limits

## Step 2: Review Compliance Result
If compliance passes, proceed to Step 3.
If compliance fails, report the specific violation to the user and do NOT submit the order.

## Step 3: Submit the Order (only if compliant)
Call `submit_trade_order` with:
- `portfolio_id` — the target portfolio
- `ticker` — stock to trade
- `action` — buy or sell
- `shares` — number of shares
- `order_type` — market, limit, or stop
- `limit_price` — required for limit/stop orders

## Regulatory Status
Call `get_regulatory_status` to check KYC/AML status for a portfolio's client.
Always verify regulatory status is "approved" before large trades.

## Presenting Results
- Always show the compliance check result first
- If order is placed, show the order confirmation details
- Flag any warnings from the compliance check even if it passed
