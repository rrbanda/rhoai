"""Deterministic seed data for the FinanceInsights Advisory Platform MCP server.

Generates stocks, portfolios, transactions, market indices, watchlists, and
compliance rules with a fixed random seed for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import random

SEED = 42

# ---------------------------------------------------------------------------
# Sector / exchange pools
# ---------------------------------------------------------------------------
SECTORS = [
    "Technology",
    "Healthcare",
    "Financial Services",
    "Consumer Discretionary",
    "Energy",
    "Industrials",
    "Real Estate",
    "Utilities",
]

EXCHANGES = ["NYSE", "NASDAQ"]

# ---------------------------------------------------------------------------
# Realistic ticker → company mappings (by sector)
# ---------------------------------------------------------------------------
STOCK_POOL: dict[str, list[tuple[str, str, str]]] = {
    "Technology": [
        ("AAPL", "Apple Inc.", "NASDAQ"),
        ("MSFT", "Microsoft Corp.", "NASDAQ"),
        ("GOOGL", "Alphabet Inc.", "NASDAQ"),
        ("NVDA", "NVIDIA Corp.", "NASDAQ"),
        ("META", "Meta Platforms Inc.", "NASDAQ"),
        ("AVGO", "Broadcom Inc.", "NASDAQ"),
        ("CRM", "Salesforce Inc.", "NYSE"),
        ("ORCL", "Oracle Corp.", "NYSE"),
    ],
    "Healthcare": [
        ("JNJ", "Johnson & Johnson", "NYSE"),
        ("UNH", "UnitedHealth Group", "NYSE"),
        ("PFE", "Pfizer Inc.", "NYSE"),
        ("ABBV", "AbbVie Inc.", "NYSE"),
        ("MRK", "Merck & Co.", "NYSE"),
        ("TMO", "Thermo Fisher Scientific", "NYSE"),
    ],
    "Financial Services": [
        ("JPM", "JPMorgan Chase & Co.", "NYSE"),
        ("GS", "Goldman Sachs Group", "NYSE"),
        ("BAC", "Bank of America Corp.", "NYSE"),
        ("MS", "Morgan Stanley", "NYSE"),
        ("V", "Visa Inc.", "NYSE"),
        ("MA", "Mastercard Inc.", "NYSE"),
        ("BLK", "BlackRock Inc.", "NYSE"),
        ("C", "Citigroup Inc.", "NYSE"),
    ],
    "Consumer Discretionary": [
        ("AMZN", "Amazon.com Inc.", "NASDAQ"),
        ("TSLA", "Tesla Inc.", "NASDAQ"),
        ("HD", "The Home Depot Inc.", "NYSE"),
        ("NKE", "Nike Inc.", "NYSE"),
        ("SBUX", "Starbucks Corp.", "NASDAQ"),
        ("MCD", "McDonald's Corp.", "NYSE"),
    ],
    "Energy": [
        ("XOM", "Exxon Mobil Corp.", "NYSE"),
        ("CVX", "Chevron Corp.", "NYSE"),
        ("COP", "ConocoPhillips", "NYSE"),
        ("SLB", "Schlumberger Ltd.", "NYSE"),
        ("EOG", "EOG Resources Inc.", "NYSE"),
        ("OXY", "Occidental Petroleum", "NYSE"),
    ],
    "Industrials": [
        ("CAT", "Caterpillar Inc.", "NYSE"),
        ("BA", "Boeing Co.", "NYSE"),
        ("HON", "Honeywell International", "NASDAQ"),
        ("UPS", "United Parcel Service", "NYSE"),
        ("GE", "GE Aerospace", "NYSE"),
        ("LMT", "Lockheed Martin Corp.", "NYSE"),
    ],
    "Real Estate": [
        ("AMT", "American Tower Corp.", "NYSE"),
        ("PLD", "Prologis Inc.", "NYSE"),
        ("CCI", "Crown Castle Inc.", "NYSE"),
        ("SPG", "Simon Property Group", "NYSE"),
        ("EQIX", "Equinix Inc.", "NASDAQ"),
    ],
    "Utilities": [
        ("NEE", "NextEra Energy Inc.", "NYSE"),
        ("DUK", "Duke Energy Corp.", "NYSE"),
        ("SO", "Southern Company", "NYSE"),
        ("D", "Dominion Energy Inc.", "NYSE"),
        ("AEP", "American Electric Power", "NASDAQ"),
    ],
}

PRICE_RANGES: dict[str, tuple[float, float]] = {
    "Technology": (100.0, 900.0),
    "Healthcare": (80.0, 550.0),
    "Financial Services": (40.0, 600.0),
    "Consumer Discretionary": (50.0, 400.0),
    "Energy": (40.0, 130.0),
    "Industrials": (80.0, 500.0),
    "Real Estate": (60.0, 250.0),
    "Utilities": (50.0, 100.0),
}

TAGS_POOL: dict[str, list[str]] = {
    "Technology": ["AI", "cloud", "SaaS", "semiconductor", "big-tech", "growth"],
    "Healthcare": ["pharma", "biotech", "medical-devices", "insurance", "defensive"],
    "Financial Services": ["banking", "fintech", "asset-management", "dividend", "blue-chip"],
    "Consumer Discretionary": ["e-commerce", "retail", "EV", "brand", "consumer"],
    "Energy": ["oil-gas", "upstream", "refining", "dividend", "commodity"],
    "Industrials": ["aerospace", "defense", "logistics", "infrastructure", "cyclical"],
    "Real Estate": ["REIT", "data-center", "commercial", "dividend", "yield"],
    "Utilities": ["regulated", "renewable", "dividend", "defensive", "income"],
}

RISK_TOLERANCES = ["conservative", "moderate", "aggressive"]
ACCOUNT_TYPES = ["individual", "ira", "joint", "trust"]
INVESTMENT_OBJECTIVES = ["growth", "income", "balanced", "preservation"]
TRANSACTION_ACTIONS = ["buy", "sell", "dividend", "fee"]
ORDER_TYPES = ["market", "limit", "stop"]
TRANSACTION_STATUSES = ["completed", "pending", "cancelled"]

CLIENT_NAMES = [
    "Margaret Chen",
    "Robert Alvarez",
    "Sophia Patel",
    "James Worthington III",
    "Elizabeth Nakamura",
]

STRESS_SCENARIOS = [
    "market_crash_2008",
    "rate_hike_300bps",
    "tech_selloff_30pct",
    "oil_shock",
    "pandemic_lockdown",
]


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------
def _uid(prefix: str, idx: int) -> str:
    return f"{prefix}-{idx:04d}"


def _iso_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _random_date(rng: random.Random, start: datetime, end: datetime) -> datetime:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, max(delta, 1)))


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def _generate_stocks(rng: random.Random) -> list[dict[str, Any]]:
    stocks: list[dict[str, Any]] = []
    idx = 1
    for sector, tickers in STOCK_POOL.items():
        lo, hi = PRICE_RANGES[sector]
        tags_pool = TAGS_POOL[sector]
        for ticker, name, exchange in tickers:
            price = round(rng.uniform(lo, hi), 2)
            prev_close = round(price * rng.uniform(0.97, 1.03), 2)
            day_high = round(price * rng.uniform(1.001, 1.04), 2)
            day_low = round(price * rng.uniform(0.96, 0.999), 2)
            volume = rng.randint(500_000, 80_000_000)
            avg_volume = rng.randint(1_000_000, 60_000_000)
            beta = round(rng.uniform(0.4, 2.2), 2)
            pe_ratio = round(rng.uniform(8.0, 60.0), 2)
            eps = round(price / pe_ratio, 2)
            dividend_yield = round(rng.uniform(0.0, 4.5), 2)
            year_high = round(price * rng.uniform(1.05, 1.40), 2)
            year_low = round(price * rng.uniform(0.55, 0.90), 2)

            if sector in ("Technology", "Consumer Discretionary"):
                market_cap = rng.randint(50, 3000) * 1_000_000_000
            elif sector in ("Financial Services", "Healthcare"):
                market_cap = rng.randint(30, 600) * 1_000_000_000
            else:
                market_cap = rng.randint(10, 200) * 1_000_000_000

            stocks.append(
                {
                    "ticker": ticker,
                    "name": name,
                    "sector": sector,
                    "exchange": exchange,
                    "current_price": price,
                    "prev_close": prev_close,
                    "day_high": day_high,
                    "day_low": day_low,
                    "volume": volume,
                    "avg_volume": avg_volume,
                    "market_cap": market_cap,
                    "pe_ratio": pe_ratio,
                    "eps": eps,
                    "dividend_yield": dividend_yield,
                    "beta": beta,
                    "year_high": year_high,
                    "year_low": year_low,
                    "tags": rng.sample(
                        tags_pool, k=min(rng.randint(2, 4), len(tags_pool))
                    ),
                }
            )
            idx += 1
    return stocks


def _generate_portfolios(
    rng: random.Random, stocks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    portfolios: list[dict[str, Any]] = []
    for i, client_name in enumerate(CLIENT_NAMES, 1):
        risk = rng.choice(RISK_TOLERANCES)
        acct_type = rng.choice(ACCOUNT_TYPES)
        objective = rng.choice(INVESTMENT_OBJECTIVES)

        num_positions = rng.randint(5, 15)
        selected = rng.sample(stocks, k=min(num_positions, len(stocks)))
        positions = []
        for s in selected:
            shares = rng.randint(10, 500)
            avg_cost = round(s["current_price"] * rng.uniform(0.70, 1.15), 2)
            current_value = round(shares * s["current_price"], 2)
            positions.append(
                {
                    "ticker": s["ticker"],
                    "shares": shares,
                    "avg_cost": avg_cost,
                    "current_value": current_value,
                }
            )

        cash = round(rng.uniform(5_000, 250_000), 2)
        margin = round(rng.uniform(0, 100_000), 2) if acct_type == "individual" else 0.0
        created = _random_date(rng, datetime(2020, 1, 1), datetime(2024, 12, 31))

        portfolios.append(
            {
                "portfolio_id": _uid("PORT", i),
                "client_id": _uid("CLI", i),
                "client_name": client_name,
                "risk_tolerance": risk,
                "account_type": acct_type,
                "positions": positions,
                "cash_balance": cash,
                "margin_available": margin,
                "created_at": _iso_date(created),
                "investment_objective": objective,
            }
        )
    return portfolios


def _generate_transactions(
    rng: random.Random,
    portfolios: list[dict[str, Any]],
    stocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    transactions: list[dict[str, Any]] = []
    for i in range(1, 201):
        portfolio = rng.choice(portfolios)
        stock = rng.choice(stocks)
        action = rng.choice(TRANSACTION_ACTIONS)

        if action in ("buy", "sell"):
            shares = rng.randint(1, 200)
            price = round(stock["current_price"] * rng.uniform(0.90, 1.10), 2)
            total = round(shares * price, 2)
            order_type = rng.choice(ORDER_TYPES)
        elif action == "dividend":
            shares = rng.randint(10, 500)
            price = round(rng.uniform(0.20, 2.50), 2)
            total = round(shares * price, 2)
            order_type = "market"
        else:
            shares = 0
            price = 0.0
            total = round(rng.uniform(5.0, 50.0), 2)
            order_type = "market"

        tx_date = _random_date(rng, datetime(2025, 1, 1), datetime(2025, 12, 31))

        transactions.append(
            {
                "transaction_id": _uid("TXN", i),
                "portfolio_id": portfolio["portfolio_id"],
                "ticker": stock["ticker"],
                "action": action,
                "shares": shares,
                "price": price,
                "total_amount": total,
                "date": _iso_date(tx_date),
                "status": rng.choice(TRANSACTION_STATUSES),
                "order_type": order_type,
            }
        )
    return transactions


def _generate_market_indices(rng: random.Random) -> list[dict[str, Any]]:
    indices = [
        ("SPX", "S&P 500", 4800.0, 5600.0),
        ("DJI", "Dow Jones Industrial Average", 38000.0, 44000.0),
        ("IXIC", "NASDAQ Composite", 15000.0, 18500.0),
        ("RUT", "Russell 2000", 1900.0, 2300.0),
    ]
    result: list[dict[str, Any]] = []
    for symbol, name, lo, hi in indices:
        value = round(rng.uniform(lo, hi), 2)
        change = round(rng.uniform(-120, 120), 2)
        result.append(
            {
                "symbol": symbol,
                "name": name,
                "value": value,
                "change": change,
                "change_pct": round(change / value * 100, 2),
            }
        )
    return result


def _generate_watchlists(
    rng: random.Random, stocks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    tech_stocks = [s["ticker"] for s in stocks if s["sector"] == "Technology"]
    value_candidates = [s["ticker"] for s in stocks if s["pe_ratio"] < 25]
    dividend_candidates = [s["ticker"] for s in stocks if s["dividend_yield"] > 2.0]

    return [
        {
            "watchlist_id": _uid("WL", 1),
            "name": "Tech Growth",
            "description": "High-growth technology stocks with strong momentum",
            "tickers": tech_stocks[:8],
        },
        {
            "watchlist_id": _uid("WL", 2),
            "name": "Value Picks",
            "description": "Undervalued stocks with low P/E ratios",
            "tickers": rng.sample(
                value_candidates, k=min(8, len(value_candidates))
            ),
        },
        {
            "watchlist_id": _uid("WL", 3),
            "name": "Dividend Kings",
            "description": "High-yield dividend payers for income portfolios",
            "tickers": rng.sample(
                dividend_candidates, k=min(8, len(dividend_candidates))
            ),
        },
    ]


def _generate_compliance_rules() -> dict[str, Any]:
    return {
        "restricted_tickers": ["GME", "AMC", "BBBY", "DWAC", "PHUN"],
        "max_single_position_pct": 25,
        "max_sector_concentration_pct": 40,
        "min_cash_reserve_pct": 5,
        "prohibited_order_types": {
            "conservative": ["stop"],
        },
    }


# ---------------------------------------------------------------------------
# DataStore
# ---------------------------------------------------------------------------
@dataclass
class DataStore:
    """Container for all seed data."""

    stocks: list[dict[str, Any]] = field(default_factory=list)
    portfolios: list[dict[str, Any]] = field(default_factory=list)
    transactions: list[dict[str, Any]] = field(default_factory=list)
    market_indices: list[dict[str, Any]] = field(default_factory=list)
    watchlists: list[dict[str, Any]] = field(default_factory=list)
    compliance_rules: dict[str, Any] = field(default_factory=dict)


def create_data_store(seed: int = SEED) -> DataStore:
    """Create a fully populated DataStore with deterministic data."""
    rng = random.Random(seed)
    stocks = _generate_stocks(rng)
    portfolios = _generate_portfolios(rng, stocks)
    transactions = _generate_transactions(rng, portfolios, stocks)
    market_indices = _generate_market_indices(rng)
    watchlists = _generate_watchlists(rng, stocks)
    compliance_rules = _generate_compliance_rules()
    return DataStore(
        stocks=stocks,
        portfolios=portfolios,
        transactions=transactions,
        market_indices=market_indices,
        watchlists=watchlists,
        compliance_rules=compliance_rules,
    )


if __name__ == "__main__":
    store = create_data_store()
    print(f"Stocks:          {len(store.stocks)}")
    print(f"Portfolios:      {len(store.portfolios)}")
    print(f"Transactions:    {len(store.transactions)}")
    print(f"Market indices:  {len(store.market_indices)}")
    print(f"Watchlists:      {len(store.watchlists)}")
    print(f"Compliance keys: {list(store.compliance_rules.keys())}")
    print()
    print("Sample stock:", store.stocks[0])
    print("Sectors:", sorted({s["sector"] for s in store.stocks}))
