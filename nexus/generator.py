"""Reproducible synthetic financial data generator."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

MERCHANT_CATEGORIES = ["grocery", "electronics", "travel", "food", "utilities", "retail", "gaming"]
CHANNELS = ["pix", "card", "ted", " boleto"]
CITIES = ["São Paulo", "Rio de Janeiro", "Brasília", "Belo Horizonte", "Curitiba", "Recife", "Salvador"]

def generate_data(rows: int = 100_000, seed: int = 42, output_dir: str = "data/bronze") -> dict[str, pd.DataFrame]:
    if rows < 100: raise ValueError("rows must be >= 100")
    rng = np.random.default_rng(seed)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    n_customers = max(1000, rows // 20)
    customers = pd.DataFrame({
        "customer_id": [f"C{i:08d}" for i in range(n_customers)],
        "age": rng.integers(18, 86, n_customers),
        "segment": rng.choice(["mass", "affluent", "premium", "business"], n_customers, p=[.52,.28,.16,.04]),
        "state": rng.choice(["SP","RJ","MG","PR","BA","PE","DF"], n_customers),
        "account_open_date": pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 3000, n_customers), unit="D"),
    })
    accounts = pd.DataFrame({
        "account_id": [f"A{i:09d}" for i in range(n_customers)],
        "customer_id": customers.customer_id,
        "account_type": rng.choice(["checking","savings","investment"], n_customers, p=[.75,.15,.10]),
        "risk_tier": rng.choice(["low","medium","high"], n_customers, p=[.70,.25,.05]),
    })
    cust_idx = rng.integers(0,n_customers,rows)
    timestamps = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365*24*60, rows), unit="m")
    amount = np.clip(rng.lognormal(mean=3.7, sigma=1.05, size=rows), 1, 25000).round(2)
    channel = rng.choice(["pix","card","ted","boleto"], rows, p=[.48,.30,.12,.10])
    merchant_cat = rng.choice(MERCHANT_CATEGORIES, rows)
    device = rng.choice(["known","new","shared"], rows, p=[.88,.09,.03])
    # Synthetic fraud mechanism; deliberately rare, feature-dependent, and explainable.
    fraud_prob = .002 + (amount > 5000)*.018 + (device=="new")*.012 + (channel=="pix")*.003 + (merchant_cat=="electronics")*.004
    fraud = rng.binomial(1, np.clip(fraud_prob, 0, .15))
    fraud_amount = amount.copy()
    fraud_amount[fraud == 1] *= rng.uniform(1.2, 2.5, (fraud == 1).sum())
    tx = pd.DataFrame({
        "transaction_id": [f"T{i:012d}" for i in range(rows)],
        "account_id": accounts.account_id.to_numpy()[cust_idx],
        "customer_id": customers.customer_id.to_numpy()[cust_idx],
        "transaction_timestamp": timestamps,
        "amount": fraud_amount.round(2),
        "currency": "BRL",
        "channel": channel,
        "merchant_category": merchant_cat,
        "city": rng.choice(CITIES, rows),
        "device_status": device,
        "is_fraud": fraud.astype(int),
        "status": rng.choice(["approved","declined"], rows, p=[.94,.06]),
    })
    tx.to_csv(out/"transactions.csv", index=False)
    customers.to_csv(out/"customers.csv", index=False)
    accounts.to_csv(out/"accounts.csv", index=False)
    return {"transactions":tx, "customers":customers, "accounts":accounts}
