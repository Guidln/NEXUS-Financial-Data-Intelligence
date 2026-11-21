"""Data quality checks and report serialization."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def evaluate_quality(tx: pd.DataFrame, accounts: pd.DataFrame, customers: pd.DataFrame) -> dict:
    checks = {}
    checks["unique_transaction_id"] = bool(tx.transaction_id.is_unique)
    checks["non_null_required"] = bool(tx[["transaction_id","account_id","customer_id","transaction_timestamp","amount"]].notna().all().all())
    checks["positive_amount"] = bool((tx.amount > 0).all())
    checks["fraud_label_binary"] = bool(tx.is_fraud.isin([0,1]).all())
    checks["account_referential_integrity"] = bool(tx.account_id.isin(accounts.account_id).all())
    checks["customer_referential_integrity"] = bool(tx.customer_id.isin(customers.customer_id).all())
    checks["valid_timestamps"] = bool(pd.to_datetime(tx.transaction_timestamp, errors="coerce").notna().all())
    duplicate_count = int(tx.transaction_id.duplicated().sum())
    null_cells = int(tx.isna().sum().sum())
    score = round(100 * sum(checks.values()) / max(len(checks),1), 2)
    return {"dataset":"transactions", "rows":len(tx), "quality_score":score, "checks":checks,
            "duplicate_transaction_ids":duplicate_count, "null_cells":null_cells,
            "fraud_rate":round(float(tx.is_fraud.mean()),6), "generated_at":pd.Timestamp.utcnow().isoformat()}

def save_report(report: dict, path="artifacts/quality_report.json"):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
