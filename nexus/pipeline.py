"""Batch pipeline: raw bronze to clean silver and analytical gold."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .quality import evaluate_quality, save_report

def run_pipeline(data_dir="data", artifacts_dir="artifacts") -> dict:
    root=Path(data_dir); bronze=root/"bronze"; silver=root/"silver"; gold=root/"gold"
    silver.mkdir(parents=True,exist_ok=True); gold.mkdir(parents=True,exist_ok=True)
    tx=pd.read_csv(bronze/"transactions.csv",parse_dates=["transaction_timestamp"])
    accounts=pd.read_csv(bronze/"accounts.csv")
    customers=pd.read_csv(bronze/"customers.csv",parse_dates=["account_open_date"])
    report=evaluate_quality(tx,accounts,customers)
    # Cleaning: normalize categories, remove duplicates, enforce valid records.
    tx=tx.drop_duplicates("transaction_id").copy()
    tx["channel"]=tx.channel.str.strip().str.lower()
    tx["merchant_category"]=tx.merchant_category.str.strip().str.lower()
    tx["transaction_date"]=tx.transaction_timestamp.dt.date.astype(str)
    tx["hour"]=tx.transaction_timestamp.dt.hour
    tx["day_of_week"]=tx.transaction_timestamp.dt.dayofweek
    tx["is_weekend"]=(tx.day_of_week>=5).astype(int)
    tx=tx[tx.amount.gt(0) & tx.transaction_timestamp.notna()]
    tx["amount_log"]=tx.amount.clip(lower=0).apply(lambda x: __import__("numpy").log1p(x))
    tx["customer_account_age_days"]=(tx.transaction_timestamp - tx.customer_id.map(customers.set_index("customer_id").account_open_date)).dt.days.clip(lower=0)
    tx["device_risk"]=tx.device_status.map({"known":0.1,"new":0.65,"shared":0.9}).fillna(.5)
    tx.to_csv(silver/"transactions.csv",index=False)
    try: tx.to_parquet(silver/"transactions.parquet",index=False)
    except Exception: pass
    accounts.to_csv(silver/"accounts.csv",index=False); customers.to_csv(silver/"customers.csv",index=False)
    daily=tx.groupby("transaction_date",as_index=False).agg(
        transaction_count=("transaction_id","count"), total_volume=("amount","sum"),
        average_ticket=("amount","mean"), fraud_count=("is_fraud","sum"),
        fraud_volume=("amount",lambda s: float(s[tx.loc[s.index,"is_fraud"].eq(1)].sum())),
        approval_rate=("status",lambda s: float(s.eq("approved").mean()))
    )
    daily["fraud_rate"]=daily.fraud_count/daily.transaction_count
    daily.to_csv(gold/"daily_metrics.csv",index=False)
    customer=tx.groupby("customer_id",as_index=False).agg(
        transactions=("transaction_id","count"), total_volume=("amount","sum"),
        average_ticket=("amount","mean"), fraud_count=("is_fraud","sum"),
        last_transaction=("transaction_timestamp","max")
    )
    customer["risk_segment"]=pd.cut(customer.fraud_count,[-1,0,1,100000],labels=["low","medium","high"]).astype(str)
    customer.to_csv(gold/"customer_metrics.csv",index=False)
    segments=tx.groupby(["channel","merchant_category"],as_index=False).agg(
        count=("transaction_id","count"), volume=("amount","sum"), frauds=("is_fraud","sum"),
        fraud_rate=("is_fraud","mean")
    ).sort_values("fraud_rate",ascending=False)
    segments.to_csv(gold/"fraud_segments.csv",index=False)
    save_report(report,Path(artifacts_dir)/"quality_report.json")
    return {"transactions":tx,"daily":daily,"customers":customer,"segments":segments,"quality":report}
