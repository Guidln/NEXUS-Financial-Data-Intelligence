import json
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="NEXUS | Intelligence",page_icon="🔷",layout="wide")
st.title("🔷 NEXUS Financial Intelligence")
st.caption("Synthetic-data portfolio environment · Fraud & transaction analytics")
DATA=Path("data"); ART=Path("artifacts")
tx_path=DATA/"silver"/"transactions.csv"
if not tx_path.exists():
    st.warning("No pipeline data found. Run `python -m nexus.cli run --rows 100000` from the project directory.")
    st.stop()
tx=pd.read_csv(tx_path,parse_dates=["transaction_timestamp"])
gold=DATA/"gold"/"daily_metrics.csv"
daily=pd.read_csv(gold) if gold.exists() else pd.DataFrame()
scores=ART/"scored_transactions.csv"
scored=pd.read_csv(scores) if scores.exists() else tx
metrics_path=ART/"model_metrics.json"
metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else None

with st.sidebar:
    st.header("Filters")
    channels=st.multiselect("Channel",sorted(tx.channel.unique()),default=list(tx.channel.unique()))
    start=tx.transaction_timestamp.min().date(); end=tx.transaction_timestamp.max().date()
    dates=st.date_input("Date range",value=(start,end))
    threshold=st.slider("Risk score threshold",0.0,1.0,.65,.05)
filtered=tx[tx.channel.isin(channels)]
if isinstance(dates,tuple) and len(dates)==2:
    filtered=filtered[filtered.transaction_timestamp.dt.date.between(dates[0],dates[1])]
cols=st.columns(5)
cols[0].metric("Transactions",f"{len(filtered):,}")
cols[1].metric("Total volume",f"R$ {filtered.amount.sum():,.0f}")
cols[2].metric("Average ticket",f"R$ {filtered.amount.mean():,.2f}" if len(filtered) else "R$ 0")
cols[3].metric("Fraud labels",f"{filtered.is_fraud.sum():,}")
cols[4].metric("Fraud rate",f"{filtered.is_fraud.mean():.2%}" if len(filtered) else "0%")
st.subheader("Transaction volume & activity")
if daily.empty:
    daily=filtered.assign(date=filtered.transaction_timestamp.dt.date.astype(str)).groupby("date",as_index=False).agg(transaction_count=("transaction_id","count"),total_volume=("amount","sum"))
c1,c2=st.columns(2)
with c1: st.plotly_chart(px.line(daily,x="transaction_date" if "transaction_date" in daily else "date",y="total_volume",title="Daily transaction volume"),use_container_width=True)
with c2: st.plotly_chart(px.bar(filtered.groupby("channel",as_index=False).agg(volume=("amount","sum")),x="channel",y="volume",title="Volume by channel"),use_container_width=True)
st.subheader("Fraud & risk")
c3,c4=st.columns(2)
with c3: st.plotly_chart(px.histogram(filtered,x="amount",color="is_fraud",nbins=50,title="Amount distribution by fraud label"),use_container_width=True)
with c4: st.plotly_chart(px.bar(filtered.groupby("merchant_category",as_index=False).agg(fraud_rate=("is_fraud","mean")).sort_values("fraud_rate",ascending=False),x="merchant_category",y="fraud_rate",title="Fraud rate by merchant category"),use_container_width=True)
if "risk_score" in scored:
    st.subheader("Highest risk transactions")
    st.dataframe(scored[scored.risk_score>=threshold].sort_values("risk_score",ascending=False).head(100),use_container_width=True)
if metrics:
    st.subheader("Model performance")
    mc=st.columns(5)
    for col,key in zip(mc,["precision","recall","f1","roc_auc","pr_auc"]):
        col.metric(key.upper(),f"{metrics.get(key,0):.3f}" if metrics.get(key) is not None else "N/A")
    st.json(metrics.get("confusion_matrix"))
st.caption("Simulation only. Fraud labels are synthetic and model performance is not representative of real-world deployments.")
