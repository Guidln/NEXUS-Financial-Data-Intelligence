"""FastAPI read and prediction endpoints."""
from pathlib import Path
import json
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

app=FastAPI(title="NEXUS Financial Intelligence API",version="1.0.0",description="Synthetic-data portfolio demo API.")
DATA=Path("data"); ART=Path("artifacts")
FEATURES=["amount","hour","day_of_week","is_weekend","device_risk","customer_account_age_days","channel","merchant_category","city","status"]

class PredictionRequest(BaseModel):
    amount: float=Field(gt=0,le=1_000_000)
    hour: int=Field(ge=0,le=23)
    day_of_week: int=Field(ge=0,le=6)
    is_weekend: int=Field(ge=0,le=1)
    device_risk: float=Field(ge=0,le=1)
    customer_account_age_days: int=Field(ge=0)
    channel: str="pix"
    merchant_category: str="retail"
    city: str="São Paulo"
    status: str="approved"

def read_csv(path):
    if not path.exists(): raise HTTPException(status_code=404,detail="Pipeline output not found. Run `python -m nexus.cli run` first.")
    return pd.read_csv(path)

@app.get("/health")
def health(): return {"status":"ok","service":"nexus-api"}

@app.get("/metrics")
def metrics():
    report=ART/"quality_report.json"; model=ART/"model_metrics.json"
    return {"quality":json.loads(report.read_text()) if report.exists() else None,
            "model":json.loads(model.read_text()) if model.exists() else None}

@app.get("/analytics/overview")
def overview():
    tx=read_csv(DATA/"silver"/"transactions.csv")
    return {"transactions":int(len(tx)),"volume":float(tx.amount.sum()),"average_ticket":float(tx.amount.mean()),
            "fraud_count":int(tx.is_fraud.sum()),"fraud_rate":float(tx.is_fraud.mean()),
            "approval_rate":float(tx.status.eq("approved").mean())}

@app.get("/analytics/daily")
def daily():
    return read_csv(DATA/"gold"/"daily_metrics.csv").to_dict(orient="records")

@app.get("/transactions")
def transactions(limit:int=Query(50,ge=1,le=500),offset:int=Query(0,ge=0)):
    df=read_csv(DATA/"silver"/"transactions.csv")
    return df.sort_values("transaction_timestamp",ascending=False).iloc[offset:offset+limit].to_dict(orient="records")

@app.get("/fraud/alerts")
def alerts(threshold:float=Query(.65,ge=0,le=1),limit:int=Query(50,ge=1,le=500)):
    df=read_csv(ART/"scored_transactions.csv")
    if "risk_score" not in df: raise HTTPException(404,"Model scores unavailable")
    return df[df.risk_score>=threshold].sort_values("risk_score",ascending=False).head(limit).to_dict(orient="records")

@app.post("/fraud/predict")
def predict(request:PredictionRequest):
    model_path=ART/"fraud_model.joblib"
    if not model_path.exists(): raise HTTPException(404,"Model not trained. Run training pipeline first.")
    model=joblib.load(model_path)
    row=pd.DataFrame([request.model_dump()])[FEATURES]
    probability=float(model.predict_proba(row)[0,1])
    return {"fraud_probability":probability,"risk_level":"high" if probability>=.65 else "medium" if probability>=.30 else "low",
            "threshold":.5,"disclaimer":"Synthetic educational model; not a financial decision."}
