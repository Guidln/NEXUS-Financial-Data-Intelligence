"""Fraud classification and anomaly detection with reproducible evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

FEATURES=["amount","hour","day_of_week","is_weekend","device_risk","customer_account_age_days","channel","merchant_category","city","status"]
NUM=["amount","hour","day_of_week","is_weekend","device_risk","customer_account_age_days"]
CAT=["channel","merchant_category","city","status"]

def train_models(data_path="data/silver/transactions.csv", artifacts_dir="artifacts") -> dict:
    p=Path(data_path); out=Path(artifacts_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(p)
    if len(df)<100: raise ValueError("Need at least 100 records to train.")
    X=df[FEATURES].replace([np.inf,-np.inf],np.nan)
    X[NUM]=X[NUM].fillna(0); X[CAT]=X[CAT].fillna("unknown")
    y=df.is_fraud.astype(int)
    strat=y if y.nunique()==2 and y.value_counts().min()>=2 else None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=strat)
    pre=ColumnTransformer([("numeric","passthrough",NUM),("categorical",OneHotEncoder(handle_unknown="ignore"),CAT)])
    clf=Pipeline([("preprocess",pre),("model",RandomForestClassifier(n_estimators=160,max_depth=12,min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=42))])
    clf.fit(Xtr,ytr)
    prob=clf.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int)
    metrics={"model":"RandomForestClassifier","threshold":0.5,"rows":len(df),"fraud_rate":float(y.mean()),
      "precision":float(precision_score(yte,pred,zero_division=0)),
      "recall":float(recall_score(yte,pred,zero_division=0)),
      "f1":float(f1_score(yte,pred,zero_division=0)),
      "roc_auc":float(roc_auc_score(yte,prob)) if yte.nunique()==2 else None,
      "pr_auc":float(average_precision_score(yte,prob)) if yte.nunique()==2 else None,
      "confusion_matrix":confusion_matrix(yte,pred).tolist()}
    # Isolation Forest score over numeric subset; higher normalized score means more anomalous.
    iso=IsolationForest(n_estimators=180,contamination="auto",random_state=42,n_jobs=-1)
    iso.fit(X[NUM])
    raw=-iso.decision_function(X[NUM])
    score=(raw-raw.min())/(raw.max()-raw.min()+1e-9)
    df["fraud_probability"]=clf.predict_proba(X)[:,1]
    df["anomaly_score"]=score
    df["risk_score"]=0.65*df.fraud_probability+0.35*df.anomaly_score
    df.to_csv(out/"scored_transactions.csv",index=False)
    joblib.dump(clf,out/"fraud_model.joblib"); joblib.dump(iso,out/"anomaly_model.joblib")
    (out/"model_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    return metrics
