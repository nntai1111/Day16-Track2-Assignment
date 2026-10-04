"""LightGBM benchmark on the Credit Card Fraud Detection dataset (Lab 16, CPU flow)."""
import json
import os
import platform
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.expanduser("~/ml-benchmark/creditcard.csv")
RESULT_PATH = os.path.expanduser("~/ml-benchmark/benchmark_result.json")
SEED = 42
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")

print("=" * 60)
print(" LAB 16 - LightGBM Benchmark (Credit Card Fraud Detection)")
print("=" * 60)
print(f"Host: {platform.node()} | CPU cores: {os.cpu_count()} | LightGBM {lgb.__version__}")

# 1. Load data
t0 = time.perf_counter()
df = pd.read_csv(DATA_PATH)
load_time = time.perf_counter() - t0
X = df.drop(columns=["Class"])
y = df["Class"]
print(f"\n[1] Loaded {len(df):,} rows x {X.shape[1]} features in {load_time:.3f}s")
print(f"    Fraud ratio: {y.mean() * 100:.4f}% ({int(y.sum())} frauds)")

# Split train / valid / test (stratified because the data is highly imbalanced)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED)
print(f"    Train: {len(X_tr):,} | Valid: {len(X_val):,} | Test: {len(X_test):,}")

# 2. Train
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_child_samples=50,
    reg_lambda=1.0,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
t0 = time.perf_counter()
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100, verbose=False, first_metric_only=True)],
)
train_time = time.perf_counter() - t0
best_iter = int(model.best_iteration_)
print(f"\n[2] Training done in {train_time:.3f}s (best iteration: {best_iter})")

# 3. Evaluate
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
metrics = {
    "auc_roc": roc_auc_score(y_test, proba),
    "accuracy": accuracy_score(y_test, pred),
    "f1_score": f1_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
}
print("\n[3] Test metrics")
for k, v in metrics.items():
    print(f"    {k:<10}: {v:.4f}")

# 4. Inference latency (1 row) and throughput (1000 rows)
row = X_test.iloc[[0]]
for _ in range(10):  # warm-up
    model.predict_proba(row)
lat = []
for _ in range(200):
    t0 = time.perf_counter()
    model.predict_proba(row)
    lat.append(time.perf_counter() - t0)
latency_ms = float(np.median(lat) * 1000)

batch = X_test.iloc[:1000]
runs = []
for _ in range(20):
    t0 = time.perf_counter()
    model.predict_proba(batch)
    runs.append(time.perf_counter() - t0)
batch_ms = float(np.median(runs) * 1000)
throughput = 1000 / (batch_ms / 1000)
print("\n[4] Inference")
print(f"    Latency (1 row, median of 200)     : {latency_ms:.3f} ms")
print(f"    1000 rows (median of 20)           : {batch_ms:.3f} ms")
print(f"    Throughput                         : {throughput:,.0f} rows/s")

result = {
    "host": platform.node(),
    "cpu_cores": os.cpu_count(),
    "lightgbm_version": lgb.__version__,
    "dataset": "mlg-ulb/creditcardfraud",
    "rows": int(len(df)),
    "features": int(X.shape[1]),
    "load_time_s": round(load_time, 4),
    "training_time_s": round(train_time, 4),
    "best_iteration": best_iter,
    **{k: round(float(v), 6) for k, v in metrics.items()},
    "inference_latency_1row_ms": round(latency_ms, 4),
    "inference_1000rows_ms": round(batch_ms, 4),
    "inference_throughput_rows_per_s": round(throughput, 1),
}
with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved results to {RESULT_PATH}")
print("=" * 60)
