"""
End-to-end E-commerce Fraud Detection Pipeline (based on the PDF)

What this script covers:
- Data loading & preprocessing
- Advanced feature engineering (time-based, velocity, simple behavior proxies)
- Classical and ensemble models (Logistic Regression, Random Forest, optional XGBoost)
- Cost-sensitive evaluation: EMV, cost curves, lift/gain
- Simple drift monitoring with Population Stability Index (PSI)
- A small in-memory "feature store" + real-time scoring simulation

NOTE:
- You MUST adapt column names to match your dataset.
- Assumed columns in the CSV:
    ["transaction_id", "user_id", "device_id", "ip_address",
     "amount", "timestamp", "label"]
  where label is 1 for fraud, 0 for genuine.
"""

import os
import math
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Try to import XGBoost; continue without it if not installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# =========================
# CONFIGURATION
# =========================

DATA_PATH = "transactions.csv"  # <-- change to your path

# Cost settings (example values; tune for your business)
GAIN_PER_TP = 50.0   # prevented loss per correctly caught fraud
COST_FP = 1.0        # cost per false positive (blocked legit transaction)
COST_FN = 100.0      # cost per missed fraud

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # from training portion


# =========================
# DATA LOADING
# =========================

def load_data(path: str) -> pd.DataFrame:
    """
    Load transaction data from CSV.

    Expected columns (rename or adapt as needed):
        - transaction_id
        - user_id
        - device_id
        - ip_address
        - amount
        - timestamp (string or numeric)
        - label (0 = non-fraud, 1 = fraud)
    """
    df = pd.read_csv(path)
    # basic sanity
    required_cols = [
        "transaction_id", "user_id", "device_id",
        "ip_address", "amount", "timestamp", "label"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df


# =========================
# FEATURE ENGINEERING
# =========================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and velocity features per user and per device.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"])

    # Time since last transaction per user (seconds)
    df["user_time_since_last"] = (
        df.groupby("user_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )
    df["user_time_since_last"].fillna(999999.0, inplace=True)

    # Number of transactions per user in last 1h and 24h (velocity)
    df.set_index("timestamp", inplace=True)

    def rolling_count(group: pd.DataFrame, window: str) -> pd.Series:
        return (
            group["transaction_id"]
            .rolling(window=window)
            .count()
            .shift(1)  # do not include current tx itself
            .fillna(0)
        )

    df["user_tx_1h"] = (
        df.groupby("user_id", group_keys=False)
        .apply(lambda g: rolling_count(g, "1H"))
    )
    df["user_tx_24h"] = (
        df.groupby("user_id", group_keys=False)
        .apply(lambda g: rolling_count(g, "24H"))
    )

    # Device-level velocity (last 24h)
    df["device_tx_24h"] = (
        df.groupby("device_id", group_keys=False)
        .apply(lambda g: rolling_count(g, "24H"))
    )

    # Restore timestamp as a column after using it as index
    df = df.reset_index()

    # Day-of-week and hour-of-day
    df["tx_hour"] = df["timestamp"].dt.hour
    df["tx_dow"] = df["timestamp"].dt.dayofweek  # Monday=0

    return df


def add_behavior_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for behavioral biometrics.

    In a real system these would come from front-end events
    (mouse movements, key timings, touch gestures).
    Here we simulate a few numeric features so the pipeline is end-to-end.
    """
    df = df.copy()

    # Example: decision latency (seconds from page load to checkout)
    # In a real system this comes from logs; here we simulate if missing.
    if "decision_latency" not in df.columns:
        rng = np.random.default_rng(RANDOM_STATE)
        df["decision_latency"] = rng.gamma(shape=2.0, scale=5.0, size=len(df))

    # Example: number of UI events (clicks, scrolls, etc.)
    if "ui_event_count" not in df.columns:
        rng = np.random.default_rng(RANDOM_STATE + 1)
        df["ui_event_count"] = rng.integers(1, 50, size=len(df))

    # Example: "smoothness" metric of mouse paths (0-1)
    if "mouse_smoothness" not in df.columns:
        rng = np.random.default_rng(RANDOM_STATE + 2)
        df["mouse_smoothness"] = rng.uniform(0.0, 1.0, size=len(df))

    return df


def add_simple_ip_device_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic frequency-based risk signals for IP and device.
    """
    df = df.copy()

    # total transactions per IP / device
    ip_counts = df["ip_address"].value_counts()
    device_counts = df["device_id"].value_counts()
    df["ip_tx_count"] = df["ip_address"].map(ip_counts)
    df["device_tx_count"] = df["device_id"].map(device_counts)

    # fraud rate per IP / device (smoothed)
    ip_fraud_rate = (
        df.groupby("ip_address")["label"]
        .mean()
        .rename("ip_fraud_rate")
    )
    device_fraud_rate = (
        df.groupby("device_id")["label"]
        .mean()
        .rename("device_fraud_rate")
    )

    # Add Laplace smoothing
    df = df.join(ip_fraud_rate, on="ip_address")
    df = df.join(device_fraud_rate, on="device_id")
    df["ip_fraud_rate"].fillna(df["label"].mean(), inplace=True)
    df["device_fraud_rate"].fillna(df["label"].mean(), inplace=True)

    return df


def build_feature_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create final feature matrix X and target y.
    """
    df = add_time_features(df)
    df = add_behavior_proxies(df)
    df = add_simple_ip_device_stats(df)

    # Select features (numeric + categorical)
    numeric_features = [
        "amount",
        "user_time_since_last",
        "user_tx_1h",
        "user_tx_24h",
        "device_tx_24h",
        "tx_hour",
        "tx_dow",
        "decision_latency",
        "ui_event_count",
        "mouse_smoothness",
        "ip_tx_count",
        "device_tx_count",
        "ip_fraud_rate",
        "device_fraud_rate"
    ]

    categorical_features = [
        "user_id",      # often dropped in real systems; kept here for example
        "device_id",
        "ip_address"
    ]

    all_features = numeric_features + categorical_features

    # Drop rows with missing target
    df = df.dropna(subset=["label"])

    X = df[all_features].copy()
    y = df["label"].astype(int)

    return X, y


# =========================
# COST-SENSITIVE EVALUATION
# =========================

def compute_emv_from_counts(tp, fp, fn,
                            gain_per_tp=GAIN_PER_TP,
                            cost_fp=COST_FP,
                            cost_fn=COST_FN) -> float:
    """
    Expected Monetary Value given confusion matrix counts.
    """
    return tp * gain_per_tp - fp * cost_fp - fn * cost_fn


def evaluate_thresholds(y_true: np.ndarray,
                        y_scores: np.ndarray,
                        thresholds: np.ndarray) -> pd.DataFrame:
    """
    Compute cost metrics and EMV for many thresholds.
    """
    rows = []
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        emv = compute_emv_from_counts(tp, fp, fn)
        rows.append({
            "threshold": t,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "EMV": emv
        })
    return pd.DataFrame(rows)


def find_best_threshold_by_emv(y_true: np.ndarray,
                               y_scores: np.ndarray) -> Tuple[float, pd.DataFrame]:
    """
    Find the threshold that maximises EMV.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    df_cost = evaluate_thresholds(y_true, y_scores, thresholds)
    best_row = df_cost.loc[df_cost["EMV"].idxmax()]
    return float(best_row["threshold"]), df_cost


def compute_lift_table(y_true: np.ndarray,
                       y_scores: np.ndarray,
                       n_bins: int = 10) -> pd.DataFrame:
    """
    Compute lift table: how concentrated fraud is in top score quantiles.
    """
    df = pd.DataFrame({"y_true": y_true, "y_scores": y_scores})
    df = df.sort_values("y_scores", ascending=False).reset_index(drop=True)
    df["bin"] = pd.qcut(df.index, q=n_bins, labels=False)

    lift_rows = []
    overall_rate = df["y_true"].mean()

    for b in range(n_bins):
        segment = df[df["bin"] == b]
        if len(segment) == 0:
            continue
        rate = segment["y_true"].mean()
        lift = rate / overall_rate if overall_rate > 0 else np.nan
        lift_rows.append({
            "bin": b + 1,  # 1-based
            "n_samples": len(segment),
            "fraud_rate": rate,
            "lift": lift
        })

    return pd.DataFrame(lift_rows)


# =========================
# DRIFT MONITORING (PSI)
# =========================

def population_stability_index(expected: np.ndarray,
                               actual: np.ndarray,
                               bins: int = 10) -> float:
    """
    Compute PSI between expected and actual distributions.

    expected: scores from reference period
    actual: scores from current period
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    # Same bin edges on both
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(expected, quantiles)

    # Avoid duplicate edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) <= 2:
        # degenerate; no variation
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    psi = 0.0
    for e, a in zip(expected_perc, actual_perc):
        if e == 0 or a == 0:
            continue
        psi += (a - e) * math.log(a / e)

    return psi


# =========================
# FEATURE TRANSFORMER & MODELS
# =========================

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build ColumnTransformer for numeric and categorical columns.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    """
    Define baseline and modern models.
    """
    models = {}

    # Logistic Regression (baseline)
    lr_clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )
    models["logistic_regression"] = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", lr_clf)
    ])

    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE
    )
    models["random_forest"] = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", rf_clf)
    ])

    # Optional XGBoost
    if XGBOOST_AVAILABLE:
        xgb_clf = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        models["xgboost"] = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", xgb_clf)
        ])

    return models


# =========================
# REAL-TIME FEATURE STORE (SIMPLE)
# =========================

@dataclass
class UserHistory:
    """
    Minimal in-memory history for one user.
    """
    last_timestamps: List[pd.Timestamp]
    last_amounts: List[float]


class InMemoryFeatureStore:
    """
    Very simplified real-time feature store:
    - caches recent transactions per user
    - computes lightweight velocity features online

    NOTE: This is just to illustrate the idea from the PDF.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.user_histories: Dict[str, UserHistory] = {}

    def _ensure_user(self, user_id: str):
        if user_id not in self.user_histories:
            self.user_histories[user_id] = UserHistory(
                last_timestamps=[],
                last_amounts=[]
            )

    def update_user(self, user_id: str, ts: pd.Timestamp, amount: float):
        self._ensure_user(user_id)
        hist = self.user_histories[user_id]
        hist.last_timestamps.append(ts)
        hist.last_amounts.append(amount)
        # Keep only last N
        if len(hist.last_timestamps) > self.window_size:
            hist.last_timestamps = hist.last_timestamps[-self.window_size:]
            hist.last_amounts = hist.last_amounts[-self.window_size:]

    def compute_online_features(self, user_id: str,
                                ts: pd.Timestamp,
                                amount: float) -> Dict[str, float]:
        """
        Compute simple online features for a new transaction:
        - time since last tx
        - count in last 1h
        - count in last 24h
        - mean amount in last 7 days
        """
        self._ensure_user(user_id)
        hist = self.user_histories[user_id]

        if len(hist.last_timestamps) == 0:
            time_since_last = 999999.0
        else:
            time_since_last = (ts - hist.last_timestamps[-1]).total_seconds()

        # 1h and 24h velocities & 7d mean
        one_hour_ago = ts - pd.Timedelta(hours=1)
        one_day_ago = ts - pd.Timedelta(days=1)
        seven_days_ago = ts - pd.Timedelta(days=7)

        last_ts = np.array(hist.last_timestamps)
        last_amt = np.array(hist.last_amounts)

        tx_1h = int(np.sum(last_ts >= one_hour_ago))
        tx_24h = int(np.sum(last_ts >= one_day_ago))
        mask_7d = (last_ts >= seven_days_ago)
        mean_amt_7d = float(last_amt[mask_7d].mean()) if np.any(mask_7d) else 0.0

        return {
            "online_user_time_since_last": time_since_last,
            "online_user_tx_1h": float(tx_1h),
            "online_user_tx_24h": float(tx_24h),
            "online_mean_amount_7d": mean_amt_7d,
        }


def simulate_real_time_scoring(trained_pipeline: Pipeline,
                               feature_store: InMemoryFeatureStore,
                               raw_event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate scoring a single incoming transaction event in real time.

    raw_event example:
    {
        "transaction_id": "tx123",
        "user_id": "u1",
        "device_id": "d1",
        "ip_address": "1.2.3.4",
        "amount": 100.0,
        "timestamp": "2025-01-01T12:30:00Z",
        "decision_latency": 4.5,
        "ui_event_count": 30,
        "mouse_smoothness": 0.6
    }
    """
    # Convert timestamp
    ts = pd.to_datetime(raw_event["timestamp"])

    # 1) online features from feature store
    online_feats = feature_store.compute_online_features(
        user_id=str(raw_event["user_id"]),
        ts=ts,
        amount=float(raw_event["amount"])
    )

    # 2) prepare single-row DataFrame aligned with training features
    row = {
        "amount": float(raw_event["amount"]),
        "user_time_since_last": online_feats["online_user_time_since_last"],
        "user_tx_1h": online_feats["online_user_tx_1h"],
        "user_tx_24h": online_feats["online_user_tx_24h"],
        "device_tx_24h": 0.0,  # unknown in this tiny example
        "tx_hour": ts.hour,
        "tx_dow": ts.dayofweek,
        "decision_latency": float(raw_event.get("decision_latency", 5.0)),
        "ui_event_count": float(raw_event.get("ui_event_count", 20)),
        "mouse_smoothness": float(raw_event.get("mouse_smoothness", 0.5)),
        "ip_tx_count": 0.0,         # would be filled if we keep an IP cache
        "device_tx_count": 0.0,     # same idea
        "ip_fraud_rate": 0.01,      # default prior
        "device_fraud_rate": 0.01,
        "user_id": str(raw_event["user_id"]),
        "device_id": str(raw_event["device_id"]),
        "ip_address": str(raw_event["ip_address"])
    }

    X_event = pd.DataFrame([row])

    # 3) score
    fraud_prob = float(trained_pipeline.predict_proba(X_event)[:, 1][0])

    # Example: use 0.5; in real deployment use EMV-optimised threshold
    decision = "REJECT" if fraud_prob >= 0.5 else "ACCEPT"

    # 4) update feature store AFTER scoring
    feature_store.update_user(
        user_id=str(raw_event["user_id"]),
        ts=ts,
        amount=float(raw_event["amount"])
    )

    return {
        "fraud_probability": fraud_prob,
        "decision": decision,
        "online_features": online_feats
    }


# =========================
# MAIN TRAINING & EVAL
# =========================

def main():
    # 1) Load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH}. Please update DATA_PATH."
        )

    df = load_data(DATA_PATH)
    print(f"[INFO] Loaded dataset with {len(df)} rows")

    # 2) Build features & target
    X, y = build_feature_dataframe(df)
    print(f"[INFO] Feature matrix shape: {X.shape}")

    # 3) Train/val/test split (stratified)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    val_relative_size = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_relative_size,
        stratify=y_train_full,
        random_state=RANDOM_STATE
    )

    print(f"[INFO] Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # 4) Preprocessor and models
    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor)

    # 5) Fit & evaluate each model
    best_model_name = None
    best_model = None
    best_val_auc = -np.inf
    model_results = {}

    for name, model in models.items():
        print(f"\n[TRAIN] Fitting model: {name}")
        model.fit(X_train, y_train)

        # Probabilities on validation
        val_scores = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, val_scores)
        pr_auc = average_precision_score(y_val, val_scores)
        print(f"[VAL] {name} ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

        # EMV-based threshold selection
        best_threshold, cost_table = find_best_threshold_by_emv(y_val.values, val_scores)
        print(f"[VAL] {name} Best EMV threshold: {best_threshold:.3f}")
        print(cost_table.sort_values("EMV", ascending=False).head())

        # Lift table
        lift_table = compute_lift_table(y_val.values, val_scores, n_bins=10)
        print(f"[VAL] {name} Lift table (top 3 bins):")
        print(lift_table.sort_values("bin").head(3))

        model_results[name] = {
            "model": model,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "best_threshold": best_threshold
        }

        # Track best by ROC-AUC
        if roc_auc > best_val_auc:
            best_val_auc = roc_auc
            best_model_name = name
            best_model = model

    print(f"\n[INFO] Best model by ROC-AUC on validation: {best_model_name}")

    # 6) Final test evaluation of best model
    if best_model is None:
        raise RuntimeError("No model was trained.")

    test_scores = best_model.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, test_scores)
    test_pr_auc = average_precision_score(y_test, test_scores)
    print(f"\n[TEST] {best_model_name} ROC-AUC: {test_roc_auc:.4f}, PR-AUC: {test_pr_auc:.4f}")

    # Use EMV-optimised threshold from validation
    best_threshold = model_results[best_model_name]["best_threshold"]
    y_test_pred = (test_scores >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    emv_test = compute_emv_from_counts(tp, fp, fn)
    print(f"[TEST] Confusion matrix (tn, fp, fn, tp): {tn}, {fp}, {fn}, {tp}")
    print(f"[TEST] EMV at threshold {best_threshold:.3f}: {emv_test:.2f}")
    print("[TEST] Classification report:")
    print(classification_report(y_test, y_test_pred))

    # 7) Simple drift monitoring example with PSI
    #    Assume "train" scores are reference, "test" scores are current
    train_scores_ref = best_model.predict_proba(X_train)[:, 1]
    psi_value = population_stability_index(train_scores_ref, test_scores, bins=10)
    print(f"\n[DRIFT] PSI between train and test score distributions: {psi_value:.4f}")
    if psi_value > 0.25:
        print("[DRIFT] WARNING: PSI suggests significant drift. Consider retraining / investigating.")
    elif psi_value > 0.1:
        print("[DRIFT] PSI indicates moderate drift. Monitor closely.")
    else:
        print("[DRIFT] PSI indicates minimal drift.")

    # 8) Demonstrate real-time scoring simulation
    feature_store = InMemoryFeatureStore(window_size=50)

    # Prime the feature store with some historical transactions for a user
    # In production this happens naturally over time as events stream in
    example_user_id = str(X_train.index[0])  # arbitrary; you would use real user IDs

    # For realism, let's pull a few historical rows for that user (if any exist)
    # This requires we still have the original df with timestamps
    # Here, we just simulate a couple of past events
    for day_offset in [3, 1]:
        past_event = {
            "transaction_id": f"hist_{day_offset}",
            "user_id": example_user_id,
            "device_id": "device_sim",
            "ip_address": "10.0.0.1",
            "amount": 50.0 + 10 * day_offset,
            "timestamp": (pd.Timestamp.utcnow() - pd.Timedelta(days=day_offset)).isoformat(),
            "decision_latency": 5.0,
            "ui_event_count": 25,
            "mouse_smoothness": 0.7
        }
        _ = simulate_real_time_scoring(best_model, feature_store, past_event)

    # Now score a new transaction for the same user
    new_event = {
        "transaction_id": "tx_live_001",
        "user_id": example_user_id,
        "device_id": "device_sim",
        "ip_address": "10.0.0.1",
        "amount": 120.0,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "decision_latency": 3.2,
        "ui_event_count": 15,
        "mouse_smoothness": 0.4
    }
    live_result = simulate_real_time_scoring(best_model, feature_store, new_event)
    print("\n[REAL-TIME] Live scoring result for example event:")
    print(json.dumps(live_result, indent=2))


if __name__ == "__main__":
    main()
