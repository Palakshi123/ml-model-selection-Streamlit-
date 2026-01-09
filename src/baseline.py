import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

def build_and_eval_baseline(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    y = df[target_col]
    X = df.drop(columns=[target_col])

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", cat_tf, cat_cols),
        ],
        remainder="drop",
    )

    # Pick a robust baseline:
    # - Classification: LogisticRegression if sparse/high-dim; else HistGB is strong too.
    # We'll run ONE baseline for speed: HistGB (needs dense) can be tricky with sparse one-hot,
    # so default to LogisticRegression for classification, Ridge for regression.
    notes = []
    if problem_type == "classification":
        model = LogisticRegression(max_iter=2000, n_jobs=None)
        model_name = "LogisticRegression(max_iter=2000)"
        notes.append("Using Logistic Regression as fast, reliable baseline for classification.")
    else:
        model = Ridge(alpha=1.0, random_state=random_state)
        model_name = "Ridge(alpha=1.0)"
        notes.append("Using Ridge Regression as fast, reliable baseline for regression.")

    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])

    stratify = y if problem_type == "classification" and y.nunique(dropna=True) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    if problem_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro")),
        }
        # AUC only for binary + if predict_proba available
        if y_test.nunique(dropna=True) == 2 and hasattr(pipe.named_steps["model"], "predict_proba"):
            proba = pipe.predict_proba(X_test)[:, 1]
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
            except Exception:
                pass
        else:
            notes.append("ROC-AUC not computed (non-binary target or proba not available).")
    else:
        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(mean_squared_error(y_test, preds, squared=False)),
            "r2": float(r2_score(y_test, preds)),
        }

    return {"model_name": model_name, "metrics": metrics, "notes": notes}
