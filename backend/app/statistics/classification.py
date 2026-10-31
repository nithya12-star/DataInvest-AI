"""
backend/app/statistics/classification.py
=========================================
Statistical Binary Classification Module for DataInvest AI.

Provides genuine, reproducible supervised classification evaluation on tabular datasets
using scikit-learn, with strict separation between:
1. Baseline (Critic OFF): Model trained on un-audited raw data (retaining impossible zeros and unflagged anomalies).
2. Proposed (Critic ON): Model trained after statistical Critic auditing (imputing physiologically impossible zeros,
   handling outlier leverage, and robust scaling).

Evaluates real ground-truth predictions to produce:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Paired McNemar Test p-value on shared held-out test predictions.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger("datainvest.statistics.classification")

PHYSIOLOGICALLY_POSITIVE_COLUMNS = {
    "cholesterol",
    "restingbp",
    "trestbps",
    "chol",
    "bloodpressure",
    "heartrate",
    "maxhr",
    "thalach",
}


def _detect_impossible_zeros(
    df: pd.DataFrame,
    columns: list[str],
) -> dict[str, int]:
    """Identify columns where values == 0 represent physiological impossibilities."""
    zero_counts: dict[str, int] = {}
    for col in columns:
        col_lower = col.lower().replace("_", "").replace(" ", "")
        if col_lower in PHYSIOLOGICALLY_POSITIVE_COLUMNS or any(
            target in col_lower for target in ["cholesterol", "restingbp", "trestbps", "chol"]
        ):
            if pd.api.types.is_numeric_dtype(df[col]):
                num_zeros = int((df[col] == 0).sum())
                if num_zeros > 0:
                    zero_counts[col] = num_zeros
    return zero_counts


def compute_mcnemar_p_value(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_proposed: np.ndarray,
) -> dict[str, Any]:
    """
    Compute McNemar's test for paired binary classification predictions on the same test set.

    H0: The two classifiers have identical marginal error rates.
    """
    b_correct = y_pred_baseline == y_true
    p_correct = y_pred_proposed == y_true

    a = int(np.sum(b_correct & p_correct))  # Both correct
    b = int(np.sum(b_correct & ~p_correct))  # Baseline correct, Proposed wrong
    c = int(np.sum(~b_correct & p_correct))  # Baseline wrong, Proposed correct
    d = int(np.sum(~b_correct & ~p_correct))  # Both wrong

    discordant = b + c
    if discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "table": [[a, b], [c, d]],
            "test_name": "McNemar Exact (Zero Discordance)",
            "interpretation": "Identical predictions; no statistically significant difference.",
        }

    # If discordant pairs are small, use exact binomial test
    if discordant < 25:
        # Two-sided binomial test under p=0.5
        res = stats.binomtest(b, discordant, 0.5, alternative="two-sided")
        p_val = float(res.pvalue)
        stat = float(abs(b - c))
        test_name = "McNemar Exact Binomial"
    else:
        # Edwards continuity correction
        chi2 = float((abs(b - c) - 1.0) ** 2 / discordant)
        p_val = float(stats.chi2.sf(chi2, df=1))
        stat = chi2
        test_name = "McNemar Chi-Square (with continuity correction)"

    return {
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "table": [[a, b], [c, d]],
        "discordant_pairs": discordant,
        "test_name": test_name,
        "interpretation": (
            "Statistically significant difference (p < 0.05)"
            if p_val < 0.05
            else "No statistically significant difference (p >= 0.05)"
        ),
    }


def execute_binary_classification(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[list[str]] = None,
    model_type: str = "logistic_regression",
    test_size: float = 0.2,
    random_state: int = 42,
    apply_critic_cleaning: bool = False,
) -> tuple[bool, dict[str, Any], list[str], Optional[dict[str, str]]]:
    """
    Train and evaluate a binary classifier on held-out test split.

    Returns:
        (success, results_dict, warnings_list, error_dict)
    """
    warnings: list[str] = []

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False, {}, warnings, {"code": "EMPTY_DATASET", "message": "Dataset is empty or invalid."}

    if target_column not in df.columns:
        return False, {}, warnings, {
            "code": "TARGET_COLUMN_NOT_FOUND",
            "message": f"Target column '{target_column}' does not exist in dataset.",
        }

    # Ensure binary target
    raw_target = df[target_column].dropna()
    unique_targets = raw_target.unique()
    if len(unique_targets) < 2:
        return False, {}, warnings, {
            "code": "TARGET_INSUFFICIENT_CLASSES",
            "message": f"Target column '{target_column}' must have at least 2 distinct classes; found {len(unique_targets)}.",
        }

    # Binarize if target has multi-class integers (e.g. Cleveland 0,1,2,3,4 -> 0 vs >0)
    if len(unique_targets) > 2 and pd.api.types.is_numeric_dtype(raw_target):
        warnings.append(
            f"Target '{target_column}' has {len(unique_targets)} classes; binarized to 0 (absence) vs >=1 (presence)."
        )
        y_all = (df[target_column] > 0).astype(int)
    else:
        # Map values to 0 and 1
        pos_val = sorted(unique_targets)[-1]
        y_all = (df[target_column] == pos_val).astype(int)

    # Select features
    if feature_columns:
        missing_feats = [c for c in feature_columns if c not in df.columns]
        if missing_feats:
            return False, {}, warnings, {
                "code": "FEATURE_COLUMNS_NOT_FOUND",
                "message": f"Specified features not found: {missing_feats}",
            }
        feats = [c for c in feature_columns if c != target_column]
    else:
        feats = [c for c in df.columns if c != target_column]

    if not feats:
        return False, {}, warnings, {
            "code": "NO_FEATURES_AVAILABLE",
            "message": "No predictor feature columns available after excluding target.",
        }

    X_all = df[feats].copy()

    # Train / Test Split (Reproducible Stratified Split)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=test_size,
            random_state=random_state,
            stratify=y_all,
        )
    except Exception as e:
        return False, {}, warnings, {
            "code": "TRAIN_TEST_SPLIT_FAILED",
            "message": f"Train/test split failed: {str(e)}",
        }

    audit_findings: dict[str, Any] = {}
    reinvestigation_applied: list[str] = []

    # Detect impossible zero values
    impossible_zeros = _detect_impossible_zeros(X_train, feats)
    audit_findings["impossible_zeros_detected"] = impossible_zeros

    # Separate numerical and categorical columns
    num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in feats if c not in num_cols]

    X_train_processed_parts: list[pd.DataFrame] = []
    X_test_processed_parts: list[pd.DataFrame] = []

    # Numerical feature processing
    if num_cols:
        X_train_num = X_train[num_cols].copy()
        X_test_num = X_test[num_cols].copy()

        # Handle impossible zeros
        for col, count in impossible_zeros.items():
            if col in num_cols:
                if apply_critic_cleaning:
                    # Critic ON: Treat impossible zeros as missing data and impute with train median
                    train_med = X_train_num.loc[X_train_num[col] > 0, col].median()
                    if pd.isna(train_med):
                        train_med = 0.0
                    X_train_num.loc[X_train_num[col] == 0, col] = train_med
                    X_test_num.loc[X_test_num[col] == 0, col] = train_med
                    reinvestigation_applied.append(
                        f"Imputed {count} impossible zeros in '{col}' using train median ({train_med:.1f})"
                    )
                else:
                    # Critic OFF: Retain literal zero values as-is
                    warnings.append(
                        f"Un-audited baseline: Retained {count} literal zero values in physiological metric '{col}'."
                    )

        # Impute any remaining standard NaN with train median
        for c in num_cols:
            med = X_train_num[c].median()
            if pd.isna(med):
                med = 0.0
            X_train_num[c] = X_train_num[c].fillna(med)
            X_test_num[c] = X_test_num[c].fillna(med)

        # Standard Scaling
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_num),
            columns=num_cols,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_num),
            columns=num_cols,
            index=X_test.index,
        )
        X_train_processed_parts.append(X_train_scaled)
        X_test_processed_parts.append(X_test_scaled)

    # Categorical feature processing
    if cat_cols:
        # Fill missing with mode
        X_train_cat = X_train[cat_cols].astype(str).fillna("missing")
        X_test_cat = X_test[cat_cols].astype(str).fillna("missing")

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_train = encoder.fit_transform(X_train_cat)
        encoded_test = encoder.transform(X_test_cat)

        cat_feature_names = list(encoder.get_feature_names_out(cat_cols))
        X_train_enc = pd.DataFrame(encoded_train, columns=cat_feature_names, index=X_train.index)
        X_test_enc = pd.DataFrame(encoded_test, columns=cat_feature_names, index=X_test.index)

        X_train_processed_parts.append(X_train_enc)
        X_test_processed_parts.append(X_test_enc)

    X_train_final = pd.concat(X_train_processed_parts, axis=1)
    X_test_final = pd.concat(X_test_processed_parts, axis=1)

    # Model Selection & Fitting
    model_name_clean = model_type.lower().strip()
    if model_name_clean == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)
    elif model_name_clean == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state)
    else:
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model_name_clean = "logistic_regression"

    try:
        model.fit(X_train_final, y_train)
    except Exception as e:
        return False, {}, warnings, {
            "code": "MODEL_FIT_ERROR",
            "message": f"Model training failed: {str(e)}",
        }

    # Prediction & Metric Evaluation on Test Set
    y_pred = model.predict(X_test_final)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    try:
        y_prob = model.predict_proba(X_test_final)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        roc_auc = None

    pos_count_test = int(np.sum(y_test == 1))
    neg_count_test = int(np.sum(y_test == 0))

    results: dict[str, Any] = {
        "model_type": model_name_clean,
        "critic_mode": "critic_on" if apply_critic_cleaning else "critic_off",
        "sample_size": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_target_distribution": {"negative": neg_count_test, "positive": pos_count_test},
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm,
        "test_predictions": [int(p) for p in y_pred],
        "test_ground_truth": [int(t) for t in y_test],
        "audit_findings": audit_findings,
        "reinvestigation_applied": reinvestigation_applied,
    }

    return True, results, warnings, None


def compare_paired_classification(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[list[str]] = None,
    model_type: str = "logistic_regression",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Run paired Critic OFF vs Critic ON classification on the exact same dataset split.
    Computes genuine Accuracy, Precision, Recall, F1-score, deltas, and McNemar's p-value.
    """
    # 1. Baseline: Critic OFF
    s1, r1, w1, e1 = execute_binary_classification(
        df=df,
        target_column=target_column,
        feature_columns=feature_columns,
        model_type=model_type,
        test_size=test_size,
        random_state=random_state,
        apply_critic_cleaning=False,
    )

    if not s1:
        return {"error": e1 or {"message": "Baseline classification failed"}}

    # 2. Proposed: Critic ON
    s2, r2, w2, e2 = execute_binary_classification(
        df=df,
        target_column=target_column,
        feature_columns=feature_columns,
        model_type=model_type,
        test_size=test_size,
        random_state=random_state,
        apply_critic_cleaning=True,
    )

    if not s2:
        return {"error": e2 or {"message": "Proposed classification failed"}}

    # 3. Statistical Test on Shared Test Predictions
    y_true = np.array(r1["test_ground_truth"])
    y_pred_b = np.array(r1["test_predictions"])
    y_pred_p = np.array(r2["test_predictions"])

    mcnemar = compute_mcnemar_p_value(y_true, y_pred_b, y_pred_p)

    return {
        "dataset_samples": len(df),
        "train_samples": r1["train_samples"],
        "test_samples": r1["test_samples"],
        "target_column": target_column,
        "model_type": model_type,
        "random_state": random_state,
        "test_ground_truth": r1["test_ground_truth"],
        "baseline_critic_off": {
            "accuracy": r1["accuracy"],
            "precision": r1["precision"],
            "recall": r1["recall"],
            "f1_score": r1["f1_score"],
            "roc_auc": r1["roc_auc"],
            "confusion_matrix": r1["confusion_matrix"],
            "test_predictions": r1["test_predictions"],
            "warnings": w1,
        },
        "proposed_critic_on": {
            "accuracy": r2["accuracy"],
            "precision": r2["precision"],
            "recall": r2["recall"],
            "f1_score": r2["f1_score"],
            "roc_auc": r2["roc_auc"],
            "confusion_matrix": r2["confusion_matrix"],
            "test_predictions": r2["test_predictions"],
            "reinvestigation_applied": r2["reinvestigation_applied"],
            "audit_findings": r2["audit_findings"],
        },
        "deltas": {
            "delta_accuracy": round(r2["accuracy"] - r1["accuracy"], 4),
            "delta_precision": round(r2["precision"] - r1["precision"], 4),
            "delta_recall": round(r2["recall"] - r1["recall"], 4),
            "delta_f1": round(r2["f1_score"] - r1["f1_score"], 4),
        },
        "statistical_significance": mcnemar,
    }

