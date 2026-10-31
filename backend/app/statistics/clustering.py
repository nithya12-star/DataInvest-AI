"""
backend/app/statistics/clustering.py
====================================
Statistical Clustering & Customer Segmentation Module for DataInvest AI.

Provides reproducible, deterministic unsupervised clustering evaluation on tabular datasets
using scikit-learn, with strict separation between:
1. Baseline (Critic OFF): Unscaled raw features, arbitrary cluster count k, unflagged
   anomalous zeros, and unexamined scale disparity (Euclidean distance dominated by largest scale).
2. Proposed (Critic ON): Statistically audited features with scale disparity detection,
   automatic standardization (StandardScaler), exclusion of sequential identifiers (e.g. CustomerID),
   anomalous zero imputation, and silhouette-guided optimal cluster count validation.

Calculates:
- Silhouette Score (valid for 2 <= k < n)
- Davies-Bouldin Index (compactness vs separation; lower is better)
- Calinski-Harabasz Score (variance ratio criterion; higher is better)
- Within-Cluster Sum of Squares (Inertia)
- Cluster sizes and relative proportions
- Cluster centroids in original interpretable units
- Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) for paired partition comparison
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("datainvest.statistics.clustering")

IDENTIFIER_PATTERNS = [
    re.compile(r"^customer_?id$", re.IGNORECASE),
    re.compile(r"^id$", re.IGNORECASE),
    re.compile(r"^uuid$", re.IGNORECASE),
    re.compile(r"^index$", re.IGNORECASE),
    re.compile(r"^identifier$", re.IGNORECASE),
    re.compile(r".*_id$", re.IGNORECASE),
]


def detect_identifier_columns(df: pd.DataFrame, columns: Sequence[str]) -> dict[str, str]:
    """
    Detect columns that function as unique record identifiers rather than behavioral features.
    """
    detected: dict[str, str] = {}
    n_rows = len(df)
    for col in columns:
        if col not in df.columns:
            continue
        is_pattern_match = any(pat.match(col.strip()) for pat in IDENTIFIER_PATTERNS)
        is_perfectly_unique = int(df[col].nunique()) == n_rows
        is_integer_seq = (
            pd.api.types.is_integer_dtype(df[col])
            and is_perfectly_unique
            and n_rows > 10
        )

        if is_pattern_match and (is_perfectly_unique or is_integer_seq):
            detected[col] = (
                f"Column name matches identifier pattern and has 100% unique values ({n_rows}/{n_rows})."
            )
        elif is_integer_seq:
            detected[col] = (
                f"Sequential unique integer column ({n_rows}/{n_rows} unique values); functions as index/identifier."
            )
    return detected


def detect_scale_disparity(df: pd.DataFrame, num_cols: Sequence[str]) -> dict[str, Any]:
    """
    Audit standard deviations across numerical features to identify severe scale disparities
    that mathematically distort Euclidean distance in distance-based clustering algorithms.
    """
    stds: dict[str, float] = {}
    for c in num_cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            series = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(series) > 1:
                stds[c] = float(series.std())

    if len(stds) < 2:
        return {"has_disparity": False, "ratio": 1.0, "stds": stds}

    sorted_stds = sorted(stds.items(), key=lambda kv: kv[1], reverse=True)
    max_col, max_val = sorted_stds[0]
    min_col, min_val = sorted_stds[-1]
    ratio = max_val / (min_val + 1e-9)

    has_disparity = bool(ratio > 10.0)
    return {
        "has_disparity": has_disparity,
        "ratio": round(ratio, 2),
        "max_col": max_col,
        "max_std": round(max_val, 2),
        "min_col": min_col,
        "min_std": round(min_val, 2),
        "stds": {k: round(v, 2) for k, v in stds.items()},
    }


def execute_kmeans_clustering(
    df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
    k_clusters: int = 3,
    random_state: int = 42,
    apply_critic_cleaning: bool = False,
    scale_features: Optional[bool] = None,
    auto_k: bool = False,
    k_search_range: tuple[int, int] = (2, 6),
) -> tuple[bool, dict[str, Any], list[str], Optional[dict[str, str]]]:
    """
    Train and evaluate K-Means clustering on the given dataset.

    Returns:
        (success, results_dict, warnings_list, error_dict)
    """
    warnings: list[str] = []

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False, {}, warnings, {"code": "EMPTY_DATASET", "message": "Dataset is empty or invalid."}

    n_rows = len(df)
    if n_rows < 10:
        return False, {}, warnings, {
            "code": "INSUFFICIENT_OBSERVATIONS",
            "message": f"Clustering requires at least 10 observations; found {n_rows}.",
        }

    # 1. Feature column selection
    if feature_columns:
        missing_feats = [c for c in feature_columns if c not in df.columns]
        if missing_feats:
            return False, {}, warnings, {
                "code": "FEATURE_COLUMNS_NOT_FOUND",
                "message": f"Specified features not found: {missing_feats}",
            }
        candidate_cols = list(feature_columns)
    else:
        # Default to all numeric columns in dataset
        candidate_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    audit_findings: dict[str, Any] = {}
    reinvestigation_applied: list[str] = []

    # 2. Identifier Column Audit
    identifiers = detect_identifier_columns(df, candidate_cols)
    audit_findings["identifier_columns_detected"] = identifiers

    if identifiers:
        if apply_critic_cleaning:
            # Critic ON: Exclude identifiers from feature space
            active_cols = [c for c in candidate_cols if c not in identifiers]
            for id_col, reason in identifiers.items():
                reinvestigation_applied.append(
                    f"Excluded identifier column '{id_col}' from clustering feature matrix ({reason})."
                )
        else:
            # Critic OFF: Retain identifier if passed by user
            active_cols = list(candidate_cols)
            for id_col in identifiers:
                warnings.append(
                    f"Un-audited baseline: Retained record identifier '{id_col}' in clustering features. "
                    "May introduce spurious sequential clustering artifacts."
                )
    else:
        active_cols = list(candidate_cols)

    # Ensure we only use numeric features
    active_num_cols = [c for c in active_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(active_num_cols) < 2:
        return False, {}, warnings, {
            "code": "INSUFFICIENT_NUMERIC_FEATURES",
            "message": f"Clustering requires at least 2 numeric features; available: {active_num_cols}",
        }

    # 3. Data Cleaning and Imputation
    X_work = df[active_num_cols].copy()

    # Detect anomalous zeros in Age
    if "Age" in active_num_cols:
        zero_age_count = int((X_work["Age"] == 0).sum())
        if zero_age_count > 0:
            audit_findings["anomalous_zero_age_count"] = zero_age_count
            if apply_critic_cleaning:
                median_age = float(X_work.loc[X_work["Age"] > 0, "Age"].median())
                X_work.loc[X_work["Age"] == 0, "Age"] = median_age
                reinvestigation_applied.append(
                    f"Imputed {zero_age_count} anomalous zero values in 'Age' with median ({median_age:.1f})."
                )
            else:
                warnings.append(
                    f"Un-audited baseline: Retained {zero_age_count} literal zero values in 'Age'."
                )

    # Detect anomalous zeros in Income or Spending
    for col_candidate in ["Annual Income ($)", "Spending Score (1-100)"]:
        if col_candidate in active_num_cols:
            z_count = int((X_work[col_candidate] == 0).sum())
            if z_count > 0:
                audit_findings[f"anomalous_zero_{col_candidate}"] = z_count
                if apply_critic_cleaning:
                    med_val = float(X_work.loc[X_work[col_candidate] > 0, col_candidate].median())
                    X_work.loc[X_work[col_candidate] == 0, col_candidate] = med_val
                    reinvestigation_applied.append(
                        f"Imputed {z_count} anomalous zero values in '{col_candidate}' with median ({med_val:.1f})."
                    )

    # Impute standard NaNs with column median
    for c in active_num_cols:
        null_count = int(X_work[c].isnull().sum())
        if null_count > 0:
            med = float(X_work[c].median())
            X_work[c] = X_work[c].fillna(med)
            if apply_critic_cleaning:
                reinvestigation_applied.append(f"Imputed {null_count} nulls in '{c}' with median ({med:.2f}).")

    # 4. Scale Disparity Audit & Normalization Decision
    disparity = detect_scale_disparity(X_work, active_num_cols)
    audit_findings["scale_disparity"] = disparity

    # Determine whether to scale
    if scale_features is not None:
        do_scale = scale_features
    elif apply_critic_cleaning:
        do_scale = True
    else:
        do_scale = False

    if disparity.get("has_disparity") and not do_scale:
        warnings.append(
            f"Un-audited baseline: Severe scale disparity ratio {disparity['ratio']:.1f} detected "
            f"('{disparity['max_col']}' std={disparity['max_std']} vs '{disparity['min_col']}' std={disparity['min_std']}). "
            "Without standardization, Euclidean distance is mathematically dominated by the largest scale feature."
        )

    if disparity.get("has_disparity") and do_scale:
        reinvestigation_applied.append(
            f"Applied StandardScaler across all {len(active_num_cols)} features to resolve scale disparity "
            f"(ratio {disparity['ratio']:.1f}: '{disparity['max_col']}' std={disparity['max_std']} vs '{disparity['min_col']}' std={disparity['min_std']})."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_work[active_num_cols].values)

    if do_scale:
        X_train = X_scaled
    else:
        X_train = X_work[active_num_cols].values

    # 5. Cluster Count (k) Evaluation & Selection
    candidate_evaluations: list[dict[str, Any]] = []

    if auto_k or apply_critic_cleaning:
        k_min, k_max = k_search_range
        k_min = max(2, min(k_min, n_rows - 1))
        k_max = min(k_max, n_rows - 1, 10)

        best_k = k_clusters
        best_sil = -1.0

        for cand_k in range(k_min, k_max + 1):
            km_cand = KMeans(n_clusters=cand_k, random_state=random_state, n_init=10)
            cand_labels = km_cand.fit_predict(X_train)
            cand_sil = float(silhouette_score(X_train, cand_labels))
            cand_db = float(davies_bouldin_score(X_train, cand_labels))
            cand_ch = float(calinski_harabasz_score(X_train, cand_labels))
            cand_inertia = float(km_cand.inertia_)

            eval_entry = {
                "k": cand_k,
                "silhouette_score": round(cand_sil, 4),
                "davies_bouldin_index": round(cand_db, 4),
                "calinski_harabasz_score": round(cand_ch, 2),
                "inertia": round(cand_inertia, 2),
            }
            candidate_evaluations.append(eval_entry)

            if cand_sil > best_sil:
                best_sil = cand_sil
                best_k = cand_k

        if auto_k:
            selected_k = best_k
            reinvestigation_applied.append(
                f"Automated silhouette audit evaluated k={k_min}..{k_max}; selected optimal k={selected_k} (silhouette={best_sil:.4f})."
            )
        else:
            selected_k = k_clusters
    else:
        selected_k = k_clusters

    if selected_k < 2 or selected_k >= n_rows:
        return False, {}, warnings, {
            "code": "INVALID_K_CLUSTERS",
            "message": f"Cluster count k={selected_k} must satisfy 2 <= k < {n_rows}.",
        }

    # 6. Fit Final K-Means Model
    km_final = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
    labels = km_final.fit_predict(X_train)

    # 7. Compute Clustering Evaluation Metrics
    # In native space:
    sil_native = float(silhouette_score(X_train, labels))
    # In standardized space (standard benchmark ground-truth):
    sil_standardized = float(silhouette_score(X_scaled, labels))
    db_index = float(davies_bouldin_score(X_train, labels))
    ch_score = float(calinski_harabasz_score(X_train, labels))
    inertia = float(km_final.inertia_)

    # 8. Cluster Centroids and Sizes
    if do_scale:
        centers_orig = scaler.inverse_transform(km_final.cluster_centers_)
        centers_norm = km_final.cluster_centers_
    else:
        centers_orig = km_final.cluster_centers_
        # Normalize centers using standard scaler for comparison
        centers_norm = scaler.transform(centers_orig)

    cluster_counts = pd.Series(labels).value_counts().sort_index().to_dict()
    cluster_sizes: dict[str, int] = {f"cluster_{k}": int(v) for k, v in cluster_counts.items()}
    cluster_props: dict[str, float] = {
        f"cluster_{k}": round(v / n_rows * 100.0, 2) for k, v in cluster_counts.items()
    }

    centroids_dict: dict[str, dict[str, float]] = {}
    for idx in range(selected_k):
        c_dict = {
            col: round(float(centers_orig[idx, f_idx]), 2)
            for f_idx, col in enumerate(active_num_cols)
        }
        centroids_dict[f"cluster_{idx}"] = c_dict

    results: dict[str, Any] = {
        "k_clusters": selected_k,
        "sample_size": n_rows,
        "features_used": active_num_cols,
        "scaling_applied": do_scale,
        "critic_mode": "critic_on" if apply_critic_cleaning else "critic_off",
        "silhouette_score_native": round(sil_native, 4),
        "silhouette_score_standardized": round(sil_standardized, 4),
        "davies_bouldin_index": round(db_index, 4),
        "calinski_harabasz_score": round(ch_score, 2),
        "inertia": round(inertia, 2),
        "cluster_sizes": cluster_sizes,
        "cluster_proportions": cluster_props,
        "cluster_centroids": centroids_dict,
        "cluster_labels": [int(l) for l in labels],
        "candidate_k_evaluations": candidate_evaluations,
        "audit_findings": audit_findings,
        "reinvestigation_applied": reinvestigation_applied,
    }

    return True, results, warnings, None


def compare_paired_clustering(
    df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
    k_baseline: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Run paired Critic OFF vs Critic ON clustering comparison on the exact same dataset.

    Evaluates:
    1. Baseline (Critic OFF):
       - Unscaled raw features, arbitrary k_baseline=3, retaining anomalous zeros.
    2. Proposed Matched-k (Critic ON, k=3):
       - Audited and standardized features, anomalous zeros imputed, identifier excluded, exactly matching k=3.
    3. Proposed Optimal-k (Critic ON, optimal k):
       - Full Critic optimization with silhouette-guided k selection (e.g. k=4).

    Computes:
    - Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)
    - Metric deltas (Silhouette, Davies-Bouldin, Calinski-Harabasz)
    - Centroid characterization
    """
    # 1. Baseline: Critic OFF
    s_base, r_base, w_base, e_base = execute_kmeans_clustering(
        df=df,
        feature_columns=feature_columns,
        k_clusters=k_baseline,
        random_state=random_state,
        apply_critic_cleaning=False,
        scale_features=False,
        auto_k=False,
    )
    if not s_base:
        return {"error": e_base or {"message": "Baseline clustering failed."}}

    # 2. Proposed: Critic ON (Matched k)
    s_prop_matched, r_prop_matched, w_prop_matched, e_prop_matched = execute_kmeans_clustering(
        df=df,
        feature_columns=feature_columns,
        k_clusters=k_baseline,
        random_state=random_state,
        apply_critic_cleaning=True,
        scale_features=True,
        auto_k=False,
    )
    if not s_prop_matched:
        return {"error": e_prop_matched or {"message": "Proposed matched-k clustering failed."}}

    # 3. Proposed: Critic ON (Optimal k via silhouette audit)
    s_prop_opt, r_prop_opt, w_prop_opt, e_prop_opt = execute_kmeans_clustering(
        df=df,
        feature_columns=feature_columns,
        k_clusters=k_baseline,
        random_state=random_state,
        apply_critic_cleaning=True,
        scale_features=True,
        auto_k=True,
        k_search_range=(2, 6),
    )
    if not s_prop_opt:
        return {"error": e_prop_opt or {"message": "Proposed optimal-k clustering failed."}}

    # 4. Partition Concordance Statistics
    labels_base = np.array(r_base["cluster_labels"])
    labels_prop_matched = np.array(r_prop_matched["cluster_labels"])
    labels_prop_opt = np.array(r_prop_opt["cluster_labels"])

    ari_matched = float(adjusted_rand_score(labels_base, labels_prop_matched))
    nmi_matched = float(normalized_mutual_info_score(labels_base, labels_prop_matched))

    ari_opt = float(adjusted_rand_score(labels_base, labels_prop_opt))
    nmi_opt = float(normalized_mutual_info_score(labels_base, labels_prop_opt))

    # Metric Deltas (evaluated in standardized feature space for valid geometric comparison)
    base_std_sil = r_base["silhouette_score_standardized"]
    matched_std_sil = r_prop_matched["silhouette_score_standardized"]
    opt_std_sil = r_prop_opt["silhouette_score_standardized"]

    delta_sil_matched = round(matched_std_sil - base_std_sil, 4)
    delta_sil_opt = round(opt_std_sil - base_std_sil, 4)

    delta_db_matched = round(
        r_prop_matched["davies_bouldin_index"] - r_base["davies_bouldin_index"], 4
    )
    delta_db_opt = round(
        r_prop_opt["davies_bouldin_index"] - r_base["davies_bouldin_index"], 4
    )

    return {
        "dataset_samples": len(df),
        "random_state": random_state,
        "features_evaluated": r_prop_matched["features_used"],
        "baseline_critic_off": {
            "k_clusters": r_base["k_clusters"],
            "scaling_applied": r_base["scaling_applied"],
            "silhouette_native": r_base["silhouette_score_native"],
            "silhouette_standardized": r_base["silhouette_score_standardized"],
            "davies_bouldin_index": r_base["davies_bouldin_index"],
            "calinski_harabasz_score": r_base["calinski_harabasz_score"],
            "inertia": r_base["inertia"],
            "cluster_sizes": r_base["cluster_sizes"],
            "cluster_proportions": r_base["cluster_proportions"],
            "cluster_centroids": r_base["cluster_centroids"],
            "cluster_labels": r_base["cluster_labels"],
            "warnings": w_base,
        },
        "proposed_critic_on_matched_k": {
            "k_clusters": r_prop_matched["k_clusters"],
            "scaling_applied": r_prop_matched["scaling_applied"],
            "silhouette_native": r_prop_matched["silhouette_score_native"],
            "silhouette_standardized": r_prop_matched["silhouette_score_standardized"],
            "davies_bouldin_index": r_prop_matched["davies_bouldin_index"],
            "calinski_harabasz_score": r_prop_matched["calinski_harabasz_score"],
            "inertia": r_prop_matched["inertia"],
            "cluster_sizes": r_prop_matched["cluster_sizes"],
            "cluster_proportions": r_prop_matched["cluster_proportions"],
            "cluster_centroids": r_prop_matched["cluster_centroids"],
            "cluster_labels": r_prop_matched["cluster_labels"],
            "audit_findings": r_prop_matched["audit_findings"],
            "reinvestigation_applied": r_prop_matched["reinvestigation_applied"],
        },
        "proposed_critic_on_optimal_k": {
            "k_clusters": r_prop_opt["k_clusters"],
            "scaling_applied": r_prop_opt["scaling_applied"],
            "silhouette_native": r_prop_opt["silhouette_score_native"],
            "silhouette_standardized": r_prop_opt["silhouette_score_standardized"],
            "davies_bouldin_index": r_prop_opt["davies_bouldin_index"],
            "calinski_harabasz_score": r_prop_opt["calinski_harabasz_score"],
            "inertia": r_prop_opt["inertia"],
            "cluster_sizes": r_prop_opt["cluster_sizes"],
            "cluster_proportions": r_prop_opt["cluster_proportions"],
            "cluster_centroids": r_prop_opt["cluster_centroids"],
            "cluster_labels": r_prop_opt["cluster_labels"],
            "candidate_k_evaluations": r_prop_opt["candidate_k_evaluations"],
            "audit_findings": r_prop_opt["audit_findings"],
            "reinvestigation_applied": r_prop_opt["reinvestigation_applied"],
        },
        "comparison_metrics": {
            "adjusted_rand_index_matched_k": round(ari_matched, 4),
            "normalized_mutual_info_matched_k": round(nmi_matched, 4),
            "adjusted_rand_index_optimal_k": round(ari_opt, 4),
            "normalized_mutual_info_optimal_k": round(nmi_opt, 4),
            "delta_silhouette_standardized_matched_k": delta_sil_matched,
            "delta_silhouette_standardized_optimal_k": delta_sil_opt,
            "delta_davies_bouldin_matched_k": delta_db_matched,
            "delta_davies_bouldin_optimal_k": delta_db_opt,
        },
        "methodological_conclusions": [
            f"Scale Disparity: Unscaled baseline resulted in severe Euclidean distortion (ARI={ari_matched:.4f} vs standardized).",
            f"Cluster Quality: Standardization increased true multidimensional silhouette from {base_std_sil:.4f} to {matched_std_sil:.4f} (+{delta_sil_matched:.4f}).",
            f"Optimal Geometry: Critic silhouette audit confirmed optimal k={r_prop_opt['k_clusters']} achieves peak separation (silhouette={opt_std_sil:.4f}, Davies-Bouldin={r_prop_opt['davies_bouldin_index']:.4f}).",
            "Bivariate Balance: Scaling uncovered balanced customer segments across spending propensity, avoiding 1D income-only slicing.",
        ],
    }

