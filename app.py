"""
app.py
======
DataInvest AI — Phase 1 Entry Point

Run with:
    streamlit run app.py

This file contains only Streamlit presentation logic.
All statistical computation lives in modules/profiler.py.
All chart construction lives in modules/visualization.py.
"""

from __future__ import annotations

import io
import traceback
from typing import Optional

import pandas as pd
import streamlit as st

from modules import profiler as pf
from modules import visualization as viz

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration — must be the first Streamlit call
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DataInvest AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# Global CSS — professional styling
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* ── Hero header ── */
    .hero-header {
        background: linear-gradient(135deg, #1a1f36 0%, #2d3561 60%, #1a1f36 100%);
        border-radius: 16px;
        padding: 2.5rem 2.8rem 2rem 2.8rem;
        margin-bottom: 2rem;
        color: white;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0 0 0.3rem 0;
        background: linear-gradient(90deg, #ffffff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #a5b4fc;
        margin: 0;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(165,180,252,0.15);
        border: 1px solid rgba(165,180,252,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        color: #a5b4fc;
        margin-bottom: 0.8rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ── Section headings ── */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1a1f36;
        border-left: 4px solid #4F8EF7;
        padding-left: 0.7rem;
        margin: 1.8rem 0 1rem 0;
    }

    /* ── Metric card overrides ── */
    [data-testid="stMetric"] {
        background: #f8faff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        color: #64748b !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        color: #1a1f36 !important;
    }

    /* ── Upload zone ── */
    .upload-zone {
        background: #f8faff;
        border: 2px dashed #4F8EF7;
        border-radius: 16px;
        padding: 2rem 2rem 1.5rem 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .upload-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .upload-hint { color: #64748b; font-size: 0.9rem; margin-top: 0.5rem; }

    /* ── Info / warning boxes ── */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        color: #1e40af;
        font-size: 0.9rem;
        margin: 0.8rem 0;
    }
    .warn-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        color: #92400e;
        font-size: 0.9rem;
        margin: 0.8rem 0;
    }
    .success-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        color: #14532d;
        font-size: 0.9rem;
        margin: 0.8rem 0;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }

    /* ── Divider ── */
    hr { border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _section(title: str, icon: str = "") -> None:
    prefix = f"{icon} " if icon else ""
    st.markdown(f'<div class="section-header">{prefix}{title}</div>', unsafe_allow_html=True)


def _load_csv(uploaded_file) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely load a CSV from a Streamlit UploadedFile.

    Returns (DataFrame, None) on success or (None, error_message) on failure.
    Never exposes a Python stack trace to the caller.
    """
    try:
        content = uploaded_file.read()
        if len(content) == 0:
            return None, "The uploaded file is empty. Please upload a non-empty CSV file."

        # Try UTF-8 first, fall back to latin-1.
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                if df.empty:
                    return None, (
                        "The CSV file was read successfully but contains no rows. "
                        "Please upload a file with at least one data row."
                    )
                return df, None
            except UnicodeDecodeError:
                continue
            except pd.errors.EmptyDataError:
                return None, "The file appears to be empty or contains only headers with no data rows."
            except pd.errors.ParserError as exc:
                return None, f"Could not parse the CSV file: {exc}"

        return None, (
            "Unable to decode the file. Please save it as UTF-8 or Latin-1 encoded CSV and try again."
        )

    except Exception:
        # Log internally but never show the raw traceback to the user.
        return None, (
            "An unexpected error occurred while reading the file. "
            "Please verify that it is a valid CSV file."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Rendering sections
# ──────────────────────────────────────────────────────────────────────────────


def _render_overview(profile: dict, filename: str) -> None:
    _section("Dataset Overview", "📋")
    ov = profile["overview"]
    miss = profile["missing"]
    dups = profile["duplicates"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{ov['n_rows']:,}")
    c2.metric("Columns", f"{ov['n_cols']:,}")
    c3.metric("Missing Cells", f"{miss['total_missing']:,}", f"{miss['total_missing_pct']:.1f}%")
    c4.metric("Duplicate Rows", f"{dups['n_duplicates']:,}", f"{dups['duplicate_pct']:.1f}%")

    st.markdown(
        f'<div class="info-box">📁 <strong>{filename}</strong> &nbsp;·&nbsp; '
        f"{ov['memory_usage_bytes'] / 1024:.1f} KB in memory</div>",
        unsafe_allow_html=True,
    )


def _render_column_type_overview(profile: dict) -> None:
    col_types = profile["column_types"]
    from collections import Counter

    counts = Counter(col_types.values())
    label_map = {"numerical": "🔢 Numerical", "categorical": "🔤 Categorical", "datetime": "📅 Datetime"}

    cols = st.columns(len(counts))
    for i, (kind, count) in enumerate(sorted(counts.items())):
        cols[i].metric(label_map.get(kind, kind), count)

    # Donut chart
    fig_donut = viz.plot_column_type_donut(col_types)
    if fig_donut:
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.plotly_chart(fig_donut, use_container_width=True)
        with col_right:
            _render_column_table(profile)
    else:
        _render_column_table(profile)


def _render_column_table(profile: dict) -> None:
    ov = profile["overview"]
    col_types = profile["column_types"]
    card = profile["cardinality"]["per_column"]
    miss_per_col = profile["missing"]["per_column"]

    rows = []
    for col in ov["column_names"]:
        rows.append(
            {
                "Column": col,
                "Type": col_types.get(col, "—"),
                "Dtype": ov["dtypes"].get(col, "—"),
                "Unique": card[col]["n_unique"],
                "Missing (%)": f"{miss_per_col[col]['missing_pct']:.1f}%",
                "Constant": "✓" if card[col]["is_constant"] else "",
                "High Cardinality": "✓" if card[col]["is_high_cardinality"] else "",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_data_quality(profile: dict) -> None:
    _section("Data Quality", "🔍")

    miss = profile["missing"]
    dups = profile["duplicates"]
    card = profile["cardinality"]

    # ── Missing values ──────────────────────────────────────────────────────
    st.markdown("**Missing Values**")
    if miss["total_missing"] == 0:
        st.markdown('<div class="success-box">✅ No missing values detected across the entire dataset.</div>', unsafe_allow_html=True)
    else:
        fig_miss = viz.plot_missing_bar(miss)
        if fig_miss:
            st.plotly_chart(fig_miss, use_container_width=True)

        # Table: only columns with missing values
        miss_rows = [
            {
                "Column": col,
                "Missing Count": info["missing_count"],
                "Missing (%)": f"{info['missing_pct']:.2f}%",
            }
            for col, info in miss["per_column"].items()
            if info["missing_count"] > 0
        ]
        if miss_rows:
            st.dataframe(pd.DataFrame(miss_rows), use_container_width=True, hide_index=True)

    # ── Duplicates ──────────────────────────────────────────────────────────
    st.markdown("**Duplicate Rows**")
    if dups["n_duplicates"] == 0:
        st.markdown('<div class="success-box">✅ No duplicate rows detected.</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="warn-box">⚠️ Found <strong>{dups["n_duplicates"]:,}</strong> duplicate '
            f'rows ({dups["duplicate_pct"]:.1f}% of total). Consider deduplication before analysis.</div>',
            unsafe_allow_html=True,
        )

    # ── Constant / high-cardinality columns ─────────────────────────────────
    if card["constant_columns"]:
        st.markdown("**Constant Columns** (single unique value — may not add analytical value)")
        for c in card["constant_columns"]:
            st.markdown(f"- `{c}`")

    if card["high_cardinality_columns"]:
        st.markdown("**High-Cardinality Categorical Columns** (>50% unique values — may be IDs or free text)")
        for c in card["high_cardinality_columns"]:
            st.markdown(f"- `{c}`")


def _render_numerical_stats(profile: dict) -> None:
    _section("Numerical Statistics", "📈")

    num_stats = profile["numerical_stats"]
    if not num_stats:
        st.info("No numerical columns detected in this dataset.")
        return

    rows = []
    for col, s in num_stats.items():
        rows.append(
            {
                "Column": col,
                "Valid n": s["n_valid"],
                "Mean": f"{s['mean']:.4f}" if s["mean"] is not None else "—",
                "Median": f"{s['median']:.4f}" if s["median"] is not None else "—",
                "Std Dev": f"{s['std']:.4f}" if s["std"] is not None else "—",
                "Min": f"{s['min']:.4f}" if s["min"] is not None else "—",
                "Q1": f"{s['q1']:.4f}" if s["q1"] is not None else "—",
                "Q3": f"{s['q3']:.4f}" if s["q3"] is not None else "—",
                "Max": f"{s['max']:.4f}" if s["max"] is not None else "—",
                "IQR": f"{s['iqr']:.4f}" if s["iqr"] is not None else "—",
                "Skewness": f"{s['skewness']:.4f}" if s["skewness"] is not None else "—",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Individual distribution charts ──────────────────────────────────────
    st.markdown("**Column Distributions**")
    num_cols = list(num_stats.keys())

    # Render distributions in a selectbox to keep the page manageable.
    if len(num_cols) == 1:
        selected = num_cols[0]
    else:
        selected = st.selectbox("Select column to inspect", num_cols, key="dist_select")

    if selected:
        # We need the original DataFrame — stored in session state.
        df = st.session_state.get("current_df")
        if df is not None:
            fig_dist = viz.plot_numerical_distribution(df, selected)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)

    # ── Correlation matrix ──────────────────────────────────────────────────
    df = st.session_state.get("current_df")
    if df is not None and df.select_dtypes(include="number").shape[1] >= 2:
        with st.expander("Correlation Matrix (Pearson)", expanded=False):
            fig_corr = viz.plot_correlation_matrix(df)
            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)


def _render_outliers(profile: dict) -> None:
    _section("Potential Outlier Detection", "⚠️")

    outliers = profile["outliers"]
    if not outliers:
        st.info("No numerical columns available for outlier analysis.")
        return

    st.markdown(
        '<div class="info-box">ℹ️ Outlier detection uses the <strong>1.5 × IQR rule</strong>. '
        "Values outside <em>Q1 − 1.5·IQR</em> or <em>Q3 + 1.5·IQR</em> are flagged as "
        "<strong>potential</strong> outliers. They may be legitimate extreme values, data entry "
        "errors, or genuinely anomalous observations — further investigation is recommended.</div>",
        unsafe_allow_html=True,
    )

    # Summary table
    rows = []
    for col, info in outliers.items():
        rows.append(
            {
                "Column": col,
                "Potential Outliers": info["n_potential_outliers"]
                if info["n_potential_outliers"] is not None
                else "—",
                "Outlier (%)": f"{info['outlier_pct']:.2f}%"
                if info["outlier_pct"] is not None
                else "—",
                "Lower Fence": f"{info['lower_fence']:.4f}"
                if info["lower_fence"] is not None
                else "—",
                "Upper Fence": f"{info['upper_fence']:.4f}"
                if info["upper_fence"] is not None
                else "—",
                "Note": info.get("note", ""),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Visual summary bar
    fig_out = viz.plot_outlier_summary(outliers)
    if fig_out:
        st.plotly_chart(fig_out, use_container_width=True)
    else:
        st.markdown(
            '<div class="success-box">✅ No potential outliers detected in any numerical column.</div>',
            unsafe_allow_html=True,
        )


def _render_column_info(profile: dict) -> None:
    _section("Column Information", "🗂️")
    _render_column_type_overview(profile)


def _render_preview(df: pd.DataFrame) -> None:
    _section("Data Preview", "👁️")
    n_preview = st.slider("Rows to preview", min_value=5, max_value=min(100, len(df)), value=10, step=5)
    st.dataframe(df.head(n_preview), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── Hero header ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-badge">Phase 1 — Dataset Profiler</div>
            <h1 class="hero-title">📊 DataInvest AI</h1>
            <p class="hero-subtitle">
                Automatically investigates tabular datasets — delivering deep statistical
                profiling, data-quality diagnostics, and outlier detection in seconds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Upload zone ───────────────────────────────────────────────────────────
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    st.markdown('<div class="upload-icon">📂</div>', unsafe_allow_html=True)
    st.markdown("**Upload your CSV dataset to begin analysis**")

    uploaded_file = st.file_uploader(
        label="Choose a CSV file",
        type=["csv"],
        help="Maximum file size is determined by your Streamlit server settings.",
        label_visibility="collapsed",
    )
    st.markdown('<p class="upload-hint">Only CSV files are accepted in Phase 1.</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Process upload ────────────────────────────────────────────────────────
    if uploaded_file is None:
        st.markdown(
            '<div class="info-box">👆 Upload a CSV file above to generate a comprehensive dataset profile.</div>',
            unsafe_allow_html=True,
        )
        _render_footer()
        return

    # Load CSV
    with st.spinner("Reading CSV file…"):
        df, error_msg = _load_csv(uploaded_file)

    if error_msg:
        st.error(f"❌ {error_msg}")
        _render_footer()
        return

    # Persist DataFrame in session state so rendering helpers can access it.
    st.session_state["current_df"] = df

    # Run profiler
    with st.spinner("Profiling dataset — please wait…"):
        try:
            profile = pf.profile_dataset(df)
        except Exception:
            st.error(
                "❌ An unexpected error occurred during profiling. "
                "Please verify your CSV file and try again."
            )
            _render_footer()
            return

    st.markdown(
        '<div class="success-box">✅ Profile complete — scroll down to explore your dataset.</div>',
        unsafe_allow_html=True,
    )

    # ── Render sections ───────────────────────────────────────────────────────
    _render_overview(profile, uploaded_file.name)
    st.divider()
    _render_column_info(profile)
    st.divider()
    _render_data_quality(profile)
    st.divider()
    _render_numerical_stats(profile)
    st.divider()
    _render_outliers(profile)
    st.divider()
    _render_preview(df)

    _render_footer()


def _render_footer() -> None:
    st.markdown(
        '<div class="footer">DataInvest AI · Phase 1 — CSV Upload & Dataset Profiler · '
        "All statistics computed deterministically by Python / Pandas / NumPy.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
