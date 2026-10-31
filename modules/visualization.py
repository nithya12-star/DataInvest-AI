"""
modules/visualization.py
========================
DataInvest AI — Phase 1: Reusable Visualization Helpers

All functions return Plotly figures that can be rendered in Streamlit with
``st.plotly_chart(fig, use_container_width=True)``.

Design principles:
- Functions accept a DataFrame (or Series) and return a figure only.
- No Streamlit calls inside this module — visualisation is decoupled from UI.
- Later phases can reuse or extend these helpers for report generation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (consistent with the DataInvest AI brand)
# ──────────────────────────────────────────────────────────────────────────────

_PRIMARY_COLOUR = "#4F8EF7"
_ACCENT_COLOUR = "#F7944F"
_DANGER_COLOUR = "#E05252"
_SUCCESS_COLOUR = "#52C41A"
_NEUTRAL_COLOUR = "#A0AEC0"

_PLOTLY_TEMPLATE = "plotly_white"


# ──────────────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────────────


def plot_missing_bar(missing_stats: dict) -> Optional[go.Figure]:
    """
    Horizontal bar chart of per-column missing-value percentages.

    Parameters
    ----------
    missing_stats : dict
        The ``'missing'`` key from ``profiler.profile_dataset()``.

    Returns
    -------
    plotly Figure or None if every column is complete.
    """
    per_col = missing_stats.get("per_column", {})
    if not per_col:
        return None

    cols = list(per_col.keys())
    pcts = [per_col[c]["missing_pct"] for c in cols]

    # Only draw columns that actually have missing values for a clean chart.
    data = [(c, p) for c, p in zip(cols, pcts) if p > 0]
    if not data:
        return None

    data.sort(key=lambda x: x[1], reverse=True)
    cols_filtered, pcts_filtered = zip(*data)

    colours = [
        _DANGER_COLOUR if p >= 50 else _ACCENT_COLOUR if p >= 20 else _PRIMARY_COLOUR
        for p in pcts_filtered
    ]

    fig = go.Figure(
        go.Bar(
            x=list(pcts_filtered),
            y=list(cols_filtered),
            orientation="h",
            marker_color=colours,
            text=[f"{p:.1f}%" for p in pcts_filtered],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Missing Values by Column (%)",
        xaxis_title="Missing (%)",
        yaxis_title="Column",
        template=_PLOTLY_TEMPLATE,
        margin=dict(l=20, r=60, t=50, b=20),
        height=max(250, 40 * len(cols_filtered)),
        yaxis=dict(autorange="reversed"),
    )
    fig.update_xaxes(range=[0, min(max(pcts_filtered) * 1.25, 100)])
    return fig


def plot_numerical_distribution(df: pd.DataFrame, col: str) -> Optional[go.Figure]:
    """
    Combined histogram + box plot for a single numerical column.

    Parameters
    ----------
    df  : pd.DataFrame
    col : str — must be a numerical column in *df*.

    Returns
    -------
    plotly Figure or None if the column has no valid data.
    """
    series = df[col].dropna()
    if len(series) == 0:
        return None

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=series,
            marker_color=_PRIMARY_COLOUR,
            opacity=0.80,
            name="Distribution",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Box plot
    fig.add_trace(
        go.Box(
            x=series,
            marker_color=_ACCENT_COLOUR,
            line_color=_ACCENT_COLOUR,
            name=col,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"Distribution — {col}",
        template=_PLOTLY_TEMPLATE,
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
    )
    return fig


def plot_correlation_matrix(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Annotated Pearson correlation heatmap for all numerical columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    plotly Figure or None if fewer than 2 numerical columns exist.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None

    corr = num_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Matrix (Numerical Columns)",
        template=_PLOTLY_TEMPLATE,
        aspect="auto",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        coloraxis_colorbar=dict(title="r"),
        height=max(350, 50 * num_df.shape[1]),
    )
    return fig


def plot_column_type_donut(column_types: dict[str, str]) -> go.Figure:
    """
    Donut chart showing the proportion of numerical / categorical / datetime
    columns.

    Parameters
    ----------
    column_types : dict
        The ``'column_types'`` key from ``profiler.profile_dataset()``.

    Returns
    -------
    plotly Figure
    """
    from collections import Counter

    counts = Counter(column_types.values())
    labels = list(counts.keys())
    values = list(counts.values())

    colour_map = {
        "numerical": _PRIMARY_COLOUR,
        "categorical": _ACCENT_COLOUR,
        "datetime": _SUCCESS_COLOUR,
    }
    colours = [colour_map.get(lbl, _NEUTRAL_COLOUR) for lbl in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker_colors=colours,
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        title="Column Types",
        template=_PLOTLY_TEMPLATE,
        margin=dict(l=10, r=10, t=50, b=10),
        height=300,
        showlegend=False,
    )
    return fig


def plot_outlier_summary(outlier_stats: dict) -> Optional[go.Figure]:
    """
    Bar chart of potential-outlier percentages per numerical column.

    Parameters
    ----------
    outlier_stats : dict
        The ``'outliers'`` key from ``profiler.profile_dataset()``.

    Returns
    -------
    plotly Figure or None if no outlier data is available.
    """
    data = [
        (col, info["outlier_pct"])
        for col, info in outlier_stats.items()
        if info.get("outlier_pct") is not None and info["outlier_pct"] > 0
    ]
    if not data:
        return None

    data.sort(key=lambda x: x[1], reverse=True)
    cols, pcts = zip(*data)

    colours = [
        _DANGER_COLOUR if p >= 10 else _ACCENT_COLOUR if p >= 5 else _PRIMARY_COLOUR
        for p in pcts
    ]

    fig = go.Figure(
        go.Bar(
            x=list(cols),
            y=list(pcts),
            marker_color=colours,
            text=[f"{p:.1f}%" for p in pcts],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Potential Outliers by Column (% of non-null values)",
        xaxis_title="Column",
        yaxis_title="Potential Outlier (%)",
        template=_PLOTLY_TEMPLATE,
        margin=dict(l=20, r=20, t=50, b=40),
        height=320,
    )
    return fig
