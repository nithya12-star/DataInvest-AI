"""
backend/app/schemas/visualization_schema.py
============================================
Pydantic v2 schemas for DataInvest AI — Phase 4: Dynamic Graphs & Visual Evidence.
Provides strongly typed contracts for graph metadata, programmatic visual diagnostics,
Plotly figure specifications, and scientific visual evidence tracking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


FigureType = Literal[
    "scatter",
    "box",
    "multi_box",
    "grouped_bar",
    "heatmap",
    "distribution_histogram",
    "unknown",
]


class VisualDiagnostics(BaseModel):
    """
    Defensible, purely programmatic visual diagnostics computed from the actual data.
    Does NOT claim or simulate visual image inspection by an LLM.
    """
    outlier_points_count: int = Field(default=0, description="Count of extreme data points outside IQR whiskers or bivariate 3-SD bounds")
    linearity_indicator: Optional[str] = Field(default=None, description="Heuristic linearity indicator from residual spread")
    iqr_overlap_ratio: Optional[float] = Field(default=None, description="Estimated overlap ratio between group IQRs")
    sparse_categories_count: int = Field(default=0, description="Count of categories with fewer than 5 observations")
    dominant_category_share: Optional[float] = Field(default=None, description="Percentage of observations in largest category")
    visual_notes: List[str] = Field(default_factory=list, description="Factual diagnostic observations derived programmatically")
    limitation_notice: str = Field(
        default="Programmatic visual diagnostics used. No vision-capable LLM inspection claimed.",
        description="Explicit disclosure that visual evidence is evaluated through programmatic metrics, not image perception."
    )
    vision_model_used: bool = Field(default=False, description="Explicit flag: always False unless genuine vision model is present")
    visual_inspection_claimed: bool = Field(default=False, description="Explicit flag: always False to maintain scientific honesty")


class VisualEvidence(BaseModel):
    """
    Complete visual evidence artifact associated with an analytical finding.
    Contains both the renderable Plotly specification and computed visual diagnostics.
    """
    graph_id: str = Field(description="Unique graph identifier, e.g. 'graph_t1_scatter'")
    task_id: str = Field(description="Associated analytical task identifier")
    figure_type: FigureType = Field(description="Type of chart generated")
    title: str = Field(description="Descriptive title of the visualization")
    x_label: str = Field(description="Label for horizontal axis")
    y_label: str = Field(description="Label for vertical axis")
    columns: List[str] = Field(default_factory=list, description="Dataset columns depicted in this figure")
    method: str = Field(description="Associated statistical method")
    data_summary: Dict[str, Any] = Field(default_factory=dict, description="Numerical summary of plotted points, medians, or counts")
    figure_spec: Optional[Dict[str, Any]] = Field(default=None, description="Plotly-compatible dictionary representation")
    visual_diagnostics: VisualDiagnostics = Field(default_factory=VisualDiagnostics, description="Programmatic visual diagnostics")
    limitation_notice: str = Field(
        default="Programmatic visual diagnostics used. No vision-capable LLM inspection claimed.",
        description="Explicit limitation notice"
    )
