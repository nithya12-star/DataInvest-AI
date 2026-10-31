"""
backend/app/schemas/analysis_schema.py
=======================================
Pydantic v2 schemas for DataInvest AI — Phase 3 Analysis Agent.
Provides strongly typed data contracts for statistical execution, factual results,
and future Critic Agent verification.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
from app.schemas.planner_schema import InvestigationTask
from app.schemas.visualization_schema import VisualEvidence


class AnalysisError(BaseModel):
    """Structured error descriptor when an analysis cannot be completed."""
    code: str = Field(description="Standardized error code (e.g. 'CONSTANT_VARIABLE', 'INSUFFICIENT_OBSERVATIONS')")
    message: str = Field(description="Human-readable explanation of why the test could not be executed")


class AnalysisResult(BaseModel):
    """
    Complete, structured result of executing a single investigation task.
    Contains purely factual, Python-calculated statistical metrics.
    No fabricated findings or conclusions.
    """
    task_id: str = Field(description="Task identifier matching the Planner task")
    status: Literal["success", "failed"] = Field(description="Execution outcome: 'success' or 'failed'")
    analysis_type: str = Field(description="Analysis category, e.g. 'numerical_vs_numerical'")
    method: str = Field(description="Actual statistical method executed, e.g. 'Pearson correlation'")
    columns: list[str] = Field(description="Columns evaluated in this analysis")
    sample_size: int = Field(default=0, description="Effective sample size (pairwise or group valid count)")
    results: dict[str, Any] = Field(default_factory=dict, description="Dictionary of factual statistical metrics")
    warnings: list[str] = Field(default_factory=list, description="Data quality or assumption warnings generated during execution")
    error: Optional[AnalysisError] = Field(default=None, description="Error details if status is 'failed'")
    visual_evidence: Optional[VisualEvidence] = Field(default=None, description="Dynamic visual evidence associated with this finding")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Execution provenance (library, version, timestamps)")


class SingleAnalysisRequest(BaseModel):
    """Request payload to execute a single investigation task."""
    dataset_id: Optional[str] = Field(default=None, description="Dataset identifier or filename")
    task: InvestigationTask = Field(description="Investigation task to execute")


class PlanAnalysisRequest(BaseModel):
    """Request payload to execute an entire list of investigation tasks."""
    dataset_id: Optional[str] = Field(default=None, description="Dataset identifier or filename")
    tasks: list[InvestigationTask] = Field(description="List of planned investigation tasks to execute")


class PlanAnalysisResponse(BaseModel):
    """Aggregate response returned when executing an entire analysis plan."""
    dataset_id: str = Field(description="Identifier of the analyzed dataset")
    total_tasks: int = Field(description="Total number of tasks dispatched")
    successful_tasks: int = Field(description="Number of successfully computed tasks")
    failed_tasks: int = Field(description="Number of tasks that encountered valid failure boundaries")
    results: list[AnalysisResult] = Field(description="Ordered list of individual task results")
