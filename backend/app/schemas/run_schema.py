"""
backend/app/schemas/run_schema.py
=================================
Pydantic v2 schemas for Experiment Run Repository & Historical Persistence.
Captures pipeline state across Phases 1 through 7 for saving, listing, and restoring runs.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class RunSummary(BaseModel):
    """Lightweight metadata summary for fast history listing."""
    run_id: str = Field(description="Unique experiment identifier")
    dataset_name: str = Field(description="Original dataset filename")
    dataset_id: str = Field(description="Backend dataset identifier")
    created_at: str = Field(description="ISO timestamp when the run was initiated")
    updated_at: str = Field(description="ISO timestamp of last phase update")
    n_rows: Optional[int] = Field(default=None, description="Total rows in dataset")
    n_cols: Optional[int] = Field(default=None, description="Total columns in dataset")
    completed_phases: List[int] = Field(default_factory=list, description="List of completed phase IDs")
    active_phase: int = Field(default=1, description="Last active phase in the stepper")
    overall_score: Optional[float] = Field(default=None, description="Overall Empirical Score (0-100)")
    overall_grade: Optional[str] = Field(default=None, description="Letter grade from System Evaluation")
    mode: Optional[str] = Field(default=None, description="Branch mode: 'baseline', 'proposed', or None")
    paired_run_id: Optional[str] = Field(default=None, description="Linked paired run ID if part of comparative experiment")
    has_visual_evidence: bool = Field(default=False, description="True if run contains Vega visual evidence")
    planner_mode: Optional[str] = Field(default=None, description="Planner mode: 'llm', 'deterministic', or None")
    planner_model: Optional[str] = Field(default=None, description="Planner model: e.g. 'gemini-3.5-flash-lite' or None")
    planner_provider: Optional[str] = Field(default=None, description="Planner provider: e.g. 'google', 'deterministic', or None")


class RunPayload(BaseModel):
    """Complete serialized experiment run state across all pipeline phases."""
    run_id: Optional[str] = Field(default=None, description="Unique experiment identifier")
    dataset_name: str = Field(default="untitled_dataset.csv", description="Original dataset filename")
    dataset_id: Optional[str] = Field(default=None, description="Backend dataset identifier")
    created_at: Optional[str] = Field(default=None, description="ISO timestamp of run initiation")
    updated_at: Optional[str] = Field(default=None, description="ISO timestamp of last update")
    active_phase: int = Field(default=1, description="Active phase number (1-7)")
    completed_phases: List[int] = Field(default_factory=list, description="Array of completed phase numbers")
    objective: Optional[str] = Field(default="Investigate this dataset.", description="User investigation goal")
    mode: Optional[str] = Field(default=None, description="Branch mode: 'baseline', 'proposed', or None")
    paired_run_id: Optional[str] = Field(default=None, description="Linked paired run ID if part of comparative experiment")
    experiment_id: Optional[str] = Field(default=None, description="Experiment identifier if paired")
    has_visual_evidence: Optional[bool] = Field(default=None, description="Optional visual evidence flag")
    profile: Optional[dict[str, Any]] = Field(default=None, description="Phase 1 Dataset Profile")
    plan: Optional[dict[str, Any]] = Field(default=None, description="Phase 2 Planner Response")
    analysis: Optional[dict[str, Any]] = Field(default=None, description="Phase 3 Analysis Response")
    critic: Optional[dict[str, Any]] = Field(default=None, description="Phase 4 Critic Review Response")
    verification: Optional[dict[str, Any]] = Field(default=None, description="Phase 5 Verification Response")
    insights: Optional[dict[str, Any]] = Field(default=None, description="Phase 6 Key Insights Response")
    evaluation: Optional[dict[str, Any]] = Field(default=None, description="Phase 7 System Evaluation Response")

    model_config = {"extra": "allow"}


class RunListResponse(BaseModel):
    """Collection of saved experiment runs."""
    total_runs: int = Field(description="Total count of saved runs")
    runs: List[RunSummary] = Field(default_factory=list, description="List of run summaries sorted newest first")


class RunSaveResponse(BaseModel):
    """Response confirming run persistence."""
    status: str = Field(default="saved", description="Status confirmation")
    run_id: str = Field(description="Unique experiment identifier")
    updated_at: str = Field(description="Timestamp of save")
    summary: RunSummary = Field(description="Updated run summary")
