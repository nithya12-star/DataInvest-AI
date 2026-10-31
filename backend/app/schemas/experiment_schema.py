"""
backend/app/schemas/experiment_schema.py
========================================
Pydantic v2 schemas for Paired Experimentation Engine.
Defines strongly typed request, branch summary, comparison metric,
and response contracts for evaluating Baseline (Plan -> Analysis -> Evaluation)
versus Proposed (Plan -> Analysis -> Critic -> Verification -> Insights -> Evaluation).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.planner_schema import PlannerResponse


class PairedExperimentRequest(BaseModel):
    """Request payload for running a paired baseline-vs-proposed experiment."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: Optional[str] = Field(
        default=None,
        description="Dataset identifier or filename for resolving cached data and provenance tracking",
    )
    profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase 1 Dataset Profile dictionary. If omitted, computed authoritatively from stored dataset.",
    )
    objective: str = Field(
        default="Investigate this dataset.",
        min_length=3,
        max_length=1000,
        description="User investigation goal in natural language",
    )
    max_reinvestigations: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Maximum re-investigation attempts per flagged finding (0 to 3)",
    )
    persist_runs: bool = Field(
        default=True,
        description="Whether to persist both baseline and proposed runs to the run repository on disk",
    )

    @field_validator("profile")
    @classmethod
    def validate_profile(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("A non-empty profile dictionary is required if provided.")
        overview = v.get("overview")
        if not overview or not isinstance(overview, dict):
            raise ValueError("Profile overview dictionary is required.")
        if overview.get("n_rows", 0) <= 0:
            raise ValueError("Dataset profile contains no valid rows (n_rows must be > 0).")
        return v

    @model_validator(mode="after")
    def validate_dataset_or_profile(self) -> PairedExperimentRequest:
        if not self.dataset_id and not self.profile:
            raise ValueError("Either 'dataset_id' or 'profile' must be provided.")
        return self


class BranchRunSummary(BaseModel):
    """Execution summary of a single experimental branch (baseline or proposed)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str = Field(description="Unique run identifier for this branch")
    mode: Literal["baseline", "proposed"] = Field(description="Experimental branch mode")
    dataset_id: str = Field(description="Dataset identifier")
    total_tasks_executed: int = Field(description="Number of analysis tasks executed")
    overall_score: float = Field(description="Empirical quality score (0-100)")
    overall_grade: str = Field(description="Letter grade assigned by Evaluation Service")
    dimension_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of score by scientific dimension",
    )
    completed_phases: List[int] = Field(
        default_factory=list,
        description="List of pipeline phases executed in this branch",
    )
    has_critic_review: bool = Field(
        default=False,
        description="Whether Critic review was conducted in this branch",
    )
    has_verification: bool = Field(
        default=False,
        description="Whether statistical re-investigation and verification was conducted",
    )
    has_insights: bool = Field(
        default=False,
        description="Whether insight generation was conducted in this branch",
    )
    persisted: bool = Field(
        default=False,
        description="Whether this branch record was successfully saved to disk",
    )
    paired_run_id: Optional[str] = Field(
        default=None,
        description="Run ID of the paired branch (baseline links to proposed, proposed links to baseline)",
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Measured wall-clock duration in seconds for branch execution",
    )


class MetricDeltaSummary(BaseModel):
    """
    Empirical delta comparisons between baseline and proposed branches.
    Strictly derived from actual service outputs; never hardcoded.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    baseline_score: float = Field(description="Baseline branch overall score")
    proposed_score: float = Field(description="Proposed branch overall score")
    overall_score_delta: float = Field(
        description="Proposed score minus baseline score (can be positive, zero, or negative)"
    )
    rigor_score_delta: float = Field(
        description="Proposed statistical rigor score minus baseline rigor score"
    )
    verification_score_delta: float = Field(
        description="Proposed verification grounding score minus baseline verification score"
    )
    dimension_deltas: Dict[str, float] = Field(
        default_factory=dict,
        description="Delta for each evaluation dimension (proposed - baseline)",
    )
    assumption_violations_flagged: int = Field(
        default=0,
        description="Number of findings flagged for assumption violations by Critic",
    )
    violations_resolved_via_reinvestigation: int = Field(
        default=0,
        description="Number of flagged findings resolved via alternative test in re-investigation",
    )
    resolution_rate: Optional[float] = Field(
        default=None,
        description="Ratio of resolved violations to flagged violations (None if no violations flagged)",
    )
    total_reinvestigations: int = Field(
        default=0,
        description="Total re-investigations conducted in proposed branch",
    )
    verified_findings_count: int = Field(
        default=0,
        description="Count of robustly verified findings in proposed branch",
    )
    questionable_findings_count: int = Field(
        default=0,
        description="Count of findings retained with methodological caveats",
    )
    rejected_findings_count: int = Field(
        default=0,
        description="Count of invalid/failed findings rejected",
    )
    initial_findings_count: Optional[int] = Field(
        default=None,
        description="Total initial findings produced by Analysis before Critic verification",
    )
    findings_accepted_directly: Optional[int] = Field(
        default=None,
        description="Findings accepted directly by Critic without requiring reinvestigation",
    )
    findings_requiring_reinvestigation: Optional[int] = Field(
        default=None,
        description="Findings flagged by Critic as requiring controlled reinvestigation",
    )
    findings_revised: Optional[int] = Field(
        default=None,
        description="Findings verified and revised after controlled alternative-test reinvestigation",
    )
    findings_unresolved: Optional[int] = Field(
        default=None,
        description="Findings unresolved (questionable or rejected) after evaluation",
    )
    scientific_verdict: str = Field(
        description="Scientific verdict based strictly on actual comparison results",
    )

    # Task 1 — Monotonic Wall-Clock Runtime Instrumentation
    total_experiment_duration_seconds: Optional[float] = Field(
        default=None,
        description="Total measured wall-clock duration of the paired experiment in seconds",
    )
    shared_duration_seconds: Optional[float] = Field(
        default=None,
        description="Shared duration of planning and initial analysis executed once for both branches",
    )
    planning_duration_seconds: Optional[float] = Field(
        default=None,
        description="Measured duration of shared planning phase in seconds",
    )
    analysis_duration_seconds: Optional[float] = Field(
        default=None,
        description="Measured duration of shared initial statistical analysis phase in seconds",
    )
    critic_duration_seconds: Optional[float] = Field(
        default=None,
        description="Measured duration of proposed branch Critic evaluation in seconds",
    )
    verification_duration_seconds: Optional[float] = Field(
        default=None,
        description="Measured duration of proposed branch verification and controlled reinvestigation in seconds",
    )
    proposed_overhead_seconds: Optional[float] = Field(
        default=None,
        description="Additional computational overhead incurred by proposed branch for self-verification",
    )
    baseline_duration_seconds: Optional[float] = Field(
        default=None,
        description="Measured duration of baseline branch-specific evaluation in seconds",
    )
    proposed_duration_seconds: Optional[float] = Field(
        default=None,
        description="Measured duration of proposed branch-specific stages in seconds",
    )

    # Task 2 — Rule-Level Critic Metrics
    cochran_violations_count: Optional[int] = Field(
        default=None,
        description="Count of findings violating Cochran (1954) contingency guidelines (expected cell count < 1.0 or >20% below 5.0)",
    )
    normality_violations_count: Optional[int] = Field(
        default=None,
        description="Count of findings violating normality assumptions (distributional skew > 1.5 or extreme boxplot outliers)",
    )
    leverage_outlier_violations_count: Optional[int] = Field(
        default=None,
        description="Outlier-influence diagnostic: identifies observations that may disproportionately influence Pearson correlation. This diagnostic alone does not establish that the linearity assumption is violated.",
    )
    rule_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Structured mapping of specific Critic decision/reason codes to their trigger counts across findings",
    )


class PairedExperimentResponse(BaseModel):
    """Complete paired experiment response comparing baseline and proposed pipelines."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    experiment_id: str = Field(description="Unique identifier for the paired experiment")
    dataset_id: str = Field(description="Dataset identifier")
    objective: str = Field(description="Investigation objective")
    created_at: str = Field(description="ISO timestamp when the experiment finished")
    shared_plan: PlannerResponse = Field(
        description="The single, shared PlannerResponse generated by the LLM Planner and executed by both branches"
    )
    baseline: BranchRunSummary = Field(description="Execution summary of the baseline branch")
    proposed: BranchRunSummary = Field(description="Execution summary of the proposed branch")
    comparison: MetricDeltaSummary = Field(description="Empirical delta metrics between the two branches")
    persisted: bool = Field(
        default=False,
        description="True if both branch runs were successfully written to disk",
    )
    partial_persistence: bool = Field(
        default=False,
        description="True if one branch was written but the other failed",
    )
    execution_mode: str = Field(
        default="research_run",
        description="Execution mode: 'research_run' (genuine LLM) or 'offline_fixture_replay' (deterministic offline benchmark)",
    )
    is_offline_fixture: bool = Field(
        default=False,
        description="True if the experiment was executed using offline fixture replay rather than live LLM planning",
    )
