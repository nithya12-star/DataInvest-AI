"""
backend/app/schemas/critic_schema.py
====================================
Pydantic v2 schemas for DataInvest AI — Phase 4 Critic Agent.
Provides strongly typed data contracts for statistical verification, finding classification,
method appropriateness auditing, and re-investigation recommendations.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from app.schemas.analysis_schema import AnalysisResult, PlanAnalysisResponse
from app.schemas.planner_schema import PlannerResponse


CriticStatus = Literal[
    "ACCEPTED",
    "QUESTIONABLE",
    "REJECTED",
    "REQUIRES_REINVESTIGATION",
]

ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]


class CriticResult(BaseModel):
    """
    Structured verification evaluation for a single Phase 3 analysis result.
    Evaluated purely from empirical evidence without fabricating statistical values.
    """
    task_id: str = Field(description="Task identifier matching the Phase 3 AnalysisResult")
    analysis_type: str = Field(description="Standardized analytical category evaluated")
    method: str = Field(description="Statistical method that was evaluated")
    variables: list[str] = Field(default_factory=list, description="Variables involved in this finding")
    status: CriticStatus = Field(description="Critic verification outcome: ACCEPTED, QUESTIONABLE, REJECTED, or REQUIRES_REINVESTIGATION")
    confidence: float = Field(default=1.0, description="Deterministic confidence score between 0.0 and 1.0")
    confidence_level: ConfidenceLevel = Field(default="HIGH", description="Categorical confidence: HIGH, MEDIUM, or LOW")
    summary: str = Field(description="Concise scientific assessment of the finding's validity")
    evidence: list[str] = Field(default_factory=list, description="Concrete empirical evidence verified from Phase 3 results")
    statistical_validity: str = Field(description="Assessment of mathematical validity, p-value bounds, and test statistic sanity")
    method_assessment: str = Field(description="Evaluation of method appropriateness vs variable data types and distribution")
    data_quality_concerns: list[str] = Field(default_factory=list, description="Missingness, constant variables, or sparsity issues")
    assumption_concerns: list[str] = Field(default_factory=list, description="Normality, outlier sensitivity, or group imbalance notes")
    effect_size_assessment: Optional[str] = Field(default=None, description="Evaluation of effect magnitude vs statistical significance")
    recommendation: str = Field(description="Actionable methodological guidance for interpreting this finding")
    requires_reinvestigation: bool = Field(default=False, description="True if an alternative statistical test should be conducted")
    recommended_method: Optional[str] = Field(default=None, description="Alternative statistical method if re-investigation is recommended")
    reason_codes: list[str] = Field(default_factory=list, description="Standardized decision codes for full traceability")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Provenance and agent metadata")


class CriticOverallSummary(BaseModel):
    """Aggregate quality and integrity summary across all evaluated findings."""
    total_results: int = Field(description="Total number of findings evaluated")
    accepted: int = Field(description="Count of robust, verified findings")
    questionable: int = Field(description="Count of findings with power, assumption, or data quality limitations")
    rejected: int = Field(description="Count of invalid, skipped, or failed findings")
    requires_reinvestigation: int = Field(description="Count of findings recommending alternative statistical methods")
    overall_status: str = Field(description="High-level integrity verdict across all evaluated findings")
    critical_issues: list[str] = Field(default_factory=list, description="Major data quality or methodological warnings")
    key_verified_insights: list[str] = Field(default_factory=list, description="Summary of high-confidence empirical insights")


class CriticRequest(BaseModel):
    """Input payload to request Critic verification of Phase 3 results."""
    analysis_response: Optional[PlanAnalysisResponse] = Field(
        default=None,
        description="Phase 3 PlanAnalysisResponse containing executed analysis results"
    )
    results: Optional[list[AnalysisResult]] = Field(
        default=None,
        description="Direct list of AnalysisResult items (if not using PlanAnalysisResponse wrapper)"
    )
    dataset_profile: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional Phase 1 dataset profile dictionary for context on missingness/skewness"
    )
    investigation_plan: Optional[PlannerResponse] = Field(
        default=None,
        description="Optional Phase 2 investigation plan for task rationale context"
    )
    dataset_id: Optional[str] = Field(
        default=None,
        description="Dataset identifier or filename for provenance"
    )


class CriticResponse(BaseModel):
    """Structured response returned by the Critic Agent."""
    dataset_id: str = Field(description="Identifier of the evaluated dataset")
    summary: CriticOverallSummary = Field(description="Aggregate evaluation summary")
    reviews: list[CriticResult] = Field(description="Ordered list of individual finding verification reviews")
    critic_version: str = Field(default="4.0", description="Critic agent version")
    mode: str = Field(default="deterministic", description="Mode used: 'deterministic' or 'llm'")
