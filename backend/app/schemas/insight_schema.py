"""
backend/app/schemas/insight_schema.py
======================================
Pydantic v2 schemas for DataInvest AI — Phase 6: Insight Generation.
Provides strongly typed contracts for converting Phase 5 verified statistical
findings into clear, human-readable, non-causal analytical insights with full
evidence traceability and deterministic strength classification.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from app.schemas.planner_schema import PlannerResponse
from app.schemas.verification_schema import (
    FinalVerificationStatus,
    VerificationResponse,
    VerifiedFinding,
)


InterpretationType = Literal[
    "ASSOCIATION",
    "GROUP_DIFFERENCE",
    "CATEGORICAL_ASSOCIATION",
    "SENSITIVITY",
    "DESCRIPTIVE",
]

InsightStrength = Literal[
    "STRONG",
    "MODERATE",
    "WEAK",
    "NOT_ASSESSED",
]


class InsightEvidence(BaseModel):
    """
    Detailed empirical evidence directly injected from Phase 5 structured findings.
    Zero statistical calculation or fabrication is performed in the insight layer.
    """
    method: str = Field(description="Statistical method that produced the verified finding")
    variables: list[str] = Field(default_factory=list, description="Variables analyzed in the finding")
    statistic: Optional[float] = Field(default=None, description="Primary test statistic (r, t, U, F, H, Chi2, etc.)")
    statistic_name: Optional[str] = Field(default=None, description="Human-readable name of the statistic (e.g. 'Spearman r', 'Welch t')")
    p_value: Optional[float] = Field(default=None, description="Exact p-value from Python/SciPy calculations")
    effect_size: Optional[float] = Field(default=None, description="Exact effect size value (r, Cohen's d, Eta-squared, Cramer's V)")
    effect_size_label: Optional[str] = Field(default=None, description="Qualitative label (e.g. 'large', 'moderate', 'small')")
    sample_size: Optional[int] = Field(default=None, description="Sample size analyzed (n)")
    group_metrics: Optional[dict[str, Any]] = Field(default=None, description="Group-specific descriptive metrics (means, medians, counts)")
    reinvestigated: bool = Field(default=False, description="True if verified after Phase 5 re-investigation")
    reinvestigated_method: Optional[str] = Field(default=None, description="Alternative method used in re-investigation if applicable")
    comparison_summary: Optional[str] = Field(default=None, description="Comparative delta summary if re-investigated")


class Insight(BaseModel):
    """
    Structured, human-readable analytical insight derived exclusively from a verified finding.
    Enforces non-causal language, evidence traceability, and deterministic strength classification.
    """
    insight_id: str = Field(description="Unique identifier for the generated insight (e.g. 'insight_1')")
    source_task_id: str = Field(description="Original task identifier from Phase 2")
    title: str = Field(description="Concise, informative headline summarizing the finding")
    insight: str = Field(description="Human-readable interpretation using strictly non-causal, observational phrasing")
    interpretation_type: InterpretationType = Field(description="Analytical category of the insight")
    strength: InsightStrength = Field(default="NOT_ASSESSED", description="Deterministic strength classification based on effect size")
    verification_status: FinalVerificationStatus = Field(description="Status from Phase 5 (VERIFIED or VERIFIED_AFTER_REINVESTIGATION)")
    confidence_score: float = Field(default=1.0, description="Confidence score from Critic evaluation (0.0 to 1.0)")
    is_statistically_significant: bool = Field(default=True, description="True if p-value is below significance threshold (alpha=0.05)")
    evidence: InsightEvidence = Field(description="Traceable empirical evidence directly linked from Phase 5")
    limitations: list[str] = Field(default_factory=list, description="Relevant methodological and observational limitations")
    practical_implication: Optional[str] = Field(default=None, description="Contextual note distinguishing statistical significance from practical magnitude")
    priority_score: float = Field(default=0.0, description="Deterministic ranking score used for ordering key insights")


class UnresolvedFinding(BaseModel):
    """
    Segregated representation of questionable or rejected findings.
    These are preserved for audit transparency but NEVER presented as verified insights.
    """
    finding_id: str = Field(description="Identifier of the unresolved finding")
    task_id: str = Field(description="Original task identifier")
    status: FinalVerificationStatus = Field(description="Status: QUESTIONABLE or REJECTED")
    title: str = Field(description="Brief headline of the unresolved item")
    summary: str = Field(description="Explanation of why this finding is questionable or rejected")
    variables: list[str] = Field(default_factory=list, description="Variables involved")
    limitations: list[str] = Field(default_factory=list, description="Documented concerns or rejection rationale")


class InsightSummary(BaseModel):
    """Dynamic aggregate statistics across all generated insights."""
    total_verified_findings: int = Field(description="Total count of verified findings evaluated")
    total_insights: int = Field(description="Total number of confirmed insights generated")
    strong_insights: int = Field(description="Count of insights with strong effect size / magnitude")
    moderate_insights: int = Field(description="Count of insights with moderate effect size / magnitude")
    weak_insights: int = Field(description="Count of insights with weak effect size / magnitude")
    not_assessed_insights: int = Field(description="Count of descriptive or sensitivity insights without formal effect size")
    unresolved_count: int = Field(description="Count of questionable or rejected findings excluded from confirmed insights")
    top_insight_title: Optional[str] = Field(default=None, description="Headline of the highest-ranked key insight")


class InsightRequest(BaseModel):
    """Input payload to request Phase 6 insight generation."""
    verification_response: Optional[VerificationResponse] = Field(
        default=None,
        description="Complete Phase 5 VerificationResponse containing verified findings"
    )
    verified_findings: Optional[list[VerifiedFinding]] = Field(
        default=None,
        description="Optional direct list of verified findings"
    )
    dataset_profile: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional Phase 1 dataset profile"
    )
    investigation_plan: Optional[PlannerResponse] = Field(
        default=None,
        description="Optional Phase 2 investigation plan"
    )
    dataset_id: Optional[str] = Field(
        default=None,
        description="Dataset identifier or filename"
    )


class InsightResponse(BaseModel):
    """Complete, structured response returned by Phase 6."""
    dataset_id: str = Field(description="Identifier of the analyzed dataset")
    summary: InsightSummary = Field(description="Dynamic aggregate summary of generated insights")
    key_insights: list[Insight] = Field(description="Ranked, deduplicated list of verified analytical insights")
    unresolved_findings: list[UnresolvedFinding] = Field(default_factory=list, description="Transparently segregated questionable and rejected findings")
    phase_version: str = Field(default="6.0", description="DataInvest AI Phase 6 version")
