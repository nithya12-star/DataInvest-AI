"""
backend/app/schemas/verification_schema.py
==========================================
Pydantic v2 schemas for DataInvest AI — Phase 5: Re-investigation & Verified Findings.
Provides strongly typed data contracts for targeted re-investigation, before/after
result comparison, audit trail provenance, and final verified insight categorization.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from app.schemas.analysis_schema import AnalysisResult, PlanAnalysisResponse
from app.schemas.critic_schema import CriticResponse, CriticResult
from app.schemas.planner_schema import PlannerResponse
from app.schemas.visualization_schema import VisualEvidence


FinalVerificationStatus = Literal[
    "VERIFIED",
    "VERIFIED_AFTER_REINVESTIGATION",
    "QUESTIONABLE",
    "REJECTED",
]


class ResultComparison(BaseModel):
    """
    Structured comparative metrics between original analysis and re-investigated analysis.
    Computed purely from Python/SciPy statistical results.
    """
    original_method: str = Field(description="Method executed in original Phase 3 analysis")
    reinvestigated_method: str = Field(description="Method executed in Phase 5 re-investigation")
    original_statistic: Optional[float] = Field(default=None, description="Primary test statistic from original analysis")
    reinvestigated_statistic: Optional[float] = Field(default=None, description="Primary test statistic from re-investigated analysis")
    original_p_value: Optional[float] = Field(default=None, description="p-value from original analysis")
    reinvestigated_p_value: Optional[float] = Field(default=None, description="p-value from re-investigated analysis")
    original_sample_size: Optional[int] = Field(default=None, description="Sample size evaluated in original analysis")
    reinvestigated_sample_size: Optional[int] = Field(default=None, description="Sample size evaluated in re-investigated analysis")
    original_effect_size: Optional[float] = Field(default=None, description="Effect size from original analysis")
    reinvestigated_effect_size: Optional[float] = Field(default=None, description="Effect size from re-investigated analysis")
    direction_consistent: Optional[bool] = Field(default=None, description="True if correlation direction or group order is preserved")
    significance_consistent: Optional[bool] = Field(default=None, description="True if statistical significance at alpha=0.05 is consistent")
    magnitude_delta: Optional[float] = Field(default=None, description="Absolute difference in primary statistic magnitude")
    effect_size_shift: Optional[str] = Field(default=None, description="Qualitative shift in effect size interpretation")
    summary: str = Field(description="Factual comparative summary of findings across both methods")


class ReinvestigationRecord(BaseModel):
    """
    Complete provenance and audit record for a re-investigated finding.
    Preserves all intermediate stages: original analysis, initial critique,
    re-investigation execution, second critique, and comparative delta.
    """
    reinvestigation_id: str = Field(description="Unique identifier for this re-investigation task")
    source_task_id: str = Field(description="Source task identifier matching the original Phase 2/3 task")
    original_method: str = Field(description="Original method evaluated in Phase 3")
    reinvestigated_method: str = Field(description="New method executed during re-investigation")
    variables: list[str] = Field(default_factory=list, description="Variables analyzed in this task")
    trigger_reason: str = Field(description="Reason why the initial Critic review requested re-investigation")
    original_analysis: AnalysisResult = Field(description="Complete original Phase 3 AnalysisResult")
    original_critic_review: CriticResult = Field(description="Initial Phase 4 CriticResult")
    reinvestigated_analysis: AnalysisResult = Field(description="New Phase 5 AnalysisResult executed via SciPy")
    second_critic_review: CriticResult = Field(description="Second Phase 5 CriticResult evaluating the new evidence")
    comparison: ResultComparison = Field(description="Structured before/after comparative delta")
    status: FinalVerificationStatus = Field(description="Final verification status for this re-investigated finding")
    attempt_number: int = Field(default=1, description="Attempt number (enforces MAX_REINVESTIGATIONS=1)")
    original_statistic: Optional[float] = Field(default=None, description="Primary test statistic from original analysis")
    reinvestigated_statistic: Optional[float] = Field(default=None, description="Primary test statistic from re-investigated analysis")
    original_p_value: Optional[float] = Field(default=None, description="p-value from original analysis")
    reinvestigated_p_value: Optional[float] = Field(default=None, description="p-value from re-investigated analysis")
    original_effect_size: Optional[float] = Field(default=None, description="Effect size from original analysis")
    reinvestigated_effect_size: Optional[float] = Field(default=None, description="Effect size from re-investigated analysis")
    original_sample_size: Optional[int] = Field(default=None, description="Sample size from original analysis")
    reinvestigated_sample_size: Optional[int] = Field(default=None, description="Sample size from re-investigated analysis")
    original_graph_id: Optional[str] = Field(default=None, description="Graph ID of original visual evidence")
    reinvestigated_graph_id: Optional[str] = Field(default=None, description="Graph ID of regenerated follow-up visual evidence")
    original_visual_evidence: Optional[VisualEvidence] = Field(default=None, description="Preserved original visual evidence")
    reinvestigated_visual_evidence: Optional[VisualEvidence] = Field(default=None, description="Preserved regenerated follow-up visual evidence")


class VerifiedFinding(BaseModel):
    """
    Final structured finding model representing a verified, questionable, or rejected analytical insight.
    Includes full statistical evidence, critic verification rationale, limitations, and audit trail.
    """
    finding_id: str = Field(description="Unique identifier for the final verified finding (e.g. 'finding_1')")
    task_id: str = Field(description="Original task identifier from Phase 2")
    variables: list[str] = Field(default_factory=list, description="Variables involved in this finding")
    method: str = Field(description="Final effective statistical method (original or re-investigated)")
    status: FinalVerificationStatus = Field(description="Final status: VERIFIED, VERIFIED_AFTER_REINVESTIGATION, QUESTIONABLE, or REJECTED")
    confidence: float = Field(default=1.0, description="Deterministic confidence score between 0.0 and 1.0")
    confidence_level: str = Field(default="HIGH", description="Categorical confidence: HIGH, MEDIUM, or LOW")
    summary: str = Field(description="Concise, verified scientific assessment of the finding")
    statistical_evidence: dict[str, Any] = Field(default_factory=dict, description="Final empirical statistics (from Python/SciPy)")
    critic_assessment: str = Field(description="Critic verification rationale supporting the final status")
    reinvestigated: bool = Field(default=False, description="True if this finding underwent targeted Phase 5 re-investigation")
    reinvestigation_record: Optional[ReinvestigationRecord] = Field(default=None, description="Detailed re-investigation audit trail if applicable")
    graph_id: Optional[str] = Field(default=None, description="Associated dynamic graph identifier")
    visual_evidence: Optional[VisualEvidence] = Field(default=None, description="Dynamic visual evidence associated with this finding")
    verification_reason: str = Field(description="Explicit explanation of why this finding was verified, questioned, or rejected")
    limitations: list[str] = Field(default_factory=list, description="Documented data quality, power, or assumption limitations")
    reason_codes: list[str] = Field(default_factory=list, description="Standardized decision codes for full traceability")


class VerificationSummary(BaseModel):
    """Dynamic aggregate metrics across all verified, questionable, and rejected findings."""
    total_findings: int = Field(description="Total number of findings evaluated")
    verified: int = Field(description="Count of directly verified findings")
    verified_after_reinvestigation: int = Field(description="Count of findings successfully verified after targeted re-investigation")
    questionable: int = Field(description="Count of findings retained with methodological caveats")
    rejected: int = Field(description="Count of invalid or failed findings excluded from verified insights")
    total_reinvestigated: int = Field(description="Total number of findings that underwent targeted re-analysis")
    overall_integrity: str = Field(description="High-level integrity verdict across all verified findings")
    unresolved_issues: list[str] = Field(default_factory=list, description="Remaining data quality or methodological limitations")


class VerificationRequest(BaseModel):
    """Input payload to request Phase 5 re-investigation and final finding verification."""
    analysis_response: Optional[PlanAnalysisResponse] = Field(
        default=None,
        description="Phase 3 PlanAnalysisResponse containing executed statistical results"
    )
    critic_response: Optional[CriticResponse] = Field(
        default=None,
        description="Phase 4 CriticResponse containing initial finding evaluations"
    )
    dataset_profile: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional Phase 1 dataset profile dictionary"
    )
    investigation_plan: Optional[PlannerResponse] = Field(
        default=None,
        description="Optional Phase 2 investigation plan"
    )
    dataset_id: Optional[str] = Field(
        default=None,
        description="Dataset identifier or filename for in-memory retrieval"
    )
    max_reinvestigations: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Configurable maximum number of re-investigation attempts per finding (0 to 3, default 1)",
    )


class VerificationResponse(BaseModel):
    """Complete, structured response returned by Phase 5."""
    dataset_id: str = Field(description="Identifier of the evaluated dataset")
    summary: VerificationSummary = Field(description="Dynamic aggregate verification summary")
    verified_findings: list[VerifiedFinding] = Field(description="List of verified and robust findings")
    questionable_findings: list[VerifiedFinding] = Field(description="List of findings with documented caveats")
    rejected_findings: list[VerifiedFinding] = Field(description="List of invalid, failed, or skipped findings")
    reinvestigations: list[ReinvestigationRecord] = Field(description="Audit records of all re-investigations conducted")
    all_findings: list[VerifiedFinding] = Field(description="Consolidated list of all findings in sequential task order")
    phase_version: str = Field(default="5.0", description="DataInvest AI Phase 5 version")
