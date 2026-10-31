"""
backend/app/schemas/report_schema.py
====================================
Pydantic v2 schemas for DataInvest AI — Phase 5/Stage B: Report Agent.
Defines strongly typed contracts for structured research reports, findings summaries,
audit trail representations, and exportable outputs.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class DatasetContext(BaseModel):
    """Contextual metadata describing the investigated dataset."""
    dataset_name: str = Field(default="untitled_dataset.csv", description="Dataset filename or identifier")
    dataset_id: str = Field(default="unknown", description="Unique backend dataset identifier")
    n_rows: Optional[int] = Field(default=None, description="Number of rows in the dataset")
    n_cols: Optional[int] = Field(default=None, description="Number of columns in the dataset")
    total_missing_cells: Optional[int] = Field(default=None, description="Total count of missing data cells")
    missingness_pct: Optional[float] = Field(default=None, description="Overall dataset missingness percentage")
    numerical_columns_count: Optional[int] = Field(default=None, description="Number of numerical columns")
    categorical_columns_count: Optional[int] = Field(default=None, description="Number of categorical columns")
    target_variable: Optional[str] = Field(default=None, description="Identified analytical target variable if any")


class AnalysisContext(BaseModel):
    """Investigation goals, branch execution mode, and task configuration."""
    objective: str = Field(default="Investigate this dataset.", description="User or framework investigation objective")
    branch_mode: Optional[str] = Field(default=None, description="Execution branch: 'baseline', 'proposed', or None")
    paired_run_id: Optional[str] = Field(default=None, description="Linked run ID if part of a comparative experiment")
    completed_phases: List[int] = Field(default_factory=list, description="List of completed pipeline phase numbers")
    total_tasks_planned: int = Field(default=0, description="Total analytical tasks planned")
    successful_tasks: int = Field(default=0, description="Tasks executed without execution errors")
    failed_tasks: int = Field(default=0, description="Tasks that encountered execution or boundary errors")
    planner_mode: Optional[str] = Field(default=None, description="Execution mode: 'llm', 'deterministic', or None")
    planner_model: Optional[str] = Field(default=None, description="Model identifier if LLM mode was used")
    planner_provider: Optional[str] = Field(default=None, description="Provider: 'google', 'deterministic', or None")


class InitialFindingSummary(BaseModel):
    """Structured summary of an initial Phase 3 statistical finding prior to verification."""
    task_id: str = Field(description="Investigation task identifier")
    variables: List[str] = Field(default_factory=list, description="Variables evaluated in this test")
    analysis_type: str = Field(default="unknown", description="Analysis category (e.g. numerical_vs_numerical)")
    method: str = Field(default="unknown", description="Statistical test executed in Phase 3")
    statistic: Optional[float] = Field(default=None, description="Observed primary test statistic")
    p_value: Optional[float] = Field(default=None, description="Calculated p-value")
    p_value_formatted: Optional[str] = Field(default=None, description="Human-readable formatted p-value")
    sample_size: Optional[int] = Field(default=None, description="Effective sample size for this analysis")
    effect_size: Optional[float] = Field(default=None, description="Effect size magnitude (r, Cramér's V, eta^2, etc.)")
    effect_size_type: Optional[str] = Field(default=None, description="Type of effect size metric reported")
    status: str = Field(default="success", description="Execution status: 'success' or 'failed'")
    warnings: List[str] = Field(default_factory=list, description="Execution warnings recorded during analysis")


class CriticVerdictSummary(BaseModel):
    """Structured summary of Phase 4 Critic verification for a finding."""
    task_id: str = Field(description="Task identifier matching the analyzed finding")
    status: str = Field(description="Critic verdict: ACCEPTED, QUESTIONABLE, REJECTED, REQUIRES_REINVESTIGATION")
    confidence: float = Field(default=1.0, description="Confidence score between 0.0 and 1.0")
    confidence_level: str = Field(default="HIGH", description="Categorical confidence: HIGH, MEDIUM, LOW")
    summary: str = Field(description="Concise scientific summary of the Critic verdict")
    statistical_validity: Optional[str] = Field(default=None, description="Assessment of mathematical validity")
    method_assessment: Optional[str] = Field(default=None, description="Assessment of statistical method appropriateness")
    data_quality_concerns: List[str] = Field(default_factory=list, description="Data quality or missingness concerns")
    assumption_concerns: List[str] = Field(default_factory=list, description="Statistical assumption violations detected")
    reason_codes: List[str] = Field(default_factory=list, description="Standardized decision reason codes")
    evidence: List[str] = Field(default_factory=list, description="Factual evidence strings supporting the verdict")


class ReinvestigationDetail(BaseModel):
    """Provenance and comparative metrics for findings that underwent targeted re-investigation."""
    reinvestigation_id: str = Field(description="Unique identifier for this re-investigation event")
    source_task_id: str = Field(description="Source task identifier matching the original finding")
    variables: List[str] = Field(default_factory=list, description="Variables analyzed")
    original_method: str = Field(description="Initial statistical method executed in Phase 3")
    reinvestigated_method: str = Field(description="Alternative or robust method executed in Phase 5")
    trigger_reason: str = Field(description="Critic rationale that triggered re-investigation")
    original_statistic: Optional[float] = Field(default=None, description="Primary statistic from initial method")
    reinvestigated_statistic: Optional[float] = Field(default=None, description="Primary statistic from revised method")
    original_p_value: Optional[float] = Field(default=None, description="p-value from initial method")
    reinvestigated_p_value: Optional[float] = Field(default=None, description="p-value from revised method")
    direction_consistent: Optional[bool] = Field(default=None, description="Whether effect direction was preserved")
    significance_consistent: Optional[bool] = Field(default=None, description="Whether alpha=0.05 significance was preserved")
    comparison_summary: str = Field(description="Comparative assessment across both statistical procedures")
    final_status: str = Field(description="Outcome status after re-investigation")


class FinalFindingSummary(BaseModel):
    """Consolidated final finding distinguishing unrevised findings from reinvestigated revisions."""
    finding_id: str = Field(description="Unique finding identifier (e.g. 'finding_1')")
    task_id: str = Field(description="Source task identifier")
    variables: List[str] = Field(default_factory=list, description="Variables involved")
    method: str = Field(description="Final effective statistical method")
    status: str = Field(description="Final verification status: VERIFIED, VERIFIED_AFTER_REINVESTIGATION, QUESTIONABLE, REJECTED")
    confidence: float = Field(default=1.0, description="Confidence score")
    confidence_level: str = Field(default="HIGH", description="Categorical confidence")
    summary: str = Field(description="Verified analytical summary")
    verification_reason: str = Field(description="Verification rationale explaining final status")
    is_revised_after_reinvestigation: bool = Field(default=False, description="True if finding was re-tested and revised")
    limitations: List[str] = Field(default_factory=list, description="Specific methodological or data caveats")
    reason_codes: List[str] = Field(default_factory=list, description="Standardized decision codes")
    has_visual_evidence: bool = Field(default=False, description="True if accompanied by visual chart evidence")


class EvaluationSummary(BaseModel):
    """High-level system evaluation score and benchmark grade if Phase 7 completed."""
    overall_score: Optional[float] = Field(default=None, description="Overall Empirical Rigor Score (0-100)")
    overall_grade: Optional[str] = Field(default=None, description="Overall letter grade (e.g. 'A+', 'B')")
    benchmark_pass_rate: Optional[str] = Field(default=None, description="Pass rate across benchmark checks")
    summary_verdict: Optional[str] = Field(default=None, description="Executive evaluation verdict")


class ResearchReport(BaseModel):
    """Complete, structured scientific report assembled from persistent run state."""
    report_id: str = Field(description="Unique report identifier")
    run_id: str = Field(description="Unique experiment run identifier this report was synthesized from")
    title: str = Field(default="DataInvest AI Investigation Report", description="Report title")
    generated_at: str = Field(description="ISO timestamp when report was synthesized")
    dataset_context: DatasetContext = Field(description="Dataset metadata and profile summary")
    analysis_context: AnalysisContext = Field(description="Investigation objective and task execution summary")
    initial_findings: List[InitialFindingSummary] = Field(default_factory=list, description="Phase 3 statistical results")
    critic_verdicts: List[CriticVerdictSummary] = Field(default_factory=list, description="Phase 4 Critic verification verdicts")
    reinvestigations: List[ReinvestigationDetail] = Field(default_factory=list, description="Phase 5 targeted re-investigation history")
    final_findings: List[FinalFindingSummary] = Field(default_factory=list, description="Consolidated final findings")
    limitations_and_unresolved_issues: List[str] = Field(default_factory=list, description="Unresolved data quality and assumption limitations")
    methodological_disclaimer: str = Field(
        default=(
            "The Critic accepted findings under the implemented verification checks. "
            "Acceptance does not establish statistical correctness, causal validity, or mathematical proof. "
            "Statistical verification indicates that parametric and non-parametric assumptions, sample size adequacy, "
            "and mathematical bounds were checked by the deterministic Critic Agent; it does not constitute mathematical "
            "or causal proof. Findings are observational and conditioned on dataset completeness and collection methodology."
        ),
        description="Formal scientific disclaimer on verification boundaries",
    )
    evaluation_summary: Optional[EvaluationSummary] = Field(default=None, description="Phase 7 benchmark scorecard summary if available")
