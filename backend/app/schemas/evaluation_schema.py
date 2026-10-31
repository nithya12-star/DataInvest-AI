# backend/app/schemas/evaluation_schema.py
"""
Pydantic v2 schemas for Phase 7 — System Evaluation & Quality Scorecard.
Quantifies pipeline reliability, statistical rigor, verification truth grounding,
and insight actionability based on the AutoMind research benchmark.
"""

from typing import List, Literal, Optional, Any, Dict, Union
from pydantic import BaseModel, Field, ConfigDict

EvaluationStatus = Literal["EXCELLENT", "GOOD", "ACCEPTABLE", "NEEDS_IMPROVEMENT"]


class MetricItem(BaseModel):
    """Granular quantitative metric entry."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metric_id: str
    name: str
    dimension: str
    value: Union[float, int, str]
    unit: str = "%"
    status: EvaluationStatus = "GOOD"
    description: str
    benchmark_target: str


class DimensionScore(BaseModel):
    """Scorecard for a specific scientific/architectural dimension."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dimension_key: str
    name: str
    score: float = Field(ge=0.0, le=25.0)
    max_score: float = 25.0
    percentage: float = Field(ge=0.0, le=100.0)
    grade: str
    metrics: List[MetricItem] = Field(default_factory=list)
    summary: str


class BenchmarkCheck(BaseModel):
    """AutoMind / Empirical standard compliance validation check."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    check_id: str
    name: str
    standard: str
    passed: bool
    detail: str


class EvaluationSummary(BaseModel):
    """Executive high-level summary of system evaluation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    overall_score: float = Field(ge=0.0, le=100.0)
    overall_grade: str
    total_tasks_planned: int = 0
    total_tasks_executed: int = 0
    total_findings_verified: int = 0
    total_insights_generated: int = 0
    zero_hallucination_verified: bool = True
    executive_assessment: str


class EvaluationRequest(BaseModel):
    """Payload for full pipeline evaluation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile: Dict[str, Any]
    plan: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    critic: Optional[Dict[str, Any]] = None
    verification: Optional[Dict[str, Any]] = None
    insights: Optional[Dict[str, Any]] = None


class EvaluationResponse(BaseModel):
    """Complete evaluation scorecard response."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    summary: EvaluationSummary
    dimensions: List[DimensionScore]
    benchmark_checks: List[BenchmarkCheck]
    detailed_metrics: List[MetricItem]
