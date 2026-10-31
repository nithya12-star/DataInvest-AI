"""
backend/app/services/analysis_service.py
========================================
Service layer orchestrating dataset resolution and Analysis Agent execution.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd

from app.agents.analysis_agent import AnalysisAgent
from app.schemas.analysis_schema import (
    AnalysisError,
    AnalysisResult,
    PlanAnalysisRequest,
    PlanAnalysisResponse,
    SingleAnalysisRequest,
)
from app.services.dataset_store import resolve_dataset


class AnalysisService:
    """Orchestrates statistical analysis execution against stored or provided datasets."""

    def __init__(self, agent: Optional[AnalysisAgent] = None):
        self.agent = agent or AnalysisAgent()

    def execute_single_task(self, request: SingleAnalysisRequest) -> AnalysisResult:
        """Execute a single investigation task against the target dataset."""
        df = resolve_dataset(request.dataset_id)
        if df is None or df.empty:
            return AnalysisResult(
                task_id=request.task.task_id,
                status="failed",
                analysis_type=request.task.task_type,
                method=request.task.method,
                columns=request.task.columns,
                sample_size=0,
                results={},
                warnings=[],
                error=AnalysisError(
                    code="DATASET_NOT_FOUND",
                    message=(
                        f"Dataset '{request.dataset_id or 'default'}' could not be resolved. "
                        "Please ensure the dataset was uploaded or cached in memory."
                    ),
                ),
            )

        return self.agent.execute_task(request.task, df)

    def execute_plan(self, request: PlanAnalysisRequest) -> PlanAnalysisResponse:
        """Execute an entire planned list of investigation tasks against the target dataset."""
        df = resolve_dataset(request.dataset_id)
        dataset_id_str = request.dataset_id or "active_dataset"

        if df is None or df.empty:
            # Return failed results for all tasks with failure isolation
            failed_results = [
                AnalysisResult(
                    task_id=task.task_id,
                    status="failed",
                    analysis_type=task.task_type,
                    method=task.method,
                    columns=task.columns,
                    sample_size=0,
                    results={},
                    warnings=[],
                    error=AnalysisError(
                        code="DATASET_NOT_FOUND",
                        message=(
                            f"Dataset '{dataset_id_str}' could not be resolved. "
                            "Please ensure the dataset was uploaded or cached in memory."
                        ),
                    ),
                )
                for task in request.tasks
            ]
            return PlanAnalysisResponse(
                dataset_id=dataset_id_str,
                total_tasks=len(request.tasks),
                successful_tasks=0,
                failed_tasks=len(request.tasks),
                results=failed_results,
            )

        return self.agent.execute_plan(
            tasks=request.tasks,
            df=df,
            dataset_id=dataset_id_str,
        )


# Global default service instance
_ANALYSIS_SERVICE = AnalysisService()


def get_analysis_service() -> AnalysisService:
    """Provide the global AnalysisService instance."""
    return _ANALYSIS_SERVICE
