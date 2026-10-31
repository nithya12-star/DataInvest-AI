"""
backend/app/agents/analysis_agent.py
====================================
Phase 3: Analysis Agent & Statistical Execution Engine.

Purpose:
    Execute planned investigation tasks using pure Python, SciPy, and NumPy.
    All p-values, test statistics, and effect sizes are strictly computed by SciPy.
    Zero hallucination or fabricated statistical results.

Failure Isolation:
    Individual task failures (e.g. constant variable, insufficient sample size)
    are caught and packaged into structured failed AnalysisResult objects without
    interrupting or terminating the rest of the plan execution.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union
import numpy as np
import pandas as pd

from app.schemas.analysis_schema import (
    AnalysisError,
    AnalysisResult,
    PlanAnalysisResponse,
)
from app.schemas.planner_schema import InvestigationTask
from app.statistics.correlation import execute_correlation
from app.statistics.group_comparison import (
    execute_two_group_comparison,
    execute_multi_group_comparison,
)
from app.statistics.categorical import execute_categorical_association
from app.statistics.sensitivity import execute_outlier_sensitivity
from app.statistics.descriptive import (
    execute_distribution_analysis,
    execute_missingness_audit,
)
from app.statistics.classification import execute_binary_classification
from app.statistics.clustering import execute_kmeans_clustering
from app.statistics.text_classification import execute_text_classification

logger = logging.getLogger("datainvest.analysis_agent")


class AnalysisAgent:
    """
    Execution engine that takes structured investigation tasks from the Planner
    and runs them against a dataset using SciPy statistical tools.
    """

    AGENT_VERSION = "3.0"

    def execute_task(
        self,
        task: Union[InvestigationTask, dict[str, Any]],
        df: pd.DataFrame,
    ) -> AnalysisResult:
        """
        Execute a single investigation task with complete error isolation.
        """
        # Convert dict to model if needed
        if isinstance(task, dict):
            task_obj = InvestigationTask(**task)
        else:
            task_obj = task

        task_id = task_obj.task_id
        task_type = task_obj.task_type
        method = task_obj.method
        columns = task_obj.columns or []

        # Validate DataFrame
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return AnalysisResult(
                task_id=task_id,
                status="failed",
                analysis_type=task_type,
                method=method,
                columns=columns,
                sample_size=0,
                results={},
                warnings=[],
                error=AnalysisError(
                    code="EMPTY_DATASET",
                    message="The provided dataset is empty or invalid.",
                ),
                metadata={"engine": "SciPy/Python", "version": self.AGENT_VERSION},
            )

        # Dispatch task to statistical execution
        try:
            return self._dispatch_task(task_obj, df)
        except Exception as e:
            logger.exception(f"Unhandled error executing task {task_id}: {e}")
            return AnalysisResult(
                task_id=task_id,
                status="failed",
                analysis_type=task_type,
                method=method,
                columns=columns,
                sample_size=0,
                results={},
                warnings=[],
                error=AnalysisError(
                    code="UNHANDLED_EXCEPTION",
                    message=f"Task execution failed unexpectedly: {str(e)}",
                ),
                metadata={"engine": "SciPy/Python", "version": self.AGENT_VERSION},
            )

    def execute_plan(
        self,
        tasks: list[Union[InvestigationTask, dict[str, Any]]],
        df: pd.DataFrame,
        dataset_id: str = "dataset",
    ) -> PlanAnalysisResponse:
        """
        Execute an entire planned list of investigation tasks.
        Guarantees failure isolation across all tasks.
        """
        results: list[AnalysisResult] = []
        successful_count = 0
        failed_count = 0

        for task in tasks:
            result = self.execute_task(task, df)
            results.append(result)
            if result.status == "success":
                successful_count += 1
            else:
                failed_count += 1

        return PlanAnalysisResponse(
            dataset_id=dataset_id,
            total_tasks=len(tasks),
            successful_tasks=successful_count,
            failed_tasks=failed_count,
            results=results,
        )

    # ------------------------------------------------------------------
    # INTERNAL DISPATCH
    # ------------------------------------------------------------------

    def _dispatch_task(
        self,
        task: InvestigationTask,
        df: pd.DataFrame,
    ) -> AnalysisResult:
        task_id = task.task_id
        task_type = task.task_type
        method = task.method
        columns = list(task.columns or [])

        success = False
        res_data: dict[str, Any] = {}
        warnings: list[str] = []
        err_data: Optional[dict[str, str]] = None
        sample_size = 0

        # 1. Numerical vs Numerical (Correlation)
        if task_type == "numerical_vs_numerical":
            col_a = task.column_a or (columns[0] if len(columns) > 0 else "")
            col_b = task.column_b or (columns[1] if len(columns) > 1 else "")
            success, res_data, warnings, err_data = execute_correlation(
                df=df, column_a=col_a, column_b=col_b, method=method
            )
            if success:
                sample_size = res_data.get("sample_size", 0)

        # 2. Two Groups vs Numerical (Welch's t-test / Mann-Whitney U)
        elif task_type == "two_groups_vs_numerical":
            grp_col = task.group_column or (columns[0] if len(columns) > 0 else "")
            val_col = task.value_column or (columns[1] if len(columns) > 1 else "")
            success, res_data, warnings, err_data = execute_two_group_comparison(
                df=df, group_column=grp_col, value_column=val_col, method=method
            )
            if success:
                sample_size = res_data.get("sample_size", 0)

        # 3. Multiple Groups vs Numerical (ANOVA / Kruskal-Wallis)
        elif task_type == "multiple_groups_vs_numerical":
            grp_col = task.group_column or (columns[0] if len(columns) > 0 else "")
            val_col = task.value_column or (columns[1] if len(columns) > 1 else "")
            success, res_data, warnings, err_data = execute_multi_group_comparison(
                df=df, group_column=grp_col, value_column=val_col, method=method
            )
            if success:
                sample_size = res_data.get("sample_size", 0)

        # 4. Categorical vs Categorical (Chi-square & Cramér's V)
        elif task_type == "categorical_vs_categorical":
            col_a = task.column_a or (columns[0] if len(columns) > 0 else "")
            col_b = task.column_b or (columns[1] if len(columns) > 1 else "")
            success, res_data, warnings, err_data = execute_categorical_association(
                df=df, column_a=col_a, column_b=col_b, method=method
            )
            if success:
                sample_size = res_data.get("sample_size", 0)

        # 5. Outlier Sensitivity (IQR Trimming Comparison)
        elif task_type == "outlier_sensitivity":
            col_a = task.column_a or (columns[0] if len(columns) > 0 else "")
            success, res_data, warnings, err_data = execute_outlier_sensitivity(
                df=df, column_a=col_a, method=method
            )
            if success:
                sample_size = res_data.get("sample_size", 0)

        # 6. Distribution Analysis (Moments & Normality)
        elif task_type in ("distribution_analysis", "descriptive_statistics"):
            col_a = task.column_a or (columns[0] if len(columns) > 0 else "")
            success, res_data, warnings, err_data = execute_distribution_analysis(
                df=df, column_a=col_a, method=method
            )
            if success:
                sample_size = res_data.get("sample_size", 0)

        # 7. Missing Value Investigation
        elif task_type == "missing_value_investigation":
            success, res_data, warnings, err_data = execute_missingness_audit(
                df=df, columns=columns, method=method
            )
            if success:
                sample_size = res_data.get("total_rows", 0)

        # 8. Duplicate Investigation
        elif task_type == "duplicate_investigation":
            n_rows = len(df)
            n_dups = int(df.duplicated().sum())
            dup_pct = (n_dups / n_rows * 100.0) if n_rows > 0 else 0.0
            sample_size = n_rows
            success = True
            res_data = {
                "total_rows": n_rows,
                "duplicate_rows": n_dups,
                "duplicate_percentage": round(dup_pct, 2),
                "unique_rows": n_rows - n_dups,
                "has_duplicates": bool(n_dups > 0),
            }

        # 9. Binary Classification
        elif task_type == "binary_classification":
            target_col = task.column_a or (columns[0] if len(columns) > 0 else "")
            feature_cols = [c for c in columns if c != target_col] if len(columns) > 1 else None
            apply_cleaning = bool(
                "critic_on" in (method or "").lower()
                or (task.notes and any("critic_on" in n.lower() for n in task.notes))
            )
            # Default model is logistic_regression unless specified
            m_type = method if method and ("forest" in method.lower() or "boosting" in method.lower()) else "logistic_regression"
            success, res_data, warnings, err_data = execute_binary_classification(
                df=df,
                target_column=target_col,
                feature_columns=feature_cols,
                model_type=m_type,
                apply_critic_cleaning=apply_cleaning,
            )
            if success:
                sample_size = res_data.get("sample_size", 0)

        # 10. Clustering / Customer Segmentation
        elif task_type in ("clustering", "customer_segmentation"):
            feature_cols = columns if columns else None
            apply_cleaning = bool(
                "critic_on" in (method or "").lower()
                or (task.notes and any("critic_on" in n.lower() for n in task.notes))
            )
            k_val = 3
            if method:
                import re
                m = re.search(r"k\s*=\s*(\d+)", method.lower())
                if m:
                    k_val = int(m.group(1))
            success, res_data, warnings, err_data = execute_kmeans_clustering(
                df=df,
                feature_columns=feature_cols,
                k_clusters=k_val,
                apply_critic_cleaning=apply_cleaning,
            )
            if success:
                sample_size = res_data.get("sample_size", 0)

        # 11. Text Classification / Fake News
        elif task_type in ("text_classification", "fake_news_classification"):
            txt_cols = columns if columns else [c for c in ["title", "text"] if c in df.columns]
            label_col = task.column_a or "label"
            apply_cleaning = bool(
                "critic_on" in (method or "").lower()
                or (task.notes and any("critic_on" in n.lower() for n in task.notes))
            )
            success, res_data, warnings, err_data = execute_text_classification(
                df=df,
                text_columns=txt_cols,
                label_column=label_col,
                apply_critic_cleaning=apply_cleaning,
            )
            if success:
                sample_size = res_data.get("sample_size", 0)

        # 12. Fallback / Unrecognized task type
        else:
            return AnalysisResult(
                task_id=task_id,
                status="failed",
                analysis_type=task_type,
                method=method,
                columns=columns,
                sample_size=0,
                results={},
                warnings=warnings,
                error=AnalysisError(
                    code="UNSUPPORTED_TASK_TYPE",
                    message=f"Task type '{task_type}' is not supported by the Analysis Agent.",
                ),
                metadata={"engine": "SciPy/Python", "version": self.AGENT_VERSION},
            )

        # Package final AnalysisResult
        error_obj = AnalysisError(**err_data) if err_data else None
        status_val = "success" if success else "failed"

        # Combine any task notes with generated warnings
        all_warnings = list(warnings)
        if task.notes:
            for note in task.notes:
                if note not in all_warnings:
                    all_warnings.append(note)

        # Determine effective method stamped from statistical execution if present
        effective_method = res_data.get("method") or method

        # Generate dynamic visual evidence for successful tasks
        visual_evidence = None
        if status_val == "success":
            try:
                from app.services.visualization_service import get_visualization_service
                viz_svc = get_visualization_service()

                # Prioritize analyzed variable columns from computed results to maintain strict alignment
                analyzed_cols = []
                if "column_a" in res_data and "column_b" in res_data:
                    analyzed_cols = [res_data["column_a"], res_data["column_b"]]
                elif "group_column" in res_data and "value_column" in res_data:
                    analyzed_cols = [res_data["group_column"], res_data["value_column"]]
                elif "column" in res_data:
                    analyzed_cols = [res_data["column"]]

                visual_evidence = viz_svc.generate_finding_visualization(
                    task_id=task_id,
                    analysis_type=task_type,
                    method=effective_method,
                    columns=analyzed_cols or columns,
                    df=df,
                    results=res_data,
                )
            except Exception as e:
                logger.warning("Could not generate visual evidence for task %s: %s", task_id, e)

        return AnalysisResult(
            task_id=task_id,
            status=status_val,
            analysis_type=task_type,
            method=effective_method,
            columns=columns,
            sample_size=sample_size,
            results=res_data,
            warnings=all_warnings,
            error=error_obj,
            visual_evidence=visual_evidence,
            metadata={
                "engine": "SciPy/Python",
                "version": self.AGENT_VERSION,
                "task_priority": task.priority,
                "task_title": task.task,
            },
        )
