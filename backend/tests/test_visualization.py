"""
backend/tests/test_visualization.py
===================================
Unit test suite for Phase 4: Dynamic Graphs & Visual Evidence.

Tests:
1. Dynamic Plotly graph generation across all finding types:
   - Numerical vs Numerical (Scatter plot + Trendline)
   - Two Groups vs Numerical (Side-by-side Box plot)
   - Multi-Group vs Numerical (Multi-box plot)
   - Categorical vs Categorical (Contingency Grouped Bar chart)
   - Outlier Sensitivity (Distribution Histogram with IQR fences)
2. Proper association with finding metadata and computed statistics.
3. Defensible programmatic visual diagnostics.
4. Scientific honesty: explicit confirmation that no vision model or visual inspection is falsely claimed.
5. Association of visual evidence with AnalysisResult, VerifiedFinding, and ReinvestigationRecord.
6. Graph regeneration on follow-up re-investigation.
"""

import pytest
import numpy as np
import pandas as pd

from app.schemas.analysis_schema import AnalysisResult
from app.schemas.planner_schema import InvestigationTask
from app.agents.analysis_agent import AnalysisAgent
from app.services.visualization_service import VisualizationService, get_visualization_service
from app.services.verification_service import VerificationService
from app.schemas.critic_schema import CriticResult
from app.schemas.verification_schema import VerificationRequest
from app.services.dataset_store import store_dataset, clear_cache


@pytest.fixture(autouse=True)
def clean_store():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def viz_service():
    return VisualizationService()


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 80
    experience = np.linspace(1, 15, n)
    salary = 40000 + 4500 * experience + np.random.normal(0, 3000, n)
    # Add 2 extreme outlier points
    salary[0] = 500000.0
    salary[-1] = -50000.0

    dept = np.random.choice(["Sales", "Engineering", "Marketing"], size=n)
    education = np.random.choice(["BSc", "MSc", "PhD"], size=n)

    return pd.DataFrame({
        "experience": experience,
        "salary": salary,
        "dept": dept,
        "education": education,
    })


def test_scatter_plot_generation_numerical_vs_numerical(viz_service, sample_data):
    """Correlation finding generates scatter plot with trendline and programmatic residual diagnostics."""
    ev = viz_service.generate_finding_visualization(
        task_id="t_corr",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["experience", "salary"],
        df=sample_data,
        results={"correlation_coefficient": 0.85, "sample_size": len(sample_data)},
    )

    assert ev is not None
    assert ev.graph_id == "graph_t_corr_scatter"
    assert ev.figure_type == "scatter"
    assert "experience" in ev.title and "salary" in ev.title
    assert ev.figure_spec is not None
    assert "data" in ev.figure_spec

    # Check programmatic visual diagnostics
    diag = ev.visual_diagnostics
    assert diag.vision_model_used is False
    assert diag.visual_inspection_claimed is False
    assert "Programmatic visual diagnostics used" in diag.limitation_notice
    assert len(diag.visual_notes) > 0


def test_two_group_box_plot_generation(viz_service, sample_data):
    """Two-group comparison generates side-by-side box plots with IQR overlap estimation."""
    two_grp_df = sample_data[sample_data["dept"].isin(["Engineering", "Sales"])]
    ev = viz_service.generate_finding_visualization(
        task_id="t_group",
        analysis_type="two_groups_vs_numerical",
        method="Welch's t-test",
        columns=["dept", "salary"],
        df=two_grp_df,
        results={"mean_difference": 12500.0},
    )

    assert ev is not None
    assert ev.graph_id == "graph_t_group_twogroup_box"
    assert ev.figure_type == "box"
    assert ev.visual_diagnostics.vision_model_used is False
    assert ev.visual_diagnostics.visual_inspection_claimed is False
    assert ev.visual_diagnostics.iqr_overlap_ratio is not None


def test_multi_group_box_plot_generation(viz_service, sample_data):
    """Multi-group ANOVA generates multi-category box plot."""
    ev = viz_service.generate_finding_visualization(
        task_id="t_multi",
        analysis_type="multiple_groups_vs_numerical",
        method="One-way ANOVA",
        columns=["dept", "salary"],
        df=sample_data,
        results={"F_statistic": 8.42, "num_groups": 3},
    )

    assert ev is not None
    assert ev.figure_type == "multi_box"
    assert ev.data_summary.get("k_groups") == 3
    assert ev.visual_diagnostics.vision_model_used is False


def test_categorical_bar_chart_generation(viz_service, sample_data):
    """Chi-square categorical association generates grouped bar chart with cell sparsity diagnostics."""
    ev = viz_service.generate_finding_visualization(
        task_id="t_cat",
        analysis_type="categorical_vs_categorical",
        method="Chi-square test of independence",
        columns=["dept", "education"],
        df=sample_data,
        results={"chi2_statistic": 5.21},
    )

    assert ev is not None
    assert ev.figure_type == "grouped_bar"
    assert ev.visual_diagnostics.sparse_categories_count >= 0
    assert ev.visual_diagnostics.vision_model_used is False


def test_distribution_histogram_generation(viz_service, sample_data):
    """Outlier sensitivity generates distribution histogram with marked IQR fences."""
    ev = viz_service.generate_finding_visualization(
        task_id="t_out",
        analysis_type="outlier_sensitivity",
        method="IQR-based sensitivity analysis",
        columns=["salary"],
        df=sample_data,
        results={"outlier_count": 2},
    )

    assert ev is not None
    assert ev.figure_type == "distribution_histogram"
    assert ev.visual_diagnostics.outlier_points_count >= 1
    assert ev.visual_diagnostics.vision_model_used is False


def test_analysis_agent_attaches_visual_evidence(sample_data):
    """AnalysisAgent automatically attaches dynamic VisualEvidence to successful AnalysisResult."""
    agent = AnalysisAgent()
    task = InvestigationTask(
        task_id="task_corr_1",
        task_type="numerical_vs_numerical",
        task="Investigate correlation between experience and salary",
        description="Run Pearson correlation",
        method="Pearson correlation",
        columns=["experience", "salary"],
        reason="Evaluate linear association between tenure and compensation",
        priority="high",
    )

    result = agent.execute_task(task, sample_data)
    assert result.status == "success"
    assert result.visual_evidence is not None
    assert result.visual_evidence.task_id == "task_corr_1"
    assert result.visual_evidence.figure_type == "scatter"
    assert result.visual_evidence.visual_diagnostics.vision_model_used is False


def test_verification_service_regenerates_and_preserves_graph_references(sample_data):
    """VerificationService preserves original visual evidence and regenerates follow-up graph on re-investigation."""
    store_dataset("test_ds", sample_data)
    verif_svc = VerificationService()

    # Create original analysis with scatter plot
    viz_svc = get_visualization_service()
    orig_viz = viz_svc.generate_finding_visualization(
        task_id="t1",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["experience", "salary"],
        df=sample_data,
        results={"correlation_coefficient": 0.85},
    )

    orig_analysis = AnalysisResult(
        task_id="t1",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["experience", "salary"],
        sample_size=len(sample_data),
        results={
            "correlation_coefficient": 0.85,
            "p_value": 0.0001,
            "sample_size": len(sample_data),
            "p_value_below_0_05": True,
        },
        visual_evidence=orig_viz,
    )

    # Critic requests re-investigation
    critic_rev = CriticResult(
        task_id="t1",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        variables=["experience", "salary"],
        status="REQUIRES_REINVESTIGATION",
        confidence=0.85,
        confidence_level="HIGH",
        summary="Extreme outliers detected; Pearson may be compromised.",
        evidence=["Outliers present"],
        statistical_validity="Valid",
        method_assessment="Linearity questionable",
        recommendation="Run Spearman rank correlation",
        requires_reinvestigation=True,
        recommended_method="Spearman rank correlation",
        reason_codes=["RECOMMEND_SPEARMAN_RANK"],
    )

    request = VerificationRequest(
        dataset_id="test_ds",
        analysis_response=None,
        critic_response=None,
        dataset_profile=None,
    )
    # Mock analysis_map and critic_map execution through process_verification
    from app.schemas.critic_schema import CriticResponse, CriticOverallSummary
    from app.schemas.analysis_schema import PlanAnalysisResponse

    request.analysis_response = PlanAnalysisResponse(
        dataset_id="test_ds",
        total_tasks=1,
        successful_tasks=1,
        failed_tasks=0,
        results=[orig_analysis],
    )
    request.critic_response = CriticResponse(
        dataset_id="test_ds",
        summary=CriticOverallSummary(
            total_results=1, accepted=0, questionable=0, rejected=0,
            requires_reinvestigation=1, overall_status="Requires re-investigation",
            critical_issues=[], key_verified_insights=[],
        ),
        reviews=[critic_rev],
    )

    response = verif_svc.process_verification(request)

    assert len(response.reinvestigations) == 1
    record = response.reinvestigations[0]
    # Check graph preservation
    assert record.original_graph_id is not None
    assert record.reinvestigated_graph_id is not None
    assert record.original_visual_evidence is not None
    assert record.reinvestigated_visual_evidence is not None
    assert record.original_graph_id != record.reinvestigated_graph_id

    # Check verified finding has associated visual evidence
    assert len(response.all_findings) == 1
    vf = response.all_findings[0]
    assert vf.visual_evidence is not None
    assert vf.graph_id is not None
