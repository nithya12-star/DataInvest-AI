"""
backend/app/statistics
======================
Statistical tool layer for DataInvest AI — Phase 3.
Provides pure Python, SciPy, and NumPy implementations for all planned analytical methods.
Zero LLM calculation, zero hallucinated statistical values.
"""

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
from app.statistics.clustering import (
    execute_kmeans_clustering,
    compare_paired_clustering,
)
from app.statistics.text_classification import (
    execute_text_classification,
    compare_paired_text_classification,
)

__all__ = [
    "execute_correlation",
    "execute_two_group_comparison",
    "execute_multi_group_comparison",
    "execute_categorical_association",
    "execute_outlier_sensitivity",
    "execute_distribution_analysis",
    "execute_missingness_audit",
    "execute_kmeans_clustering",
    "compare_paired_clustering",
    "execute_text_classification",
    "compare_paired_text_classification",
]
