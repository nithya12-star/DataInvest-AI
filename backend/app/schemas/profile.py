"""
backend/app/schemas/profile.py
================================
Pydantic v2 response schemas for DataInvest AI — Phase 1 Dataset Profiler.
Provides structured, type-safe data contracts for frontend and future agents.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class OverviewSchema(BaseModel):
    n_rows: int = Field(description="Total number of rows in the dataset")
    n_cols: int = Field(description="Total number of columns in the dataset")
    column_names: list[str] = Field(description="Ordered list of column names")
    dtypes: dict[str, str] = Field(description="Detected data type string for each column")
    memory_usage_bytes: int = Field(description="Estimated memory consumption in bytes")


class ColumnMissingSchema(BaseModel):
    missing_count: int = Field(description="Number of missing (null/NaN) values in the column")
    missing_pct: float = Field(description="Percentage of missing values in the column (0-100)")


class MissingStatsSchema(BaseModel):
    per_column: dict[str, ColumnMissingSchema] = Field(description="Missing stats breakdown per column")
    total_missing: int = Field(description="Total count of missing cells across the entire dataset")
    total_cells: int = Field(description="Total count of cells (n_rows * n_cols)")
    total_missing_pct: float = Field(description="Overall percentage of missing cells (0-100)")
    provenance: Optional[dict[str, Any]] = Field(default=None, description="Provenance distinguishing raw sentinels from interpreted missing values")


class DuplicatesSchema(BaseModel):
    n_duplicates: int = Field(description="Number of completely identical duplicate rows")
    duplicate_pct: float = Field(description="Percentage of duplicate rows (0-100)")


class ColumnCardinalitySchema(BaseModel):
    n_unique: Optional[int] = Field(default=None, description="Count of distinct non-null values")
    is_constant: bool = Field(description="True if column has <= 1 distinct value")
    is_high_cardinality: bool = Field(description="True if categorical column has >50% unique values")


class CardinalitySchema(BaseModel):
    per_column: dict[str, ColumnCardinalitySchema] = Field(description="Cardinality breakdown per column")
    constant_columns: list[str] = Field(description="List of constant column names")
    high_cardinality_columns: list[str] = Field(description="List of high-cardinality categorical column names")


class NumericalStatSchema(BaseModel):
    n_valid: int = Field(description="Count of non-null observations in the column")
    mean: Optional[float] = Field(default=None, description="Arithmetic mean")
    median: Optional[float] = Field(default=None, description="Median (50th percentile)")
    std: Optional[float] = Field(default=None, description="Sample standard deviation")
    min: Optional[float] = Field(default=None, description="Minimum observed value")
    max: Optional[float] = Field(default=None, description="Maximum observed value")
    q1: Optional[float] = Field(default=None, description="First quartile (25th percentile)")
    q3: Optional[float] = Field(default=None, description="Third quartile (75th percentile)")
    iqr: Optional[float] = Field(default=None, description="Interquartile Range (Q3 - Q1)")
    skewness: Optional[float] = Field(default=None, description="Sample skewness")


class OutlierStatSchema(BaseModel):
    n_potential_outliers: Optional[int] = Field(default=None, description="Count of potential outliers via 1.5*IQR")
    outlier_pct: Optional[float] = Field(default=None, description="Percentage of valid observations flagged as potential outliers")
    lower_fence: Optional[float] = Field(default=None, description="Lower fence: Q1 - 1.5 * IQR")
    upper_fence: Optional[float] = Field(default=None, description="Upper fence: Q3 + 1.5 * IQR")
    note: Optional[str] = Field(default=None, description="Contextual note or explanation")


class ProfileResponse(BaseModel):
    overview: OverviewSchema
    missing: MissingStatsSchema
    duplicates: DuplicatesSchema
    column_types: dict[str, str] = Field(description="Column category mapping: 'numerical', 'categorical', or 'datetime'")
    cardinality: CardinalitySchema
    numerical_stats: dict[str, NumericalStatSchema] = Field(description="Descriptive statistics for each numerical column")
    outliers: dict[str, OutlierStatSchema] = Field(description="Potential outlier statistics for numerical columns")
    datetime_cols: list[str] = Field(description="Columns identified as datetime-like")
    filename: Optional[str] = Field(default=None, description="Uploaded file name")
    dataset_id: Optional[str] = Field(default=None, description="Identifier for dataset retrieval in analysis")
    preview_rows: Optional[list[dict[str, Any]]] = Field(default=None, description="First rows of the dataset as clean JSON records")
    preprocessing_rules: Optional[dict[str, Any]] = Field(default=None, description="Preprocessing and sentinel rules active during profiling")


class ErrorResponse(BaseModel):
    error: str = Field(description="Human-readable error description")


class HealthResponse(BaseModel):
    status: str = Field(description="Health status indicator ('healthy')")


class RootResponse(BaseModel):
    status: str
    app: str
    phase: int
