# DataInvest AI — Phase 2: Planner Agent Documentation 📋

## Overview

**Phase 2** introduces the **Planner Agent**, an intelligent orchestration component inspired by the AutoMind research paper (*"AutoMind: Automated Insight Discovery via Multi-Agent Navigation"*). 

The Planner takes the structured profile from **Phase 1** and a natural-language **user objective** to generate a prioritized, non-redundant statistical investigation plan.

```
CSV Upload
    ↓
Dataset Profiler (Phase 1)
    ↓
User Investigation Objective
    ↓
Planner Agent (Phase 2)
    ↓
Structured Investigation Tasks
    ↓
[Future Phase 3: Analysis Agent]
```

---

## Core Responsibilities & Constraints

### What the Planner Does
1. **Objective Interpretation**: Translates natural language questions (e.g. *"Find what affects selling price"*) into explicit target variables, candidate predictors, and analytical goals.
2. **Quality-Aware Column Selection**: Filters out constant columns, excessive missingness ($>60\%$), and high-cardinality non-numeric identifiers.
3. **Statistical Method Selection**: Recommends appropriate methods based on data shape, distribution skewness, and outliers:
   - **Numerical vs Numerical**: Pearson correlation or Spearman rank correlation.
   - **Two Groups vs Numerical**: Welch's t-test or Mann-Whitney U test.
   - **Multiple Groups vs Numerical**: One-way ANOVA or Kruskal-Wallis test.
   - **Categorical vs Categorical**: Chi-square test of independence with Cramér's V.
   - **Outlier Sensitivity**: IQR-based sensitivity analysis.
   - **Missing Data Investigation**: Missingness pattern and completeness audit.
4. **Non-Symmetric Task Deduplication**: Ensures pairs like $A \leftrightarrow B$ and $B \leftrightarrow A$ are never generated redundantly.
5. **Prioritization**: Ranks tasks into `high`, `medium`, and `low` based on objective relevance and statistical suitability.

### Strict Statistical Rules
- **No Statistical Calculations**: The Planner does **NOT** calculate correlations, p-values, t-statistics, F-statistics, or effect sizes.
- **No Fabricated Findings**: The Planner creates hypotheses and plans, never claiming conclusions before execution.

---

## API Specification

### Endpoint
`POST /api/plan`

### Request Payload (`PlannerRequest`)
```json
{
  "profile": { /* Phase 1 ProfileResponse object */ },
  "objective": "Understand factors affecting salary and performance score",
  "dataset_id": "sample.csv"
}
```

### Response Payload (`PlannerResponse`)
```json
{
  "objective": "Understand factors affecting salary and performance score",
  "interpreted_goal": "Investigate relationships, group differences, and potential predictive factors for 'salary' based on dataset characteristics.",
  "target_column": "salary",
  "selected_columns": [
    {
      "column": "salary",
      "data_type": "numerical",
      "reason": "Primary target variable identified from user objective",
      "is_target": true
    },
    {
      "column": "score",
      "data_type": "numerical",
      "reason": "Explicitly referenced in user investigation objective",
      "is_target": false
    }
  ],
  "analysis_tasks": [
    {
      "task_id": "task_1",
      "task_type": "missing_value_investigation",
      "task": "Investigate missing data mechanisms",
      "description": "Assess whether data is MCAR or MAR to guide imputation.",
      "method": "Missingness pattern analysis & completeness audit",
      "columns": ["id", "name", "age", "salary", "department", "join_date", "score", "notes"],
      "reason": "Unaddressed missing values can introduce bias into subsequent statistical estimations.",
      "priority": "high",
      "notes": ["Imputation or complete-case filtering may be required before regression/correlation."]
    },
    {
      "task_id": "task_4",
      "task_type": "numerical_vs_numerical",
      "task": "Analyze relationship: salary vs score",
      "description": "Measure direction and strength of association between 'salary' and 'score'.",
      "method": "Pearson correlation",
      "columns": ["salary", "score"],
      "column_a": "salary",
      "column_b": "score",
      "reason": "Both columns are numerical. Pearson correlation is suitable for approximately linear continuous variables.",
      "priority": "high",
      "notes": ["Missing-value handling may be required."]
    }
  ],
  "reasoning": [
    "Dataset dimensions: 60 rows and 8 columns.",
    "Identified 4 eligible numerical variable(s) and 1 eligible categorical variable(s).",
    "Objective focus: Investigate relationships, group differences, and potential predictive factors for 'salary'..."
  ],
  "warnings": [
    "Dataset contains 56 missing values (11.67% of total cells) across 1 column(s): notes."
  ],
  "expected_outputs": [
    "Statistical correlation matrix (Pearson/Spearman)",
    "Group comparison test statistics (t-test/ANOVA/non-parametric)",
    "Contingency independence metrics and effect sizes (Chi-square / Cramér's V)",
    "Outlier sensitivity impact assessments"
  ],
  "planner_mode": "deterministic"
}
```

---

## Future Compatibility (Phase 3 & Phase 4)

- **Phase 3 (Analysis Agent)**: Will iterate through the `analysis_tasks` array and execute the planned statistical methods via SciPy/Statsmodels without needing to re-parse the dataset.
- **Phase 4 (Critic Agent)**: Will compare the original `objective`, `analysis_tasks`, and Phase 3 statistical outputs to evaluate hypothesis support, identify gaps, and recommend re-investigation if needed.
