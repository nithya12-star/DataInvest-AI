# DataInvest AI — Cross-Dataset Benchmark Summary

**Date:** 2026-10-23 08:13:04 UTC
**Execution Mode:** Offline Deterministic Fixture Replay (Verification Validation)
**Offline Fixture Validation:** Yes (Deterministic Offline Testing)

> [!NOTE]
> This benchmark harness validates runtime instrumentation, Critic rule-level metrics,
> and pipeline reproducibility across multiple real-world datasets.
> Shared planning/analysis time is isolated from proposed Critic and reinvestigation overhead.

## Consolidated Benchmark Matrix

| Dataset | Dimensions | Status | Tasks | Base Score | Prop Score | Delta | Reinv Req | Revised | Cochran | Normality | Leverage | Shared Time (s) | Critic+Reinv Overhead (s) | Total Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sample.csv` | 60x8 | **FAILED** | 0 | - | - | - | 0 | 0 | N/A | N/A | N/A | - | - | - |
| `student-mat.csv` | 395x33 | **FAILED** | 0 | - | - | - | 0 | 0 | N/A | N/A | N/A | - | - | - |
| `winequality-red.csv` | 1599x12 | **FAILED** | 0 | - | - | - | 0 | 0 | N/A | N/A | N/A | - | - | - |
| `processed.cleveland.txt` | 303x14 | **FAILED** | 0 | - | - | - | 0 | 0 | N/A | N/A | N/A | - | - | - |

## Execution Notes & Observations

### Partial Failures / Unprocessed Datasets
- **sample.csv**: `Experiment execution failed: 2 validation errors for SelectedColumn
column
  Field required [type=missing, input_value={'column_name': 'age', 'r...data_type': 'numerical'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.14/v/missing
reason
  Field required [type=missing, input_value={'column_name': 'age', 'r...data_type': 'numerical'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.14/v/missing`
- **student-mat.csv**: `Experiment execution failed: 2 validation errors for SelectedColumn
column
  Field required [type=missing, input_value={'column_name': 'sex', 'r...ta_type': 'categorical'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.14/v/missing
reason
  Field required [type=missing, input_value={'column_name': 'sex', 'r...ta_type': 'categorical'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.14/v/missing`
- **winequality-red.csv**: `Experiment execution failed: 2 validation errors for SelectedColumn
column
  Field required [type=missing, input_value={'column_name': 'alcohol'...data_type': 'numerical'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.14/v/missing
reason
  Field required [type=missing, input_value={'column_name': 'alcohol'...data_type': 'numerical'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.14/v/missing`
- **processed.cleveland.txt**: `Experiment execution failed: 2 validation errors for SelectedColumn
column
  Field required [type=missing, input_value={'column_name': 'sex', 'r...ta_type': 'categorical'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.14/v/missing
reason
  Field required [type=missing, input_value={'column_name': 'sex', 'r...ta_type': 'categorical'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.14/v/missing`

### Statistical Rule Audit Breakdown
- **Cochran (1954) Guidelines**: Evaluates contingency tables for expected counts < 1.0 or >20% cells below 5.0. Recorded as N/A when no categorical tests were evaluated.
- **Normality & Distributional Skewness**: Flags severe skewness (>1.5) or small-group distribution violations, triggering Mann-Whitney or Kruskal-Wallis reinvestigations.
- **Residual Leverage & Outlier Diagnostics**: Flags bivariate outlier points (|z| > 3.0) compromising Pearson linearity.

### Fairness in Runtime Instrumentation
- **Shared Phase Time**: Measures planning and Phase 3 statistical execution. It is shared across both branches and never attributed to either baseline or proposed execution alone.
- **Proposed Overhead**: Strictly isolates Phase 4 Critic verification and Phase 5 controlled reinvestigation.
