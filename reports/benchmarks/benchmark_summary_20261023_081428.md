# DataInvest AI — Cross-Dataset Benchmark Summary

**Date:** 2026-10-23 08:14:28 UTC
**Execution Mode:** Offline Deterministic Fixture Replay (Verification Validation)
**Offline Fixture Validation:** Yes (Deterministic Offline Testing)

> [!NOTE]
> This benchmark harness validates runtime instrumentation, Critic rule-level metrics,
> and pipeline reproducibility across multiple real-world datasets.
> Shared planning/analysis time is isolated from proposed Critic and reinvestigation overhead.

## Consolidated Benchmark Matrix

| Dataset | Dimensions | Status | Tasks | Base Score | Prop Score | Delta | Reinv Req | Revised | Cochran | Normality | Leverage | Shared Time (s) | Critic+Reinv Overhead (s) | Total Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sample.csv` | 60x8 | **SUCCESS** | 14 | 35.9 | 91.7 | +55.8 | 0 | 0 | N/A | 0 | 0 | 0.543 | 0.003 | 0.557 |
| `student-mat.csv` | 395x33 | **SUCCESS** | 15 | 35.2 | 89.7 | +54.5 | 0 | 0 | N/A | 0 | 1 | 0.160 | 0.013 | 0.236 |
| `winequality-red.csv` | 1599x12 | **SUCCESS** | 12 | 33.5 | 90.3 | +56.8 | 0 | 0 | N/A | 0 | 2 | 0.196 | 0.042 | 0.308 |
| `processed.cleveland.txt` | 303x14 | **SUCCESS** | 13 | 32.9 | 88.5 | +55.6 | 0 | 0 | N/A | 0 | 2 | 0.136 | 0.003 | 0.150 |

## Execution Notes & Observations

All target datasets completed pipeline execution without failure.

### Statistical Rule Audit Breakdown
- **Cochran (1954) Guidelines**: Evaluates contingency tables for expected counts < 1.0 or >20% cells below 5.0. Recorded as N/A when no categorical tests were evaluated.
- **Normality & Distributional Skewness**: Flags severe skewness (>1.5) or small-group distribution violations, triggering Mann-Whitney or Kruskal-Wallis reinvestigations.
- **Residual Leverage & Outlier Diagnostics**: Flags bivariate outlier points (|z| > 3.0) compromising Pearson linearity.

### Fairness in Runtime Instrumentation
- **Shared Phase Time**: Measures planning and Phase 3 statistical execution. It is shared across both branches and never attributed to either baseline or proposed execution alone.
- **Proposed Overhead**: Strictly isolates Phase 4 Critic verification and Phase 5 controlled reinvestigation.
