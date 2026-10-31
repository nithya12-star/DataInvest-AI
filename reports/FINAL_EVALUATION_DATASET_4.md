# FINAL EVALUATION — EXPERIMENT 4 OF 5: CLEVELAND HEART DISEASE REPORT

**Project:** DataInvest AI  
**Dataset:** `data/processed.cleveland.txt` (Cleveland Heart Disease Benchmark)  
**Execution Date:** 2026-10-25 (UTC 12:12:03)  
**Evaluation Verdict:** **PASS**  
**Authoritative Working Root:** `C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\Major_Project`

---

## 1. Executive Summary

Experiment 4 of 5 evaluated the DataInvest AI pipeline on the clinical benchmark dataset `data/processed.cleveland.txt` (Detrano et al., 1989; UCI Machine Learning Repository). This evaluation examined the framework's behavior on smaller sample sizes (N = 303) with headerless schemas and missing-value markers (`?`).

Key achievements:
- **Single Outbound Planner Request:** Exactly ONE request was made to `gemini-3.5-flash-lite`. Zero retries, zero subsequent external API calls.
- **Architectural Parity:** The baseline and proposed branches shared the identical dataset profile, genuine LLM-generated plan (8 tasks), and initial statistical analysis.
- **Headerless Ingestion & Schema Binding:** The established benchmark schema (`VERIFIED_BENCHMARK_COLUMNS`) correctly mapped all 14 clinical features. Missing value markers (`?`) in `ca` (4) and `thal` (2) were parsed to NaN and handled cleanly via pairwise complete masks.
- **Null-Hypothesis Respect (Zero P-Hacking):** The statistical engine and Critic agent cleanly retained non-significant findings (e.g. resting blood pressure `trestbps` vs diagnosis, $p = 0.1425$; cholesterol `chol` vs diagnosis, $p = 0.0653$) without p-hacking or attempting spurious reinvestigations (`P_HACKING_PREVENTED`).
- **Controlled Reinvestigation:** Reinvestigation on multi-column correlation (`task_4`) strictly preserved its designated bivariate pair (`['age', 'thalach']`), confirming a robust negative monotonic association ($\rho = -0.3916, p = 1.51 \times 10^{-12}$).
- **Non-Destructive Persistence:** Both branch run artifacts were persisted under unique IDs without modifying historical files.

### Primary Artifact Identifiers:
- **Experiment ID:** `exp_20261025_121203_processedclevelandtxt_3e6141`
- **Baseline Run ID:** `run_20261025_121203_processedclevelandtxt_baseline_3e6141`
- **Proposed Run ID:** `run_20261025_121203_processedclevelandtxt_proposed_3e6141`
- **Dataset File:** `data/processed.cleveland.txt` (303 rows x 14 columns)

---

## 2. Dataset Dimensions & Preprocessing Profile

`data/processed.cleveland.txt` contains clinical records collected from the Cleveland Clinic Foundation for predicting the presence of coronary artery disease.

| Dataset Dimension | Characteristic | Notes |
| :--- | :--- | :--- |
| **Dimensions** | 303 rows x 14 columns | Clinical benchmark dataset (N = 303). |
| **Format & Header** | Headerless CSV/comma-separated | Column names mapped via `upload_service.VERIFIED_BENCHMARK_COLUMNS["processed.cleveland.txt"]`. |
| **Assigned Columns** | `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target` | 14 numerical features (integers and floats). |
| **Missing Values** | 6 missing values (0.14% of cells) | Represented as `?` in raw text: 4 in `ca` (vessels colored), 2 in `thal` (defect). Successfully parsed to NaN and handled via pairwise complete observation masks in `correlation.py` and `group_comparison.py`. |
| **Duplicate Observations** | 0 duplicate rows (0.0%) | Clean individual patient diagnostic profiles. |
| **Target Variable** | `target` | Diagnosis of heart disease (integer 0: absence; integers 1–4: presence). |
| **Memory Footprint** | ~33 KB in memory | Compact clinical dataset. |

---

## 3. Planner Execution, Latency & Resource Utilization

| Metric / Parameter | Observed Value | Validation Notes |
| :--- | :---: | :--- |
| **Planner Model** | `gemini-3.5-flash-lite` | Configured default model. |
| **Planner Request Count** | **1** | Tracked via `InstrumentedGeminiClient` at `2026-10-25T12:12:03.975222Z`. |
| **Planner Response Status** | `HTTP/1.1 200 OK` | Succeeded in **5.73s** (prompt length: 2,498 characters). Zero retries. |
| **Secondary LLM Calls** | **0** | Critic, verification, insights, and evaluation ran 100% locally offline. |
| **Initial Analysis Latency** | 0.4756 s | SciPy statistical engine executed across 8 tasks in ~475 ms. |
| **Shared Duration** | **6.2062 s** | **Identical shared cost across both branches.** |
| **Baseline Branch Latency** | 0.0003 s | Evaluation calculation only. |
| **Proposed Branch Overhead** | **0.0124 s** | Critic review (0.0004s) + Verification/Reinvestigation (0.0120s). Overhead: ~12 ms. |
| **Total Experiment Monotonic Time** | **6.6753 s** | End-to-end execution including two-branch JSON serialization and disk persistence. |

---

## 4. Plan-Sharing and Initial Analysis Equality Check

**Research Objective:**  
`"Investigate how patient clinical indicators (age, sex, chest pain type cp, resting blood pressure trestbps, serum cholesterol chol, maximum heart rate thalach, and exercise-induced angina exang) relate to heart disease diagnosis presence (target)."`

The Planner formulated 8 structured investigation tasks:

| Task ID | Task Type | Configured Method | Assigned Columns | Execution Status |
| :--- | :--- | :--- | :--- | :---: |
| `task_1` | `missing_value_investigation` | Missing value counts & percentage summary | `['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'target']` | `success` |
| `task_2` | `two_groups_vs_numerical` | Welch's t-test / Mann-Whitney U test | `['age', 'trestbps', 'chol', 'thalach', 'target']` | `success` |
| `task_2b` | `two_groups_vs_numerical` | Welch's t-test / Mann-Whitney U test | `['trestbps', 'target']` | `success` |
| `task_2c` | `two_groups_vs_numerical` | Welch's t-test / Mann-Whitney U test | `['chol', 'target']` | `success` |
| `task_2d` | `two_groups_vs_numerical` | Welch's t-test / Mann-Whitney U test | `['thalach', 'target']` | `success` |
| `task_3` | `categorical_vs_categorical` | Chi-square test of independence | `['sex', 'cp', 'exang', 'target']` | `success` |
| `task_4` | `numerical_vs_numerical` | Pearson and Spearman correlation coefficients | `['age', 'trestbps', 'chol', 'thalach', 'target']` | `success` |
| `task_5` | `outlier_sensitivity` | IQR-based outlier detection & sensitivity summary | `['age', 'trestbps', 'chol', 'thalach']` | `success` |

- **Plan Equality:** `base.plan == prop.plan` (**True**).
- **Analysis Equality:** `base.analysis == prop.analysis` (**True**).
- **Baseline Architectural Purity:** Confirmed `critic=None`, `verification=None`, `insights=None`.

---

## 5. Critic Review and Verification Audit

The proposed pipeline audited all 8 findings and evaluated methodological assumptions:

| Finding ID | Task ID | Evaluated Variables | Primary Statistic & p-value | Critic Assessment & Concerns | Final Verification Status | Reason Codes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`finding_1`** | `task_1` | `['target', 'sex']` | Missingness overview | `ACCEPTED` (Completeness acceptable) | `VERIFIED` | `ACCEPTABLE_COMPLETENESS`, `ADEQUATE_SAMPLE_SIZE`, `DIRECTLY_VERIFIED` |
| **`finding_2`** | `task_2` | `['target', 'age']` | Mann-Whitney $U = 1864.5, p = 0.0005$ | `ACCEPTED` (Significant group difference; older age associated with disease) | `VERIFIED` | `GROUP_COMPARISON_VALIDATED`, `VISUAL_OUTLIERS_DETECTED`, `SIGNIFICANT_GROUP_DIFFERENCE`, `ADEQUATE_SAMPLE_SIZE`, `DIRECTLY_VERIFIED` |
| **`finding_3`** | `task_2b` | `['target', 'trestbps']` | Mann-Whitney $U = 2492.0, p = 0.1425$ | `ACCEPTED` (Non-significant difference preserved honestly) | `VERIFIED` | `GROUP_COMPARISON_VALIDATED`, `NON_SIGNIFICANT_GROUP_DIFFERENCE`, `P_HACKING_PREVENTED`, `ADEQUATE_SAMPLE_SIZE`, `DIRECTLY_VERIFIED` |
| **`finding_4`** | `task_2c` | `['target', 'chol']` | Mann-Whitney $U = 2372.0, p = 0.0653$ | `ACCEPTED` (Marginally non-significant difference preserved honestly) | `VERIFIED` | `GROUP_COMPARISON_VALIDATED`, `VISUAL_OUTLIERS_DETECTED`, `NON_SIGNIFICANT_GROUP_DIFFERENCE`, `P_HACKING_PREVENTED`, `ADEQUATE_SAMPLE_SIZE`, `DIRECTLY_VERIFIED` |
| **`finding_5`** | `task_2d` | `['target', 'thalach']` | Mann-Whitney $U = 4778.0, p = 6.39 \times 10^{-9}$ | `ACCEPTED` (Highly significant group difference; disease associated with lower max heart rate) | `VERIFIED` | `GROUP_COMPARISON_VALIDATED`, `SIGNIFICANT_GROUP_DIFFERENCE`, `ADEQUATE_SAMPLE_SIZE`, `DIRECTLY_VERIFIED` |
| **`finding_6`** | `task_3` | `['sex', 'target']` | $\chi^2 = 23.43, p = 0.0001$ | `ACCEPTED` (Cochran condition satisfied: all expected cells $\ge 5$; significant association) | `VERIFIED` | `COCHRAN_CONDITION_SATISFIED`, `SIGNIFICANT_CATEGORICAL_ASSOCIATION`, `ADEQUATE_SAMPLE_SIZE`, `DIRECTLY_VERIFIED` |
| **`finding_7`** | `task_4` | `['age', 'thalach']` | Spearman $\rho = -0.3916, p = 1.51 \times 10^{-12}$ | `REQUIRES_REINVESTIGATION` (Skewed distributions in cholesterol/age; non-parametric check recommended) | `VERIFIED_AFTER_REINVESTIGATION` | `METHOD_VALIDATED`, `VISUAL_OUTLIERS_DETECTED`, `SIGNIFICANT_WEAK_EFFECT`, `ADEQUATE_SAMPLE_SIZE`, `REINVESTIGATION_RESOLVED` |
| **`finding_8`** | `task_5` | `['chol', 'trestbps']` | Minimal outlier shift | `ACCEPTED` (Outlier sensitivity minimal) | `VERIFIED` | `MINIMAL_OUTLIER_INFLUENCE`, `ADEQUATE_SAMPLE_SIZE`, `VISUAL_OUTLIERS_DETECTED`, `DIRECTLY_VERIFIED` |

### Key Statistical Validations:
1. **Target Variable Pair Preservation:**
   - Multi-column `task_4` (`['age', 'trestbps', 'chol', 'thalach', 'target']`) identified `['age', 'thalach']` as the primary correlation finding.
   - Reinvestigation evaluated strictly `['age', 'thalach']` via `Spearman rank correlation`. Variable pair matched 100%.
   - Negative direction was confirmed: $\rho = -0.3916$ ($p = 1.51 \times 10^{-12}$).
2. **P-Hacking Prevention:**
   - Findings 3 (`trestbps` vs `target`, $p = 0.1425$) and 4 (`chol` vs `target`, $p = 0.0653$) were not manipulated or discarded. The pipeline explicitly validated that failing to reject the null hypothesis is reported honestly (`P_HACKING_PREVENTED`).

---

## 6. Evaluation Metrics & Comparative Scorecard

### A. Dimensional Breakdown

| Evaluation Dimension | Baseline Score | Proposed Score | Absolute Delta | Analytical Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Execution Reliability** | 25.0 / 25 (100.0%) | 25.0 / 25 (100.0%) | 0.0 pts | 100% of planned tasks completed cleanly with 0% computational errors across 303 rows. |
| **Statistical Rigor** | 12.8 / 25 (51.2%) | 24.5 / 25 (98.0%) | **+11.7 pts** | Proposed audited 8/8 tasks, satisfied Cochran conditions, and resolved the flagged violation. |
| **Verification & Grounding** | 0.0 / 25 (0.0%) | 25.0 / 25 (100.0%) | **+25.0 pts** | Baseline omitted verification. Proposed yielded 8 verified findings (100% yield) with full deterministic provenance. |
| **Insight Actionability** | 0.0 / 25 (0.0%) | 18.0 / 25 (72.0%) | **+18.0 pts** | Baseline produced 0 insights. Proposed synthesized 4 verified insights with 100% non-causal phrasing compliance. |
| **Overall Composite Score** | **37.8 / 100 (Grade C)** | **92.5 / 100 (Grade A)** | **+54.7 pts** | Outstanding scientific rigor and truth-grounding. |

> [!IMPORTANT]
> **Composite Score Limitations:**  
> The composite score delta (+54.7 points) reflects the internal execution of Critic auditing and verified insight synthesis in the proposed branch, which are bypassed by design in the baseline branch. This score confirms that the proposed pipeline successfully executed its complete auditing lifecycle on clinical benchmark data, but does NOT establish general superiority over AutoMind (IEEE Access 2026).

---

## 7. Limitations & Technical Notes

1. **Association vs. Causation on Age and Heart Rate:**  
   The negative association between age and maximum achievable heart rate (`thalach`, $\rho = -0.3916, p < 10^{-11}$) reflects standard cardiovascular physiology (declining chronotropic capacity with age). It is an observational correlation; maximum heart rate is also confounded by beta-blocker medications commonly prescribed to heart disease patients.
2. **Small Clinical Sample Size (N = 303):**  
   While sufficient for bivariate group comparisons (Welch's t-test and Mann-Whitney U), subgroup analyses (e.g. female patients with typical angina) yield smaller cell counts. The system checked Cochran conditions to ensure statistical validity.
3. **Missing Value Imputation vs. Complete Cases:**  
   Six observations in `ca` and `thal` contained missing value markers (`?`). Pairwise complete observation masks safely excluded missing cases from specific tests without discarding entire patient rows from other clinical analyses.

---

## 8. Artifact Deliverables Summary

1. **Baseline Run Artifact:**  
   [`backend/data/runs/run_20261025_121203_processedclevelandtxt_baseline_3e6141.json`](file:///C:/Users/SPURGE/MAJOR_PROJECT_DOCS/REVIEW%202/Major_Project/backend/data/runs/run_20261025_121203_processedclevelandtxt_baseline_3e6141.json)
2. **Proposed Run Artifact:**  
   [`backend/data/runs/run_20261025_121203_processedclevelandtxt_proposed_3e6141.json`](file:///C:/Users/SPURGE/MAJOR_PROJECT_DOCS/REVIEW%202/Major_Project/backend/data/runs/run_20261025_121203_processedclevelandtxt_proposed_3e6141.json)
3. **Evaluation Report:**  
   [`reports/FINAL_EVALUATION_DATASET_4.md`](file:///C:/Users/SPURGE/MAJOR_PROJECT_DOCS/REVIEW%202/Major_Project/reports/FINAL_EVALUATION_DATASET_4.md)

---

## 9. Final Conclusion

Experiment 4 of 5 on Cleveland Heart Disease completed with complete plan and analysis equality between branches, exactly one outbound Gemini Planner request, clean headerless schema ingestion, honest reporting of non-significant findings, and strict variable-pair preservation during reinvestigation.

**Experiment 4 Verdict: PASS**
