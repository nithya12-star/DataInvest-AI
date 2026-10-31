# FINAL EVALUATION — EXPERIMENT 5 OF 5: ADULT CENSUS INCOME REPORT

**Project:** DataInvest AI  
**Dataset:** `data/adult.txt` (UCI Adult Census Income Benchmark)  
**Execution Date:** 2026-10-25 (UTC 12:16:48)  
**Evaluation Verdict:** **PASS**  
**Authoritative Working Root:** `C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\Major_Project`

---

## 1. Executive Summary

Experiment 5 of 5 evaluated the DataInvest AI pipeline on the benchmark dataset `data/adult.txt` (Kohavi, 1996; UCI Machine Learning Repository). This evaluation completed the multi-dataset research evaluation, validating the framework on a high-cardinality demographic dataset (32,561 rows, 15 columns) featuring substantial missingness and zero-inflated financial distributions.

Key achievements:
- **Single Outbound Planner Request:** Exactly ONE request was made to `gemini-3.5-flash-lite`. Zero retries, zero subsequent external API calls.
- **Architectural Parity:** The baseline and proposed branches shared the identical dataset profile, genuine LLM-generated plan (5 tasks), and initial statistical analysis.
- **Headerless Ingestion & Missingness Preservation:** The benchmark schema (`VERIFIED_BENCHMARK_COLUMNS`) correctly mapped all 15 features. The 4,262 missing values (`?` markers across `workclass`, `occupation`, and `native_country`) were parsed to NaN and handled cleanly via pairwise complete observation masks without deleting any of the 32,561 records.
- **Honest Outlier & Zero-Inflation Retention:** The Critic and Verification services flagged extreme zero-inflation and skewness in capital gains/losses (-100.0% trim shift), honestly retaining `finding_5` as `QUESTIONABLE` with explicit reason codes (`SUBSTANTIAL_OUTLIER_INFLUENCE`, `QUESTIONABLE_FINDING`).
- **Non-Destructive Persistence:** Both branch run artifacts were persisted under unique IDs without modifying historical files.

### Primary Artifact Identifiers:
- **Experiment ID:** `exp_20261025_121648_adulttxt_177195`
- **Baseline Run ID:** `run_20261025_121648_adulttxt_baseline_177195`
- **Proposed Run ID:** `run_20261025_121648_adulttxt_proposed_177195`
- **Dataset File:** `data/adult.txt` (32,561 rows x 15 columns)

---

## 2. Dataset Dimensions & Preprocessing Profile

`data/adult.txt` contains 1994 U.S. Census Bureau records extracted by Ronny Kohavi to predict whether an individual's annual income exceeds $50,000.

| Dataset Dimension | Characteristic | Notes |
| :--- | :--- | :--- |
| **Dimensions** | 32,561 rows x 15 columns | Large-scale demographic benchmark. Total cells: 488,415. |
| **Format & Header** | Headerless CSV/comma-separated | Mapped via `upload_service.VERIFIED_BENCHMARK_COLUMNS["adult.txt"]`. |
| **Assigned Columns** | `age`, `workclass`, `fnlwgt`, `education`, `education_num`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `capital_gain`, `capital_loss`, `hours_per_week`, `native_country`, `income` | 6 Numerical, 9 Categorical. |
| **Missing Values** | 4,262 missing cells (0.87% of cells) | Represented as `?` in raw text: `workclass` (1,836), `occupation` (1,843), `native_country` (583). Successfully parsed to NaN and handled cleanly via pairwise complete observation masks without row dropping. |
| **Duplicate Observations** | 24 duplicate rows (0.07%) | Natural census profile duplications preserved intact. |
| **Whitespace Handling** | Stripped leading whitespace | Managed via `skipinitialspace=True` in `upload_service.parse_csv`. |
| **Target Variable** | `income` | Binary categorical target: `"<=50K"` (75.9%) vs. `">50K"` (24.1%). |
| **Memory Footprint** | ~11.5 MB in memory | Processed efficiently with sub-second execution overhead. |

---

## 3. Planner Execution, Latency & Resource Utilization

| Metric / Parameter | Observed Value | Validation Notes |
| :--- | :---: | :--- |
| **Planner Model** | `gemini-3.5-flash-lite` | Configured default model. |
| **Planner Request Count** | **1** | Tracked via `InstrumentedGeminiClient` at `2026-10-25T12:16:48.818583Z`. |
| **Planner Response Status** | `HTTP/1.1 200 OK` | Succeeded in **5.05s** (prompt length: 2,654 characters). Zero retries. |
| **Secondary LLM Calls** | **0** | Critic, verification, insights, and evaluation ran 100% locally offline. |
| **Initial Analysis Latency** | 0.4187 s | SciPy statistical engine executed across 32,561 rows in ~419 ms. |
| **Shared Duration** | **5.4732 s** | **Identical shared cost across both branches.** |
| **Baseline Branch Latency** | 0.0003 s | Evaluation calculation only. |
| **Proposed Branch Overhead** | **0.0578 s** | Critic review (0.0003s) + Verification/Reinvestigation (0.0575s). Overhead: ~58 ms. |
| **Total Experiment Monotonic Time** | **6.3042 s** | Complete execution including two-branch JSON serialization and disk persistence. |

---

## 4. Plan-Sharing and Initial Analysis Equality Check

**Research Objective:**  
`"Investigate how individual demographics (age, education_num, race, sex), employment characteristics (workclass, occupation, hours_per_week), and financial indicators (capital_gain, capital_loss) relate to high annual income status (>50K vs <=50K)."`

The Planner formulated 5 comprehensive investigation tasks:

| Task ID | Task Type | Configured Method | Assigned Columns | Execution Status |
| :--- | :--- | :--- | :--- | :---: |
| `task_1` | `missing_value_investigation` | Missing value summary and pattern analysis | `['workclass', 'occupation', 'native_country']` | `success` |
| `task_2` | `descriptive_statistics` | Summary statistics computation | `['age', 'education_num', 'hours_per_week', 'capital_gain', 'capital_loss']` | `success` |
| `task_3` | `two_groups_vs_numerical` | Independent samples t-test / Mann-Whitney U test | `['income', 'age', 'education_num', 'hours_per_week']` | `success` |
| `task_4` | `categorical_vs_categorical` | Chi-square test of independence | `['income', 'race', 'sex', 'workclass', 'occupation']` | `success` |
| `task_5` | `outlier_sensitivity` | IQR analysis and median absolute deviation | `['capital_gain', 'capital_loss', 'hours_per_week']` | `success` |

- **Plan Equality:** `base.plan == prop.plan` (**True**).
- **Analysis Equality:** `base.analysis == prop.analysis` (**True**).
- **Baseline Architectural Purity:** Confirmed `critic=None`, `verification=None`, `insights=None`.

---

## 5. Critic Review and Verification Audit

The proposed pipeline audited all 5 findings and evaluated methodological assumptions:

| Finding ID | Task ID | Evaluated Variables | Primary Statistic & p-value | Critic Assessment & Concerns | Final Verification Status | Reason Codes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`finding_1`** | `task_1` | `['workclass', 'occupation']` | Missingness overview (4,262 cells) | `ACCEPTED` (Missingness documented; completeness acceptable) | `VERIFIED` | `ACCEPTABLE_COMPLETENESS`, `ADEQUATE_SAMPLE_SIZE`, `DIRECTLY_VERIFIED` |
| **`finding_2`** | `task_2` | `['age', 'education_num']` | Summary descriptives | `ACCEPTED` (Skewness in capital gains/losses documented) | `VERIFIED` | `NON_NORMAL_DISTRIBUTION`, `ADEQUATE_SAMPLE_SIZE`, `DIRECTLY_VERIFIED` |
| **`finding_3`** | `task_3` | `['income', 'age']` | Mann-Whitney $U = 6.12 \times 10^7, p = 0.0$ | `ACCEPTED` (Highly significant group difference; older age associated with >50K income) | `VERIFIED` | `GROUP_COMPARISON_VALIDATED`, `SIGNIFICANT_GROUP_DIFFERENCE`, `ADEQUATE_SAMPLE_SIZE`, `DIRECTLY_VERIFIED` |
| **`finding_4`** | `task_4` | `['race', 'income']` | $\chi^2 = 330.92, p = 2.31 \times 10^{-70}, V = 0.101$ | `ACCEPTED` (Cochran condition satisfied: 0 cells < 5; significant association) | `VERIFIED` | `COCHRAN_CONDITION_SATISFIED`, `SIGNIFICANT_CATEGORICAL_ASSOCIATION`, `ADEQUATE_SAMPLE_SIZE`, `DIRECTLY_VERIFIED` |
| **`finding_5`** | `task_5` | `['capital_gain', 'capital_loss']` | Mean shift = -100.0% on outlier trim | `QUESTIONABLE` (Severe zero-inflation: >90% of respondents report 0 capital gain; trimming outliers drops mean by -100%) | `QUESTIONABLE` | `SUBSTANTIAL_OUTLIER_INFLUENCE`, `ADEQUATE_SAMPLE_SIZE`, `QUESTIONABLE_FINDING` |

### Key Statistical Validations:
1. **Reinvestigation Trigger Count:** 0 re-analyses were required because Critic directly validated `task_1`, `task_2`, `task_3`, and `task_4` under non-parametric Mann-Whitney and Chi-square frameworks.
2. **Defensive Verification of Fragile Finding:** `task_5` was flagged for extreme zero-inflation and outlier sensitivity (trimming outliers drops mean capital gain by -100.0%). The system **did not falsely verify** this finding; it correctly retained it as `QUESTIONABLE` with clear caveats.
3. **Traceability:** Variable pairs were strictly preserved and recorded.

---

## 6. Evaluation Metrics & Comparative Scorecard

### A. Dimensional Breakdown

| Evaluation Dimension | Baseline Score | Proposed Score | Absolute Delta | Analytical Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Execution Reliability** | 25.0 / 25 (100.0%) | 25.0 / 25 (100.0%) | 0.0 pts | 100% of planned tasks completed cleanly with 0% computational errors across 32,561 rows. |
| **Statistical Rigor** | 6.8 / 25 (27.2%) | 17.5 / 25 (70.0%) | **+10.7 pts** | Proposed audited 5/5 tasks, checked Cochran conditions, and flagged zero-inflation instability. |
| **Verification & Grounding** | 0.0 / 25 (0.0%) | 17.0 / 25 (68.0%) | **+17.0 pts** | Baseline omitted verification. Proposed yielded 4 verified findings (80% yield) and 1 caveated questionable finding. |
| **Insight Actionability** | 0.0 / 25 (0.0%) | 18.0 / 25 (72.0%) | **+18.0 pts** | Baseline produced 0 insights. Proposed synthesized 2 verified insights with 100% non-causal phrasing compliance. |
| **Overall Composite Score** | **31.8 / 100 (Grade C)** | **77.4 / 100 (Grade B+)** | **+45.6 pts** | Significant increase in verified truth-grounding and methodological rigor. |

> [!IMPORTANT]
> **Composite Score Limitations:**  
> The composite score delta (+45.6 points) reflects the internal execution of Critic auditing and verified insight synthesis in the proposed branch, which are bypassed by design in the baseline branch. This score confirms that the proposed pipeline successfully executed its complete auditing lifecycle on a 32,000-row demographic dataset, but does NOT establish general superiority over AutoMind (IEEE Access 2026).

---

## 7. Limitations & Technical Notes

1. **Zero-Inflation in Financial Assets:**  
   In the Adult Census dataset, over 91% of respondents report `$0` in `capital_gain` and over 95% report `$0` in `capital_loss`. Applying parametric mean and standard deviation metrics to capital gains is misleading; the pipeline correctly identified this fragility and categorized finding 5 as `QUESTIONABLE`.
2. **Observational vs. Causal Interpretation:**  
   The strong demographic associations (`age` vs `income`, $U = 6.12 \times 10^7, p = 0.0$; `race` vs `income`, $\chi^2 = 330.92, p = 2.31 \times 10^{-70}$) reflect historical socioeconomic disparities in 1994 U.S. Census data. In accordance with research guidelines, the pipeline strictly applied non-causal observational phrasing (`m_insight_causality_safety = 100.0%`).
3. **Visualization Engine Row Ceiling Warning:**  
   Altair logged standard warnings on the 5,000-row embedded chart limit (`The number of rows in your dataset (32561) is greater than the maximum allowed (5000)`). The visualization engine handled this gracefully without interrupting statistical pipeline execution.

---

## 8. Artifact Deliverables Summary

1. **Baseline Run Artifact:**  
   [`backend/data/runs/run_20261025_121648_adulttxt_baseline_177195.json`](file:///C:/Users/SPURGE/MAJOR_PROJECT_DOCS/REVIEW%202/Major_Project/backend/data/runs/run_20261025_121648_adulttxt_baseline_177195.json)
2. **Proposed Run Artifact:**  
   [`backend/data/runs/run_20261025_121648_adulttxt_proposed_177195.json`](file:///C:/Users/SPURGE/MAJOR_PROJECT_DOCS/REVIEW%202/Major_Project/backend/data/runs/run_20261025_121648_adulttxt_proposed_177195.json)
3. **Evaluation Report:**  
   [`reports/FINAL_EVALUATION_DATASET_5.md`](file:///C:/Users/SPURGE/MAJOR_PROJECT_DOCS/REVIEW%202/Major_Project/reports/FINAL_EVALUATION_DATASET_5.md)

---

## 9. Final Conclusion

Experiment 5 of 5 on Adult Census Income completed with complete plan and analysis equality between branches, exactly one outbound Gemini Planner request, clean headerless schema mapping, honest preservation of 4,262 missing values, and proper qualification of zero-inflated variables as `QUESTIONABLE`.

**Experiment 5 Verdict: PASS**
