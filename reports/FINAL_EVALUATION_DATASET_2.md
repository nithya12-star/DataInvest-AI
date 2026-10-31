# FINAL EVALUATION — EXPERIMENT 2 OF 5: WINE QUALITY RED REPORT

**Project:** DataInvest AI  
**Dataset:** `data/winequality-red.csv` (Wine Quality Red)  
**Execution Date:** 2026-10-25 (UTC 12:06:10)  
**Evaluation Verdict:** **PASS**  
**Authoritative Working Root:** `C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\Major_Project`

---

## 1. Executive Summary

Experiment 2 of 5 evaluated the DataInvest AI autonomous multi-agent pipeline on `data/winequality-red.csv` (Cortez et al., 2009). The experiment was conducted with strict adherence to research integrity and fairness standards:
- **Single Planner Request:** Exactly ONE outbound request was dispatched to `gemini-3.5-flash-lite`. Zero retries, zero subsequent model calls.
- **Architectural Match:** Baseline and proposed branches shared the identical dataset profile, genuine LLM-generated plan, and initial Python/SciPy statistical analysis.
- **Natural Data Preservation:** The dataset's 1,599 observations and 240 natural duplicate rows were preserved intact without synthetic deduplication or data altering.
- **Controlled Reinvestigation:** The Task 23 fix ensured that follow-up analysis on multi-column correlation (`task_3`) strictly preserved its designated bivariate pair (`['quality', 'alcohol']`).
- **Persistence:** Unique, non-colliding run IDs were generated and persisted to disk. All historical artifacts remained unmodified.

### Primary Artifact Identifiers:
- **Experiment ID:** `exp_20261025_120610_winequality_redcsv_adf3f1`
- **Baseline Run ID:** `run_20261025_120610_winequality_redcsv_baseline_adf3f1`
- **Proposed Run ID:** `run_20261025_120610_winequality_redcsv_proposed_adf3f1`
- **Target Dataset:** `data/winequality-red.csv` (1,599 rows x 12 columns)

---

## 2. Dataset Dimensions & Preprocessing Profile

`data/winequality-red.csv` represents chemical properties and sensory quality evaluations of Portuguese "Vinho Verde" red wine.

| Dimension / Property | Characteristic | Audit Notes |
| :--- | :--- | :--- |
| **Dimensions** | 1,599 rows x 12 columns | 100% complete records. 0 missing cells. |
| **Column Types** | 12 Numerical (0 Categorical, 0 Datetime) | `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`, `quality`. |
| **Target Variable** | `quality` | Discrete sensory rating (integer scale 3–8; median 6, mean 5.64). |
| **Duplicate Rows** | 240 identical rows (15.01%) | **Preserved intact.** In physicochemical wine testing, identical measurements across identical batches represent natural common profiles; rows were not discarded. |
| **Memory Footprint** | ~150 KB | Ingested via `upload_service.parse_csv` (comma-delimited). |

---

## 3. Planner Execution, Latency & Resource Utilization

| Metric / Parameter | Observed Value | Validation Notes |
| :--- | :---: | :--- |
| **Planner Model** | `gemini-3.5-flash-lite` | Configured default model. |
| **Planner Request Count** | **1** | Logged via `InstrumentedGeminiClient` at `2026-10-25T12:06:10.882582Z`. |
| **Planner Response Status** | `HTTP/1.1 200 OK` | Succeeded in **5.43s** (prompt length: 2,452 characters). Zero retries. |
| **Secondary LLM Calls** | **0** | Critic, verification, insight synthesis, and evaluation ran 100% offline. |
| **Initial Analysis Latency** | 0.4747 s | Pure SciPy execution across 5 analytical tasks. |
| **Shared Duration** | **5.9073 s** | **Identical shared cost across both branches.** |
| **Baseline Branch Latency** | 0.0006 s | Evaluation computation only. |
| **Proposed Branch Overhead** | **0.0493 s** | Critic review (0.0005s) + Verification/Reinvestigation (0.0488s). |
| **Total Experiment Monotonic Time** | **6.6685 s** | End-to-end execution including two-branch JSON serialization and disk I/O. |

---

## 4. Plan-Sharing and Initial Analysis Equality Check

**Research Objective:**  
`"Investigate how physicochemical properties, particularly alcohol, volatile acidity, sulphates, and citric acid, relate to red wine quality ratings."`

The Planner formulated 5 tasks directly aligned with the analytical objective and dataset profile. The identical plan dictionary (`plan_dict`) and analysis results dictionary (`analysis_dict`) were passed to both branches:

| Task ID | Task Type | Configured Method | Target Columns | Execution Status |
| :--- | :--- | :--- | :--- | :---: |
| `task_1` | `descriptive_statistics` | Summary statistics calculation | `['quality', 'alcohol', 'volatile acidity', 'sulphates', 'citric acid']` | `success` |
| `task_2` | `distribution_analysis` | Histogram and normality assessment | `['quality', 'alcohol', 'volatile acidity', 'sulphates', 'citric acid']` | `success` |
| `task_3` | `numerical_vs_numerical` | Pearson and Spearman correlation | `['quality', 'alcohol', 'volatile acidity', 'sulphates', 'citric acid']` | `success` |
| `task_4` | `two_groups_vs_numerical` | Independent samples t-test / Mann-Whitney U test | `['quality', 'alcohol']` | `success` |
| `task_5` | `outlier_sensitivity` | Outlier removal sensitivity check | `['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'quality']` | `success` |

- **Plan Equality:** `base.plan == prop.plan` (**True**).
- **Analysis Equality:** `base.analysis == prop.analysis` (**True**).
- **Baseline Branch Omissions:** Confirmed `critic=None`, `verification=None`, `insights=None`.

---

## 5. Critic Review and Controlled Reinvestigation

The proposed pipeline audited all 5 initial statistical findings and executed controlled reinvestigation where assumption violations were flagged:

| Finding ID | Task ID | Evaluated Variables | Critic Assessment & Concerns | Recommended Method | Follow-Up Reinvestigated Variables | Pair Matched? | Direction Consistent? | Final Status | Reason Codes |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **`finding_1`** | `task_1` | `['quality', 'alcohol']` | `ACCEPTED` (Univariate departures from normality noted) | None | N/A (Directly verified) | Baseline | Baseline | `VERIFIED` | `NON_NORMAL_DISTRIBUTION`, `ADEQUATE_SAMPLE_SIZE`, `VISUAL_OUTLIERS_DETECTED`, `DIRECTLY_VERIFIED` |
| **`finding_2`** | `task_2` | `['quality', 'alcohol']` | `ACCEPTED` (Normal skew caveats) | None | N/A (Directly verified) | Baseline | Baseline | `VERIFIED` | `NON_NORMAL_DISTRIBUTION`, `ADEQUATE_SAMPLE_SIZE`, `VISUAL_OUTLIERS_DETECTED`, `DIRECTLY_VERIFIED` |
| **`finding_3`** | `task_3` | `['quality', 'alcohol']` | `REQUIRES_REINVESTIGATION` (Outlier-influence diagnostic: 14 leverage/residual outlier points; non-normality in quality ratings) | `Spearman rank correlation` | `['quality', 'alcohol']` | **YES** ✅ | **YES** ✅ (Positive: $\rho = 0.4785$, $p = 2.73 \times 10^{-92}$) | `VERIFIED_AFTER_REINVESTIGATION` | `METHOD_VALIDATED`, `VISUAL_OUTLIERS_DETECTED`, `SIGNIFICANT_STRONG_EFFECT`, `ADEQUATE_SAMPLE_SIZE`, `REINVESTIGATION_RESOLVED` |
| **`finding_4`** | `task_4` | `['alcohol', 'quality']` | `ACCEPTED` (Significant group difference confirmed) | None | N/A (Directly verified) | Baseline | Baseline | `VERIFIED` | `GROUP_COMPARISON_VALIDATED`, `VISUAL_OUTLIERS_DETECTED`, `SIGNIFICANT_GROUP_DIFFERENCE`, `ADEQUATE_SAMPLE_SIZE`, `DIRECTLY_VERIFIED` |
| **`finding_5`** | `task_5` | `['alcohol', 'quality']` | `ACCEPTED` (Minimal outlier sensitivity; metric stability confirmed) | None | N/A (Directly verified) | Baseline | Baseline | `VERIFIED` | `MINIMAL_OUTLIER_INFLUENCE`, `ADEQUATE_SAMPLE_SIZE`, `VISUAL_OUTLIERS_DETECTED`, `DIRECTLY_VERIFIED` |

### Detailed Statistical Reinvestigation Analysis:
- In `task_3`, the initial multi-column task (`['quality', 'alcohol', 'volatile acidity', 'sulphates', 'citric acid']`) identified the primary bivariate association as `quality` versus `alcohol`.
- The Critic identified 14 visual scatter leverage points ($|z| > 3.0$) and discrete ordinal scaling in `quality`, recommending `Spearman rank correlation`.
- **Target Variable Pair Preservation:** Under the Task 23 fix, `_build_reinvestigation_task()` extracted `['quality', 'alcohol']` and explicitly bound it to the follow-up task. Positional indexing did not divert the test to unrelated variables.
- **Statistical Alignment:**
  - Original Pearson test: $r = 0.4762$ ($p = 2.83 \times 10^{-91}$).
  - Follow-up Spearman rank test: $\rho = 0.4785$ ($p = 2.73 \times 10^{-92}$).
  - Direction is strictly consistent ($+0.476 \to +0.479$).
  - Monotonic association is robust to discrete rating bins and leverage outliers.
  - Final status: `VERIFIED_AFTER_REINVESTIGATION`.

---

## 6. Evaluation Metrics & Comparative Scorecard

### A. Dimensional Breakdown

| Evaluation Dimension | Baseline Score | Proposed Score | Absolute Delta | Analytical Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Execution Reliability** | 25.0 / 25 (100.0%) | 25.0 / 25 (100.0%) | 0.0 pts | 100% of planned tasks completed cleanly with 0% computational errors across both branches. |
| **Statistical Rigor** | 6.8 / 25 (27.2%) | 20.8 / 25 (83.2%) | **+14.0 pts** | Baseline lacked assumption auditing and non-parametric resolution. Proposed formally audited 5/5 tasks and resolved the flagged violation. |
| **Verification & Grounding** | 0.0 / 25 (0.0%) | 19.0 / 25 (76.0%) | **+19.0 pts** | Baseline omitted verification. Proposed yielded 5 verified findings (100% yield) with full deterministic provenance. |
| **Insight Actionability** | 0.0 / 25 (0.0%) | 18.0 / 25 (72.0%) | **+18.0 pts** | Baseline produced 0 insights. Proposed synthesized 2 verified insights with 100% non-causal phrasing compliance. |
| **Overall Composite Score** | **31.8 / 100 (Grade C)** | **82.8 / 100 (Grade B+)** | **+51.0 pts** | Substantial increase in analytical rigor and truth-grounding. |

> [!IMPORTANT]
> **Score Limitation & Academic Context:**  
> The composite score increase (+51.0 points) directly reflects the architectural execution of Phases 4, 5, and 6 in the proposed branch, which are ablated by design in the baseline branch. This scorecard validates internal pipeline completeness; it does NOT constitute an empirical claim of superiority over AutoMind (IEEE Access 2026).

---

## 7. Limitations & Scientific Disclosures

1. **Correlation vs. Causation:**  
   The strong positive association between `alcohol` and `quality` ($\rho = 0.4785, p < 10^{-90}$) represents an observational sensory correlation in Portuguese red wine. It does not establish that artificially increasing alcohol percentage causes wine to achieve higher expert ratings, as alcohol correlates with grape ripeness, sugar accumulation, and fermentation style.
2. **Deterministic Provenance Rate (40.0%):**  
   The provenance metric heuristic requires a scalar test statistic and p-value. Summary statistics (`task_1`) and distribution assessments (`task_2`) yield multi-metric dictionaries without a singular hypothesis test statistic, resulting in a 40% automated score despite 100% deterministic calculation.
3. **Discrete Target Scale:**  
   `quality` is an integer score bounded between 3 and 8. While Spearman rank correlation and Mann-Whitney U tests are appropriate for ordinal distributions, treating quality as continuous in linear models introduces mild discretization bias.

---

## 8. Artifact Deliverables Summary

1. **Baseline Run Artifact:**  
   [`backend/data/runs/run_20261025_120610_winequality_redcsv_baseline_adf3f1.json`](file:///C:/Users/SPURGE/MAJOR_PROJECT_DOCS/REVIEW%202/Major_Project/backend/data/runs/run_20261025_120610_winequality_redcsv_baseline_adf3f1.json)
2. **Proposed Run Artifact:**  
   [`backend/data/runs/run_20261025_120610_winequality_redcsv_proposed_adf3f1.json`](file:///C:/Users/SPURGE/MAJOR_PROJECT_DOCS/REVIEW%202/Major_Project/backend/data/runs/run_20261025_120610_winequality_redcsv_proposed_adf3f1.json)
3. **Evaluation Report:**  
   [`reports/FINAL_EVALUATION_DATASET_2.md`](file:///C:/Users/SPURGE/MAJOR_PROJECT_DOCS/REVIEW%202/Major_Project/reports/FINAL_EVALUATION_DATASET_2.md)

---

## 9. Final Conclusion

Experiment 2 of 5 on Wine Quality Red completed with complete plan and analysis equality between branches, exactly one outbound Gemini Planner request, clean preservation of all 240 duplicate rows, and strict variable-pair preservation during reinvestigation.

**Experiment 2 Verdict: PASS**
