# FINAL EVALUATION — EXPERIMENT 3 OF 5: BANK MARKETING REPORT

**Project:** DataInvest AI  
**Dataset:** `data/bank-full.csv` (Bank Marketing Dataset)  
**Execution Date:** 2026-10-25 (UTC 12:09:20)  
**Evaluation Verdict:** **PASS**  
**Authoritative Working Root:** `C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\Major_Project`

---

## 1. Executive Summary

Experiment 3 of 5 evaluated the DataInvest AI pipeline on `data/bank-full.csv` (Moro et al., 2014), testing the framework on a scaled, real-world marketing dataset (45,211 records, 17 features) containing mixed numerical, categorical, and temporal variables.

Key achievements:
- **Single Outbound Planner Request:** Exactly ONE request was made to `gemini-3.5-flash-lite`. Zero retries, zero subsequent external API calls.
- **Architectural Parity:** The baseline and proposed branches shared the identical dataset profile, genuine LLM-generated plan, and initial statistical analysis.
- **Data Integrity:** Semicolon delimiter (`;`) was parsed accurately; 0 missing values and 0 duplicate rows were recorded; all 45,211 rows were preserved intact.
- **Honest Statistical Caveats:** The Critic and Verification services cleanly identified an extreme outlier sensitivity in `balance` (-53.0% shift), retaining `finding_4` as `QUESTIONABLE` with explicit reason codes rather than claiming false verification.
- **Graceful Visualization Constraint Handling:** Altair's 5,000-row embedded chart ceiling was caught gracefully by the visualization engine without interrupting statistical execution.
- **Non-Destructive Persistence:** Both branch run artifacts were persisted under unique IDs without modifying historical files.

### Primary Artifact Identifiers:
- **Experiment ID:** `exp_20261025_120920_bank_fullcsv_805a96`
- **Baseline Run ID:** `run_20261025_120920_bank_fullcsv_baseline_805a96`
- **Proposed Run ID:** `run_20261025_120920_bank_fullcsv_proposed_805a96`
- **Dataset File:** `data/bank-full.csv` (45,211 rows x 17 columns)

---

## 2. Dataset Dimensions & Preprocessing Profile

`data/bank-full.csv` contains direct marketing campaign records of a Portuguese banking institution regarding client subscriptions to long-term deposits (`y`).

| Dataset Dimension | Characteristic | Notes |
| :--- | :--- | :--- |
| **Dimensions** | 45,211 rows x 17 columns | Scaled tabular test. Total cells: 768,587. |
| **Delimiter** | Semicolon (`;`) | Correctly detected by `upload_service._detect_delimiter`. |
| **Missing Values** | 0 missing cells (0.0%) | Complete observations across all 17 features. |
| **Duplicate Observations** | 0 duplicate rows (0.0%) | Clean independent client interaction records. |
| **Column Classification** | 7 Numerical, 9 Categorical, 1 Datetime | Features include client demographics (`age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`) and contact data (`contact`, `day`, `month`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`, `y`). |
| **Target Variable** | `y` | Binary categorical target: `"no"` (88.3%) vs. `"yes"` (11.7%). Highly imbalanced. |
| **Memory Footprint** | ~15.9 MB in memory | Processed efficiently with sub-second execution overhead. |

---

## 3. Planner Execution, Latency & Resource Utilization

| Metric / Parameter | Observed Value | Validation Notes |
| :--- | :---: | :--- |
| **Planner Model** | `gemini-3.5-flash-lite` | Authorized LLM model. |
| **Planner Request Count** | **1** | Tracked via `InstrumentedGeminiClient` at `2026-10-25T12:09:20.143615Z`. |
| **Planner Response Status** | `HTTP/1.1 200 OK` | Succeeded in **4.70s** (prompt length: 2,467 characters). Zero retries. |
| **Secondary LLM Calls** | **0** | Critic, verification, insights, and evaluation ran 100% locally offline. |
| **Initial Analysis Latency** | 0.4152 s | SciPy statistical engine executed across 45,211 rows in ~415 ms. |
| **Shared Duration** | **5.1125 s** | **Identical shared cost across both branches.** |
| **Baseline Branch Latency** | 0.0004 s | Baseline evaluation calculation only. |
| **Proposed Branch Overhead** | **0.0296 s** | Critic audit (0.0003s) + Verification/Reinvestigation (0.0293s). Overhead: ~30 ms. |
| **Total Experiment Monotonic Time** | **6.0735 s** | Complete execution including two-branch JSON serialization and disk persistence. |

---

## 4. Plan-Sharing and Initial Analysis Equality Check

**Research Objective:**  
`"Investigate how client demographics (age, job, education, balance) and campaign contact characteristics (duration, campaign, poutcome) relate to term deposit subscription success (y)."`

The Planner generated 4 analytical tasks addressing both demographic and campaign dynamics:

| Task ID | Task Type | Configured Method | Assigned Columns | Execution Status |
| :--- | :--- | :--- | :--- | :---: |
| `task_1` | `descriptive_statistics` | Summary statistics computation | `['age', 'balance', 'duration', 'campaign']` | `success` |
| `task_2` | `two_groups_vs_numerical` | Independent t-test / Mann-Whitney U test | `['y', 'duration']` | `success` |
| `task_3` | `categorical_vs_categorical` | Chi-square test of independence | `['job', 'y']` | `success` |
| `task_4` | `outlier_sensitivity` | IQR-based outlier flagging and sensitivity checking | `['balance', 'age']` | `success` |

- **Plan Equality:** `base.plan == prop.plan` (**True**).
- **Analysis Equality:** `base.analysis == prop.analysis` (**True**).
- **Baseline Architectural Purity:** Confirmed `critic=None`, `verification=None`, `insights=None`.

---

## 5. Critic Review and Verification Audit

The proposed pipeline audited all 4 findings and evaluated methodological assumptions:

| Finding ID | Task ID | Evaluated Variables | Primary Statistic & p-value | Critic Assessment & Concerns | Final Verification Status | Reason Codes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`finding_1`** | `task_1` | `['age', 'balance']` | N/A (Summary descriptives) | `ACCEPTED` (Skewness in balance and duration noted) | `VERIFIED` | `NON_NORMAL_DISTRIBUTION`, `ADEQUATE_SAMPLE_SIZE`, `DIRECTLY_VERIFIED` |
| **`finding_2`** | `task_2` | `['y', 'duration']` | Mann-Whitney $U = 4.06 \times 10^7$, $p = 0.0$ | `ACCEPTED` (Strong significant group difference; subscription associated with longer call duration) | `VERIFIED` | `GROUP_COMPARISON_VALIDATED`, `SIGNIFICANT_GROUP_DIFFERENCE`, `ADEQUATE_SAMPLE_SIZE`, `DIRECTLY_VERIFIED` |
| **`finding_3`** | `task_3` | `['job', 'y']` | $\chi^2 = 836.11$, $p = 3.34 \times 10^{-172}$, $V = 0.136$ | `ACCEPTED` (Cochran condition satisfied: 0 cells < 5; significant categorical association) | `VERIFIED` | `COCHRAN_CONDITION_SATISFIED`, `SIGNIFICANT_CATEGORICAL_ASSOCIATION`, `ADEQUATE_SAMPLE_SIZE`, `DIRECTLY_VERIFIED` |
| **`finding_4`** | `task_4` | `['balance', 'age']` | Mean shift = -53.0% on outlier trim | `QUESTIONABLE` (High outlier sensitivity: extreme balances heavily distort mean by -53.0%) | `QUESTIONABLE` | `SUBSTANTIAL_OUTLIER_INFLUENCE`, `ADEQUATE_SAMPLE_SIZE`, `QUESTIONABLE_FINDING` |

### Audit Highlights:
1. **Reinvestigation Trigger Count:** 0 re-analyses were required because Critic directly validated `task_1`, `task_2`, and `task_3` under non-parametric Mann-Whitney and Chi-square frameworks.
2. **Defensive Verification of Fragile Finding:** `task_4` was flagged for extreme outlier sensitivity (trimming outliers changes average client balance by -53.0%). The system **did not falsely verify** this finding; it correctly retained it as `QUESTIONABLE` with clear caveats.
3. **Traceability:** Variable pairs were strictly preserved and recorded.

---

## 6. Evaluation Metrics & Comparative Scorecard

### A. Dimensional Breakdown

| Evaluation Dimension | Baseline Score | Proposed Score | Absolute Delta | Analytical Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Execution Reliability** | 25.0 / 25 (100.0%) | 25.0 / 25 (100.0%) | 0.0 pts | 100% of planned tasks completed cleanly with 0% computational errors across 45,211 rows. |
| **Statistical Rigor** | 8.5 / 25 (34.0%) | 20.8 / 25 (83.2%) | **+12.3 pts** | Baseline lacked assumption auditing and sensitivity analysis. Proposed formally audited 4/4 findings and flagged outlier instability. |
| **Verification & Grounding** | 0.0 / 25 (0.0%) | 19.0 / 25 (76.0%) | **+19.0 pts** | Baseline omitted verification. Proposed yielded 3 verified findings (75% yield) and 1 caveated questionable finding. |
| **Insight Actionability** | 0.0 / 25 (0.0%) | 18.0 / 25 (72.0%) | **+18.0 pts** | Baseline produced 0 insights. Proposed synthesized 2 verified insights with 100% non-causal phrasing compliance. |
| **Overall Composite Score** | **33.5 / 100 (Grade C)** | **82.8 / 100 (Grade B+)** | **+49.3 pts** | Significant increase in verified truth-grounding and methodological rigor. |

> [!IMPORTANT]
> **Composite Score Limitations:**  
> The composite score delta (+49.3 points) reflects the internal execution of Critic auditing and verified insight synthesis in the proposed branch, which are bypassed by design in the baseline branch. This score confirms that the proposed pipeline successfully executed its complete auditing lifecycle on a 45,000-row dataset, but does NOT establish general superiority over AutoMind (IEEE Access 2026).

---

## 7. Limitations & Technical Notes

1. **Association vs. Causation on Call Duration:**  
   The strong group difference in `duration` between subscribers and non-subscribers ($U = 4.06 \times 10^7, p = 0.0$) is a well-known benchmark phenomenon in bank marketing. As documented by Moro et al. (2014), call duration is unknown before a call is made and heavily correlates with customer interest; it cannot be interpreted as a causal lever (i.e. merely talking longer to a client does not cause them to purchase a term deposit). The proposed pipeline correctly maintained strict observational phrasing (`m_insight_causality_safety = 100.0%`).
2. **Visualization Engine Row Ceiling Warning:**  
   The visualization engine logged warnings on Altair's standard 5,000-row embedded chart limit (`The number of rows in your dataset (45211) is greater than the maximum allowed (5000)`). The system handled this gracefully via fallback handling without halting or compromising statistical analysis.
3. **Severe Balance Skewness:**  
   Client bank balances feature extreme right-skew and negative balances (overdrafts), resulting in -53.0% sensitivity shifts when trimming outliers. Non-parametric rank tests or median-based statistics are essential when analyzing customer financial assets in this dataset.

---

## 8. Artifact Deliverables Summary

1. **Baseline Run Artifact:**  
   [`backend/data/runs/run_20261025_120920_bank_fullcsv_baseline_805a96.json`](file:///C:/Users/SPURGE/MAJOR_PROJECT_DOCS/REVIEW%202/Major_Project/backend/data/runs/run_20261025_120920_bank_fullcsv_baseline_805a96.json)
2. **Proposed Run Artifact:**  
   [`backend/data/runs/run_20261025_120920_bank_fullcsv_proposed_805a96.json`](file:///C:/Users/SPURGE/MAJOR_PROJECT_DOCS/REVIEW%202/Major_Project/backend/data/runs/run_20261025_120920_bank_fullcsv_proposed_805a96.json)
3. **Evaluation Report:**  
   [`reports/FINAL_EVALUATION_DATASET_3.md`](file:///C:/Users/SPURGE/MAJOR_PROJECT_DOCS/REVIEW%202/Major_Project/reports/FINAL_EVALUATION_DATASET_3.md)

---

## 9. Final Conclusion

Experiment 3 of 5 on Bank Marketing completed with complete plan and analysis equality between branches, exactly one outbound Gemini Planner request, clean handling of a 45,211-row dataset, and honest retention of sensitive findings as `QUESTIONABLE`.

**Experiment 3 Verdict: PASS**
