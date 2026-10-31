# Base Paper Comparison & Attribution Evidence Audit: AutoMind (IEEE Access 2026) vs. DataInvest AI

## Primary Reference
- **Title**: AutoMind: Automated Insight Discovery via Multi-Agent Navigation
- **Authors**: Cong Nguyen Thanh, Son Nguyen Van, Hao Nguyen Ngoc, Cuong Van Duc, Hieu Nguyen Minh, Dung Nguyen Tien, Hanh Nguyen Thi
- **Affiliations**: Hanoi University, Phenikaa University, Hanoi University of Science and Technology (HUST), Vietnam
- **Journal**: *IEEE Access*, 2026, Volume 14, pages 23936–23955
- **Publication Date**: February 6, 2026
- **Electronic ISSN**: 2169-3536
- **DOI**: [10.1109/ACCESS.2026.3661202](https://doi.org/10.1109/ACCESS.2026.3661202)
- **IEEE Xplore**: [https://ieeexplore.ieee.org/document/11372738/](https://ieeexplore.ieee.org/document/11372738/)

> [!NOTE]
> **Disambiguation Notice**: Strictly distinguished from the unrelated preprint *"AutoMind: Adaptive Knowledgeable Agent for Automated Data Science"*, arXiv:2506.10974 (2025). All attributions and architectural comparisons in this project refer exclusively to the peer-reviewed IEEE Access 2026 publication.

---

## 1. Feature Attribution & Evidence Audit Breakdown

This audit reviews every feature described as inherited from, adopted from, or inspired by the cited AutoMind base paper, verifying each claim against the actual paper text (*IEEE Access*, Vol. 14, pp. 23936–23955).

### Item 1: Multi-Agent Pipeline & Specialized Agent Roles
- **Exact Documentation Claim**: Decomposes automated exploration into multi-agent workflows (Planner, Data Profiler, Execution).
- **Supporting Paper Evidence**: *IEEE Access*, Vol. 14, pp. 23936–23955, Abstract & Section III ("AutoMind System Architecture"): AutoMind introduces a "planner-centric, feedback-driven multi-agent architecture organized in a hub-and-spoke structure. The central Planner Agent orchestrates general data analysis and specialized processing within the Clustering and Classification branches, while the Summary Agent integrates the outcomes into coherent final insights." It delegates data profiling, cleaning, and modeling to specialized worker agents.
- **Audit Status**: **Supported**.
- **Lineage Classification**: **Adopted from Base Paper**.
- **Wording Correction**: Retained as adopted; explicitly cites the hub-and-spoke orchestrator-worker architecture.

### Item 2: Objective Decomposition & Natural Language Planning
- **Exact Documentation Claim**: Translates natural language objectives into task graphs and statistical questions.
- **Supporting Paper Evidence**: *IEEE Access*, Vol. 14, pp. 23936–23955, Abstract & Section III-A ("Planner Agent"): AutoMind "translates high-level, natural language goals into comprehensive analytical workflows" using an orchestrator agent that devises step-by-step plans and delegates tasks.
- **Audit Status**: **Supported**.
- **Lineage Classification**: **Adopted from Base Paper**.
- **Wording Correction**: Retained as adopted.

### Item 3: Autonomous Statistical Code Execution
- **Exact Documentation Claim**: Autonomous script generation or tool invocation for data computation.
- **Supporting Paper Evidence**: *IEEE Access*, Vol. 14, pp. 23936–23955, Section III-B ("Execution & Modeling"): AutoMind employs autonomous code execution and statistical modeling tools to process data and extract analytical findings.
- **Audit Status**: **Supported**.
- **Lineage Classification**: **Adopted from Base Paper**.
- **Wording Correction**: Retained as adopted.

### Item 4: Scorecard-Based Evaluation & Multi-Dimensional Metrics
- **Exact Documentation Claim**: Multi-dimensional scoring framework assessing execution stability, relevance, and report quality. / 4-dimension deterministic evaluation (Execution, Rigor, Verification, Insights).
- **Supporting Paper Evidence**: *IEEE Access*, Vol. 14, pp. 23936–23955, Section IV ("Empirical Evaluation"): AutoMind evaluates system performance using a 1–5 rubric-based expert benchmark across three pipeline stages (Planner Stage, Data Processing Stage, Model Selection & Insight Generation Stage) alongside task completion rate and efficiency. However, the specific 4-dimension scorecard (`Execution`, `Rigor`, `Verification`, `Insights`) and the automated deterministic scoring formulas used in DataInvest AI are **not** present in AutoMind.
- **Audit Status**: **Partially supported**.
- **Lineage Classification**: **Project Design Interpretation** (inspired by AutoMind's multi-stage rubric evaluation principles; specific 4-dimension metric scheme is a project design interpretation).
- **Wording Correction**: Reclassified from "Inherited" to "Project Design Interpretation inspired by base paper".

### Item 5: Non-Causal Reporting Guidelines
- **Exact Documentation Claim**: Non-causal phrasing standards for observational insights.
- **Supporting Paper Evidence**: *IEEE Access*, Vol. 14, pp. 23936–23955, Section III-D & IV: AutoMind emphasizes provenance tracking, transparency, and accurate insight communication. However, specific deterministic regex/lexical rules restricting causal verbs (e.g. "causes", "drives", "leads to") on observational data are not formulated as explicit rules in the AutoMind paper.
- **Audit Status**: **Partially supported**.
- **Lineage Classification**: **Project Design Interpretation** (inspired by scientific communication principles; non-causal linguistic enforcement is a project design policy).
- **Wording Correction**: Reclassified from "Inherited" to "Project Design Policy inspired by scientific communication standards".

### Item 6: Verification Mechanisms & Statistical Critic Agent
- **Exact Documentation Claim**: Deterministic Statistical Critic Agent enforcing formal parametric assumption verification: Cochran (1954) contingency rules, Shapiro-Wilk / D'Agostino-Pearson normality checks, standardized residual leverage detection ($|r_i| > 3.0$), and IQR boxplot outlier bounds.
- **Supporting Paper Evidence**: *IEEE Access*, Vol. 14, pp. 23936–23955, Section III ("AutoMind System Architecture") & Graphical Abstract: AutoMind explicitly incorporates a **Verifier Agent** and continuous feedback-driven verification loops. AutoMind's Verifier evaluates intermediate execution integrity, validates code outputs, fact-checks analytical statements against underlying datasets, and prompts operational error-recovery. However, AutoMind's verification operates primarily via conversational reasoning, output consistency checks, and prompt-driven reflection. It does not define deterministic statistical assumption checks (e.g. mathematical verification of Cochran expected-frequency thresholds, residual leverage limits, or formal normality tests).
- **Audit Status**: **Distinguished Implementation Mechanism** (Verification is established in the base paper; deterministic statistical assumption checking is a project-specific mechanism).
- **Lineage Classification**: **Project-Specific Implementation** (conceptual verification adopted from base paper principles; statistical assumption engine is project-specific).
- **Wording Correction**: Corrected to explicitly acknowledge AutoMind's Verifier Agent and feedback-driven verification loops, while clearly distinguishing DataInvest AI's deterministic statistical assumption checks. Disallowed any claim that AutoMind lacks verification.

### Item 7: Error Recovery & Controlled Alternative Re-Investigation
- **Exact Documentation Claim**: Controlled, 1-budget alternative test re-investigation: Formally swaps failed parametric tests to robust non-parametric equivalents (Pearson $r \rightarrow$ Spearman $\rho$, Welch's $t \rightarrow$ Mann-Whitney $U$) with quantitative delta auditing.
- **Supporting Paper Evidence**: *IEEE Access*, Vol. 14, pp. 23936–23955, Section III: AutoMind implements feedback-driven error recovery and adaptive refinement loops coordinated by the Planner Agent when the Verifier Agent flags discrepancies or execution failures. However, AutoMind relies on unconstrained LLM replanning rather than a deterministic, 1-budget statistical test substitution protocol (such as swapping failed parametric tests to predefined non-parametric alternatives upon assumption violations).
- **Audit Status**: **Distinguished Implementation Mechanism** (Adaptive refinement supported by base paper; bounded parametric-to-nonparametric test substitution is project-specific).
- **Lineage Classification**: **Project-Specific Mechanism**.
- **Wording Correction**: Acknowledges AutoMind's adaptive refinement loops while distinguishing DataInvest AI's bounded, 1-budget test substitution protocol.

### Item 8: Diagnostic Visual Evidence Integration
- **Exact Documentation Claim**: Programmatic Vega-Lite Visual Evidence coupled directly to each statistical finding with empirical diagnostic overlays (residual plots, expected vs observed tables, boxplot outliers) passed to Critic as auditing evidence.
- **Supporting Paper Evidence**: *IEEE Access*, Vol. 14, pp. 23936–23955, Section III: AutoMind generates visualizations as final artifacts to assist end users in understanding discovered insights. Our project specifically couples programmatic Vega-Lite chart specifications and empirical diagnostic overlays (such as residual distributions and expected frequency matrices) directly to individual findings as machine-readable evidence consumed by the Critic Agent.
- **Audit Status**: **Project-Specific Mechanism** (Visual output generation supported; Critic diagnostic evidence coupling is project-specific).
- **Lineage Classification**: **Project-Specific Mechanism**.
- **Wording Correction**: Balanced to acknowledge visual reporting in base paper while clarifying finding-level diagnostic coupling in this project.

### Item 9: Structured Audit Logging & Traceability
- **Exact Documentation Claim**: Redacted, structured JSON-Lines execution logging (`PLANNER_COMPLETED`, `CRITIC_COMPLETED`, `REINVESTIGATION_COMPLETED`, `REPORT_GENERATED`) with zero sensitive data leaks and immutable run histories.
- **Supporting Paper Evidence**: AutoMind discusses execution traceability and provenance tracking throughout its multi-agent pipeline. The specific redacted JSONL logging schema and reason-code lifecycle implemented here are project-specific.
- **Audit Status**: **Project-Specific Mechanism** (Traceability supported; specific redacted JSONL event schema is project-specific).
- **Lineage Classification**: **Project-Specific Mechanism**.
- **Wording Correction**: Described as a project-specific implementation of traceability principles.

### Item 10: Paired Benchmark Fairness Architecture
- **Exact Documentation Claim**: Paired Single-Plan Benchmark Architecture sharing identical dataset, genuine LLM plan, and raw SciPy results between Baseline and Proposed branches.
- **Supporting Paper Evidence**: AutoMind evaluates multi-agent vs single-agent and human expert comparisons; the paired single-plan execution protocol is designed for DataInvest AI to isolate the specific impact of the Critic Agent.
- **Audit Status**: **Project-Specific Evaluation Protocol**.
- **Lineage Classification**: **Project-Specific Evaluation Protocol**.
- **Wording Correction**: Described as a project evaluation protocol. Any comparative claim against AutoMind's published benchmark results is explicitly marked as **Requires verification**.

---

## 2. Lineage & Contribution Matrix

| Capability / Dimension | AutoMind Base Paper (*IEEE Access* 2026) | DataInvest AI (Project Implementation) | Evidence & Lineage Classification |
|---|---|---|---|
| **Multi-Agent Pipeline** | Hub-and-spoke orchestrator-worker layout: central Planner Agent coordinates specialized workers, a **Verifier Agent**, and a Summary Agent. | Specialized pipeline: Profiler $\rightarrow$ Planner $\rightarrow$ Analysis $\rightarrow$ Critic $\rightarrow$ Verification $\rightarrow$ Report. | **Adopted from Base Paper** (Supported: Sec. III) |
| **Objective Decomposition** | Central Planner translates natural language goals into analytical workflows. | Genuine Gemini-based Planner translates goals into typed SciPy tasks with quality-aware column selection. | **Adopted from Base Paper** (Supported: Sec. III-A) |
| **Statistical Code Execution** | Autonomous script generation and tool invocation for statistical computation. | Deterministic Python/SciPy execution in sandboxed service with structured failure isolation. | **Adopted from Base Paper** (Supported: Sec. III-B) |
| **Evaluation Scorecard** | 1–5 rubric-based expert benchmark across Planner, Data Processing, and Modeling stages. | Automated 4-dimension deterministic evaluation (Execution, Rigor, Verification, Insights). | **Project Design Interpretation** (Partially supported: inspired by Sec. IV rubric principles) |
| **Non-Causal Reporting** | Emphasizes transparent insight generation and provenance tracking. | Deterministic rule-based checks restricting causal claims from observational correlation findings. | **Project Design Interpretation** (Partially supported: project design policy) |
| **Verification & Critic Auditing** | **Verifier Agent & continuous feedback loops**: evaluates intermediate code outputs, fact-checks claims against data, and prompts operational error-recovery. | **Deterministic Statistical Critic Agent**: executes formal mathematical assumption checks (Cochran expected counts, normality tests, residual leverage $|r_i| > 3.0$, IQR boxplot bounds). | **Project-Specific Implementation** (Base paper incorporates Verifier Agent; mathematical assumption engine is project-specific) |
| **Error Recovery & Re-investigation** | **Feedback-driven refinement loops**: Planner adjusts execution based on Verifier feedback. | **Controlled 1-budget alternative test re-investigation**: Formally swaps failed parametric tests to robust non-parametric equivalents (Pearson $\rightarrow$ Spearman, Welch's $t \rightarrow$ Mann-Whitney $U$) with quantitative delta auditing. | **Project-Specific Mechanism** (Base paper incorporates adaptive refinement; bounded test substitution is project-specific) |
| **Visual Evidence Integration** | Plots generated as reporting outputs for human interpretation. | **Programmatic Vega-Lite Visual Evidence** coupled directly to each statistical finding with empirical diagnostic overlays passed to Critic as auditing evidence. | **Project-Specific Mechanism** (Base paper includes plotting; diagnostic evidence coupling is project-specific) |
| **Structured Traceability & Logging** | General execution logs and provenance tracking. | **Redacted, structured JSON-Lines execution logging** (`PLANNER_COMPLETED`, `CRITIC_COMPLETED`, `REINVESTIGATION_COMPLETED`, `REPORT_GENERATED`) with zero sensitive data leaks. | **Project-Specific Mechanism** (Implementation specific) |
| **Paired Benchmark Fairness** | Independent multi-agent vs single-agent and human expert comparisons. | **Paired Single-Plan Benchmark Architecture**: Baseline and Proposed branches share the exact same LLM plan and raw SciPy analysis results, isolating Critic verification impact from planner variance. | **Project-Specific Evaluation Protocol** (Evaluation protocol; quantitative superiority requires verification) |

---

## 3. Experimental Fairness Protocol

To ensure scientifically valid and defensible comparative evaluation:
1. **Identical Inputs**: Both branches evaluate the exact same uploaded dataset and natural language objective.
2. **Identical Single Plan**: Exactly **one** plan is formulated by the genuine Gemini LLM Planner (`is_research_run=True`). Zero deterministic fallback is permitted in research mode.
3. **Identical Statistical Execution**: SciPy statistical tasks are executed once. The exact same raw analysis results are passed to both branches:
   - **Baseline Branch**: Dataset $\rightarrow$ LLM Plan $\rightarrow$ Analysis $\rightarrow$ Un-critiqued Findings (100% accepted without statistical verification).
   - **Proposed Branch**: Dataset $\rightarrow$ Same LLM Plan $\rightarrow$ Same Analysis $\rightarrow$ Critic Agent $\rightarrow$ Controlled Reinvestigation $\rightarrow$ Verified Final Findings.
4. **Strict Isolation of Variance**: Any delta in scientific rigor or quality scores arises solely from the Critic's assumption auditing and non-parametric corrections, not from random prompt variance or different plans.

---

## 4. Scope, Methodological Boundaries, and Non-Exaggeration Notice

> [!IMPORTANT]
> **Scientific Integrity & Verification Boundaries**:
> 1. **Procedural Verification vs. Proof**: The Critic accepted findings under the implemented verification checks. Acceptance indicates procedural compliance with predefined criteria; it does not establish ground-truth correctness, causal validity, or mathematical proof.
> 2. **Acceptance Rate**: The finding acceptance rate reflects procedural adherence; it must **never** be cited as a measure of statistical accuracy or correctness.
> 3. **Non-Superiority Policy**: Neither branch claims unverified superiority or general novelty over the AutoMind base paper. AutoMind's verification capabilities (including its Verifier Agent and feedback loops) are explicitly acknowledged; our Critic Agent and controlled reinvestigation are presented strictly as project-specific mechanisms.
> 4. **Comparative Claims Require Verification**: Any quantitative comparative claim between AutoMind published benchmarks and DataInvest AI requires empirical execution on identical benchmark datasets under controlled protocols. All comparative claims without paired empirical execution remain explicitly marked as **Requires verification**.
