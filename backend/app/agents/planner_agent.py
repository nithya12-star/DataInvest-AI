"""
DataInvest AI — Phase 2: Planner Agent

Purpose:
    Convert a user analytical objective and Phase 1 dataset profile
    into a structured, prioritized investigation plan.

Research foundation:
    AutoMind — Automated Insight Discovery via Multi-Agent Navigation.

Project adaptation:
    This project does NOT reproduce the complete multi-agent architecture
    from the base paper. The Planner is one component in our simplified
    architecture, with a dedicated Critic Agent planned for a later phase.

CRITICAL RULES:
    - The Planner NEVER performs statistical calculations.
    - The Planner NEVER calculates p-values.
    - The Planner NEVER calculates correlation coefficients.
    - The Planner NEVER calculates test statistics.
    - The Planner NEVER creates statistical findings.
    - The Planner only decides WHAT should be investigated.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from app.schemas.planner_schema import (
    InvestigationTask,
    PlannerResponse,
    SelectedColumn,
    TaskType,
)
from app.services.gemini_client import (
    GeminiClient,
    GeminiClientError,
    GeminiMalformedResponseError,
    GeminiMissingKeyError,
)

logger = logging.getLogger("datainvest.planner.agent")

ALLOWED_TASK_TYPES: set[str] = {
    "descriptive_statistics",
    "distribution_analysis",
    "missing_value_investigation",
    "duplicate_investigation",
    "numerical_vs_numerical",
    "two_groups_vs_numerical",
    "multiple_groups_vs_numerical",
    "categorical_vs_categorical",
    "outlier_sensitivity",
    "binary_classification",
    "clustering",
    "customer_segmentation",
    "text_classification",
    "fake_news_classification",
}


class PlannerAgent:
    """
    DataInvest AI Planner Agent.
    Supports both genuine LLM-driven planning (gemini-3.5-flash-lite) and legacy deterministic planning.

    Inputs:
        - Phase 1 dataset profile
        - User investigation objective

    Output:
        - Structured investigation plan (PlannerResponse)

    The Planner selects suitable analytical tasks but does not execute them.
    Actual statistical calculations belong to Phase 3.
    """

    PLANNER_VERSION = "2.1"

    def __init__(
        self,
        use_llm: bool = False,
        gemini_client: Optional[GeminiClient] = None,
    ):
        """
        Initialize the Planner Agent.

        Parameters
        ----------
        use_llm : bool, default False
            If True, invoke the genuine Gemini LLM client.
            If False, execute the deterministic heuristic planner.
            Defaults to False to strictly preserve legacy behavior for existing callers and tests.
        gemini_client : GeminiClient, optional
            Injected GeminiClient instance (e.g. configured with MockTransport for offline tests).
            If use_llm is True and client is not provided, instantiates default GeminiClient.
        """
        self.use_llm = use_llm
        self.gemini_client = gemini_client

        if self.use_llm and self.gemini_client is None:
            self.gemini_client = GeminiClient()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def plan(
        self,
        profile: dict[str, Any],
        objective: str = "Investigate this dataset.",
    ) -> PlannerResponse:
        """
        Generate a structured investigation plan.

        The Planner consumes the Phase 1 profile and user objective.
        It does not execute statistical tests.
        """
        if not profile or not isinstance(profile, dict):
            raise ValueError(
                "A valid dataset profile dictionary is required for planning."
            )

        clean_objective = (objective or "").strip()
        if not clean_objective:
            clean_objective = "Investigate this dataset."

        if self.use_llm:
            return self._generate_llm_plan(
                profile=profile,
                objective=clean_objective,
            )

        return self._generate_deterministic_plan(
            profile=profile,
            objective=clean_objective,
        )

    # ------------------------------------------------------------------
    # GENUINE GEMINI LLM PLANNER
    # ------------------------------------------------------------------

    def _generate_llm_plan(
        self,
        profile: dict[str, Any],
        objective: str,
    ) -> PlannerResponse:
        """
        Generate an investigation plan using genuine Gemini LLM API (gemini-3.5-flash-lite).

        CRITICAL INTEGRITY RULES:
        - Never silently fall back to deterministic planning or mock data.
        - Failures (missing key, auth error, rate limit, network failure, malformed response)
          raise explicit domain exceptions immediately.
        - Validate against PlannerResponse and InvestigationTask schemas.
        - Reject unknown/hallucinated column names and invalid task types.
        - Enforce that the LLM does NOT invent numerical/statistical results.
        """
        if not self.gemini_client:
            raise GeminiMissingKeyError(
                "Gemini client is not initialized for LLM planning."
            )

        overview = profile.get("overview", {}) or {}
        column_names = overview.get("column_names", []) or []
        if not column_names:
            column_types = profile.get("column_types", {}) or {}
            column_names = list(column_types.get("numerical", [])) + list(column_types.get("categorical", []))

        valid_column_set = set(column_names)
        if not valid_column_set:
            raise ValueError("Dataset profile contains no valid column names for LLM planning.")

        prompt, system_instruction = self._build_llm_planning_prompt(
            profile=profile,
            objective=objective,
            column_names=column_names,
        )

        logger.info(
            "Invoking Gemini Planner with objective: '%s' (model: %s)",
            objective,
            getattr(self.gemini_client, "model", "gemini-3.5-flash-lite"),
        )

        # Call Gemini client (raises GeminiClientError subclasses explicitly on failure)
        raw_json = self.gemini_client.generate_json(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=4096,
        )

        if not isinstance(raw_json, dict):
            raise GeminiMalformedResponseError(
                f"Gemini returned {type(raw_json).__name__} instead of a JSON object for plan."
            )

        # 1. Semantic & Anti-Hallucination Validation
        raw_json["planner_mode"] = "llm"
        raw_json["planner_status"] = "success"
        raw_json["planner_model"] = getattr(self.gemini_client, "model", "gemini-3.5-flash-lite")
        raw_json["planner_provider"] = "google"
        raw_json["objective"] = objective

        # Ensure analysis_tasks is present
        tasks_list = raw_json.get("analysis_tasks") or raw_json.get("tasks")
        if not tasks_list or not isinstance(tasks_list, list):
            raise GeminiMalformedResponseError("Gemini plan response missing non-empty 'analysis_tasks' list.")

        for idx, task in enumerate(tasks_list):
            if not isinstance(task, dict):
                raise GeminiMalformedResponseError(f"Task index {idx} is not a valid JSON dictionary.")

            # Ensure task_id
            if not task.get("task_id"):
                task["task_id"] = f"llm_task_{idx + 1}"

            # Ensure task title ("task" field required by InvestigationTask schema)
            raw_task_title = task.get("task")
            if not isinstance(raw_task_title, str) or not raw_task_title.strip():
                candidate_title = None
                for key in ("title", "task_name", "task_title", "name"):
                    val = task.get(key)
                    if isinstance(val, str) and val.strip():
                        candidate_title = val.strip()
                        break

                if candidate_title:
                    task["task"] = candidate_title
                elif task.get("description") and isinstance(task.get("description"), str) and task["description"].strip():
                    desc = task["description"].strip()
                    first_sentence = desc.split(".")[0].strip()
                    task["task"] = (first_sentence[:80] or desc[:80])
                else:
                    cols_str = ", ".join(str(c) for c in task.get("columns", [])) if task.get("columns") else "features"
                    task_type_str = str(task.get("task_type", "task")).replace("_", " ").title()
                    task["task"] = f"{task_type_str}: {cols_str}"
            else:
                task["task"] = raw_task_title.strip()

            # Validate task_type
            task_type = task.get("task_type")
            if task_type not in ALLOWED_TASK_TYPES:
                raise GeminiMalformedResponseError(
                    f"Task '{task.get('task_id')}' specifies invalid task_type '{task_type}'. "
                    f"Must be one of: {sorted(ALLOWED_TASK_TYPES)}"
                )

            # Validate column existence (anti-hallucination)
            cols = task.get("columns", [])
            if not isinstance(cols, list):
                raise GeminiMalformedResponseError(f"Task '{task.get('task_id')}' columns must be a list of strings.")

            for c in cols:
                if c not in valid_column_set:
                    raise GeminiMalformedResponseError(
                        f"Task '{task.get('task_id')}' references hallucinated column '{c}'. "
                        f"Allowed columns: {sorted(valid_column_set)}"
                    )

            for col_field in ("column_a", "column_b", "group_column", "value_column"):
                val = task.get(col_field)
                if val and val not in valid_column_set:
                    raise GeminiMalformedResponseError(
                        f"Task '{task.get('task_id')}' field '{col_field}' references hallucinated column '{val}'."
                    )

            # Anti-fabrication verification: verify no computed statistics are claimed
            task_text = f"{task.get('description', '')} {task.get('reason', '')}"
            if re.search(r"\bp[\s=<>]+0\.\d+", task_text, re.IGNORECASE) or re.search(r"\br[\s=]+0\.\d+", task_text, re.IGNORECASE):
                raise GeminiMalformedResponseError(
                    f"Task '{task.get('task_id')}' violates scientific boundaries by claiming computed statistics."
                )

        # Validate selected_columns
        selected_cols = raw_json.get("selected_columns", [])
        if isinstance(selected_cols, list):
            for sc in selected_cols:
                if isinstance(sc, dict):
                    col_name = sc.get("column")
                    if col_name and col_name not in valid_column_set:
                        raise GeminiMalformedResponseError(
                            f"Selected column '{col_name}' does not exist in dataset."
                        )

        # Validate target_column
        target_col = raw_json.get("target_column")
        if target_col and target_col not in valid_column_set:
            raise GeminiMalformedResponseError(
                f"Identified target_column '{target_col}' does not exist in dataset."
            )

        # Sync tasks and analysis_tasks aliases
        raw_json["analysis_tasks"] = tasks_list
        raw_json["tasks"] = tasks_list

        # Attach dataset summary
        n_rows = int(overview.get("n_rows", 0) or 0)
        n_cols = int(overview.get("n_cols", 0) or 0)
        raw_json["dataset_summary"] = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "planner_version": self.PLANNER_VERSION,
            "planner_mode": "llm",
            "planner_status": "success",
            "model": getattr(self.gemini_client, "model", "gemini-3.5-flash-lite"),
            "provider": "google",
        }

        # 2. Pydantic Schema Validation
        try:
            validated_plan = PlannerResponse.model_validate(raw_json)
        except Exception as exc:
            logger.error("LLM plan failed Pydantic validation: %s", exc)
            raise GeminiMalformedResponseError(
                f"Gemini plan failed schema validation: {exc}"
            ) from exc

        return validated_plan

    def _build_llm_planning_prompt(
        self,
        profile: dict[str, Any],
        objective: str,
        column_names: list[str],
    ) -> tuple[str, str]:
        """Build the system instruction and prompt for Gemini LLM planning."""
        system_instruction = (
            "You are the DataInvest AI Planner Agent. "
            "Your responsibility is to design a sound, prioritized statistical investigation plan "
            "based on a Phase 1 dataset profile and user objective.\n\n"
            "CRITICAL SCIENTIFIC RULES:\n"
            "1. You decide WHAT should be investigated and WHY. You MUST NEVER calculate or fabricate numerical results, "
            "test statistics, correlation coefficients, p-values, or conclusions. All actual calculations are computed in Python by the Analysis Agent.\n"
            "2. You MUST ONLY use columns that exist in the provided dataset column list. NEVER hallucinate columns.\n"
            "3. Every task must use one of these standardized task_type values: "
            "['descriptive_statistics', 'distribution_analysis', 'missing_value_investigation', "
            "'duplicate_investigation', 'numerical_vs_numerical', 'two_groups_vs_numerical', "
            "'multiple_groups_vs_numerical', 'categorical_vs_categorical', 'outlier_sensitivity'].\n"
            "4. Output MUST be a single valid JSON object adhering strictly to the PlannerResponse schema.\n"
            "5. CONCISENESS & COMPLETENESS: For high-dimensional datasets, select ONLY the target column and up to 9 primary predictor columns or constructs (at most 10 items in 'selected_columns'). Design exactly 5 to 8 prioritized, non-redundant 'analysis_tasks'. Keep all descriptions and rationales concise (1-2 sentences). Ensure the JSON output is completely closed and valid without truncation."
        )

        overview = profile.get("overview", {}) or {}
        column_types = profile.get("column_types", {}) or {}
        missing = profile.get("missing", {}) or {}
        cardinality = profile.get("cardinality", {}) or {}
        outliers = profile.get("outliers", {}) or {}

        user_prompt = f"""
Dataset Overview:
- Rows: {overview.get('n_rows', 0)}
- Columns count: {overview.get('n_cols', len(column_names))}
- Column Names: {column_names}
- Column Types: {column_types}
- Missing Summary: total_missing={missing.get('total_missing', 0)}, total_missing_pct={missing.get('total_missing_pct', 0.0)}%
- Constant Columns: {cardinality.get('constant_columns', [])}
- High Cardinality Columns: {cardinality.get('high_cardinality_columns', [])}
- Potential Outlier Columns: {list(outliers.keys()) if isinstance(outliers, dict) else []}

User Objective:
"{objective}"

Generate a concise, structured investigation plan JSON adhering to these scoping constraints:
- "selected_columns": At most 10 key columns or constructs, prioritizing the target variable and primary predictors.
- "analysis_tasks": Exactly 5 to 8 prioritized, non-redundant investigation tasks.
- Keep descriptions, rationales, and notes concise (1 to 2 sentences) to guarantee complete, valid JSON output.

Fields:
{{
  "objective": "{objective}",
  "interpreted_goal": "<structured interpretation of the analytical objective>",
  "target_column": "<primary target column name or null>",
  "selected_columns": [
    {{"column": "<col_name>", "data_type": "<numerical|categorical|datetime>", "reason": "<rationale>", "is_target": <bool>}}
  ],
  "analysis_tasks": [
    {{
      "task_id": "task_1",
      "task_type": "<one of the allowed task types>",
      "task": "<short human readable title>",
      "description": "<detailed description of what analysis should be conducted>",
      "method": "<e.g. Pearson correlation, Welch's t-test, One-way ANOVA, Chi-square test of independence>",
      "columns": ["<col1>", "<col2>"],
      "column_a": "<col1 or null>",
      "column_b": "<col2 or null>",
      "group_column": "<grouping column or null>",
      "value_column": "<metric column or null>",
      "reason": "<statistical rationale>",
      "priority": "<high|medium|low>",
      "notes": ["<cautions or notes>"]
    }}
  ],
  "reasoning": ["<step 1>", "<step 2>"],
  "warnings": ["<data quality warning 1>"],
  "expected_outputs": ["<deliverable 1>"]
}}
"""
        return user_prompt.strip(), system_instruction
    # DETERMINISTIC PLANNER
    # ------------------------------------------------------------------

    def _generate_deterministic_plan(
        self,
        profile: dict[str, Any],
        objective: str,
    ) -> PlannerResponse:
        """
        Generate a deterministic investigation plan.

        IMPORTANT:
            This method only selects future investigations.
            It does not execute statistical calculations.
        """

        overview = profile.get("overview", {})

        column_names = overview.get("column_names", []) or []
        column_types = profile.get("column_types", {}) or {}

        cardinality_info = profile.get("cardinality", {}) or {}
        cardinality = cardinality_info.get("per_column", {}) or {}

        constant_cols = set(
            cardinality_info.get("constant_columns", []) or []
        )

        high_card_cols = set(
            cardinality_info.get("high_cardinality_columns", []) or []
        )

        missing_info = (
            profile.get("missing", {}).get("per_column", {})
            or {}
        )

        numerical_stats = profile.get("numerical_stats", {}) or {}
        outlier_info = profile.get("outliers", {}) or {}

        n_rows = int(overview.get("n_rows", 0) or 0)
        n_cols = int(overview.get("n_cols", 0) or 0)

        # --------------------------------------------------------------
        # 1. OBJECTIVE INTERPRETATION
        # --------------------------------------------------------------

        target_col, mentioned_cols = self._match_objective_columns(
            objective=objective,
            column_names=column_names,
        )

        # --------------------------------------------------------------
        # 2. DATA QUALITY WARNINGS
        # --------------------------------------------------------------

        warnings: list[str] = []

        missing_section = profile.get("missing", {}) or {}

        total_missing = int(
            missing_section.get("total_missing", 0) or 0
        )

        total_missing_pct = float(
            missing_section.get("total_missing_pct", 0.0) or 0.0
        )

        if total_missing > 0:
            cols_with_missing = [
                column
                for column, info in missing_info.items()
                if (info.get("missing_count", 0) or 0) > 0
            ]

            preview = ", ".join(cols_with_missing[:4])

            if len(cols_with_missing) > 4:
                preview += "..."

            warnings.append(
                f"Dataset contains {total_missing} missing values "
                f"({total_missing_pct:.2f}% of total cells) "
                f"across {len(cols_with_missing)} column(s)"
                f"{': ' + preview if preview else ''}."
            )

        duplicates_section = profile.get("duplicates", {}) or {}

        n_duplicates = int(
            duplicates_section.get("n_duplicates", 0) or 0
        )

        duplicate_pct = float(
            duplicates_section.get("duplicate_pct", 0.0) or 0.0
        )

        if n_duplicates > 0:
            warnings.append(
                f"Dataset contains {n_duplicates} duplicate rows "
                f"({duplicate_pct:.2f}%). "
                f"Duplicate handling should be considered before analysis."
            )

        if constant_cols:
            warnings.append(
                "Constant columns detected: "
                f"{', '.join(sorted(constant_cols))}. "
                "These columns are excluded from comparative analysis."
            )

        outlier_cols = [
            column
            for column, info in outlier_info.items()
            if (info.get("n_potential_outliers", 0) or 0) > 0
        ]

        if outlier_cols:
            preview = ", ".join(outlier_cols[:4])

            if len(outlier_cols) > 4:
                preview += "..."

            warnings.append(
                f"Potential outliers identified in: {preview}. "
                "Relevant analyses may require sensitivity checking."
            )

        # --------------------------------------------------------------
        # 3. CANDIDATE VARIABLES
        # --------------------------------------------------------------

        numerical_candidates = [
            column
            for column in column_names
            if column_types.get(column) == "numerical"
            and column not in constant_cols
            and self._missing_percentage(
                missing_info,
                column,
            ) < 60.0
        ]

        categorical_candidates = [
            column
            for column in column_names
            if column_types.get(column) == "categorical"
            and column not in constant_cols
            and self._missing_percentage(
                missing_info,
                column,
            ) < 60.0
            and self._is_reasonable_categorical(
                column=column,
                mentioned_cols=mentioned_cols,
                high_card_cols=high_card_cols,
                cardinality=cardinality,
            )
        ]

        # --------------------------------------------------------------
        # 4. SELECTED COLUMNS
        # --------------------------------------------------------------

        selected_columns: list[SelectedColumn] = []
        selected_set: set[str] = set()

        # Only explicit targets become targets.
        if target_col:
            selected_columns.append(
                SelectedColumn(
                    column=target_col,
                    data_type=column_types.get(
                        target_col,
                        "unknown",
                    ),
                    reason=(
                        "Explicit target variable identified from "
                        "the user's analytical objective."
                    ),
                    is_target=True,
                )
            )

            selected_set.add(target_col)

        # Explicitly mentioned variables.
        for column in mentioned_cols:
            if column in selected_set:
                continue

            if column not in column_names:
                continue

            selected_columns.append(
                SelectedColumn(
                    column=column,
                    data_type=column_types.get(
                        column,
                        "unknown",
                    ),
                    reason=(
                        "Column explicitly referenced in the "
                        "user investigation objective."
                    ),
                    is_target=False,
                )
            )

            selected_set.add(column)

        # Add numerical variables when useful.
        for column in numerical_candidates:
            if column in selected_set:
                continue

            if len(selected_columns) >= 10:
                break

            selected_columns.append(
                SelectedColumn(
                    column=column,
                    data_type="numerical",
                    reason=(
                        "Numerical variable suitable for "
                        "distribution or relationship analysis."
                    ),
                    is_target=False,
                )
            )

            selected_set.add(column)

        # Add reasonable categorical variables.
        for column in categorical_candidates:
            if column in selected_set:
                continue

            if len(selected_columns) >= 12:
                break

            n_unique = cardinality.get(column, {}).get(
                "n_unique",
                "N/A",
            )

            selected_columns.append(
                SelectedColumn(
                    column=column,
                    data_type="categorical",
                    reason=(
                        f"Categorical variable with {n_unique} "
                        "distinct categories suitable for "
                        "group or association analysis."
                    ),
                    is_target=False,
                )
            )

            selected_set.add(column)

        # --------------------------------------------------------------
        # 5. TASK GENERATION
        # --------------------------------------------------------------

        tasks: list[InvestigationTask] = []

        task_counter = 1

        def next_task_id() -> str:
            nonlocal task_counter

            task_id = f"task_{task_counter}"
            task_counter += 1

            return task_id

        # --------------------------------------------------------------
        # TASK A — MISSING DATA
        # --------------------------------------------------------------

        if total_missing > 0:
            missing_columns = [
                column
                for column, info in missing_info.items()
                if (info.get("missing_count", 0) or 0) > 0
            ]

            tasks.append(
                InvestigationTask(
                    task_id=next_task_id(),
                    task_type="missing_value_investigation",
                    task="Investigate missing data patterns",
                    description=(
                        "Assess missingness patterns and completeness "
                        "before downstream statistical analysis."
                    ),
                    method=(
                        "Missingness pattern analysis "
                        "& completeness audit"
                    ),
                    columns=missing_columns,
                    reason=(
                        "Unaddressed missing values can affect "
                        "subsequent statistical analysis."
                    ),
                    priority=(
                        "high"
                        if total_missing_pct > 10
                        else "medium"
                    ),
                    notes=[
                        "Missing-value handling may be required "
                        "before later statistical analysis."
                    ],
                )
            )

        # --------------------------------------------------------------
        # TASK B — DISTRIBUTIONS
        # --------------------------------------------------------------

        distribution_candidates = self._prioritized_numerical_columns(
            numerical_candidates=numerical_candidates,
            target_col=target_col,
            mentioned_cols=mentioned_cols,
        )

        for numerical_column in distribution_candidates[:3]:

            stats = numerical_stats.get(
                numerical_column,
                {},
            ) or {}

            valid_n = stats.get(
                "n_valid",
                n_rows,
            )

            tasks.append(
                InvestigationTask(
                    task_id=next_task_id(),
                    task_type="distribution_analysis",
                    task=(
                        f"Analyze distribution of "
                        f"{numerical_column}"
                    ),
                    description=(
                        f"Inspect central tendency, dispersion, "
                        f"and distribution shape of "
                        f"'{numerical_column}'."
                    ),
                    method=(
                        "Descriptive summary & distribution assessment"
                    ),
                    columns=[numerical_column],
                    column_a=numerical_column,
                    reason=(
                        "Understanding the distribution helps "
                        "select suitable downstream statistical methods."
                    ),
                    priority=(
                        "high"
                        if numerical_column == target_col
                        else "medium"
                    ),
                    notes=[
                        f"Sample size: {valid_n} valid observations."
                    ],
                )
            )

        # --------------------------------------------------------------
        # TASK C — OUTLIER SENSITIVITY
        # --------------------------------------------------------------

        sensitivity_columns = self._prioritized_outlier_columns(
            outlier_info=outlier_info,
            target_col=target_col,
            mentioned_cols=mentioned_cols,
        )

        for numerical_column in sensitivity_columns[:3]:

            outlier_data = outlier_info.get(
                numerical_column,
                {},
            ) or {}

            n_outliers = int(
                outlier_data.get(
                    "n_potential_outliers",
                    0,
                )
                or 0
            )

            outlier_pct = float(
                outlier_data.get(
                    "outlier_pct",
                    0.0,
                )
                or 0.0
            )

            if n_outliers <= 0:
                continue

            tasks.append(
                InvestigationTask(
                    task_id=next_task_id(),
                    task_type="outlier_sensitivity",
                    task=(
                        f"Sensitivity analysis on "
                        f"{numerical_column}"
                    ),
                    description=(
                        f"Assess whether potential outliers in "
                        f"'{numerical_column}' could influence "
                        "later analytical conclusions."
                    ),
                    method="IQR-based sensitivity analysis",
                    columns=[numerical_column],
                    column_a=numerical_column,
                    reason=(
                        f"The profiler identified {n_outliers} "
                        f"potential outlier observation(s) "
                        f"({outlier_pct:.2f}%)."
                    ),
                    priority=(
                        "high"
                        if numerical_column == target_col
                        else "medium"
                    ),
                    notes=[
                        "Do not remove observations automatically.",
                        "Actual sensitivity calculations belong to Phase 3.",
                    ],
                )
            )

        # --------------------------------------------------------------
        # TASK D — NUMERICAL RELATIONSHIPS
        # --------------------------------------------------------------

        numerical_pairs = self._build_numerical_pairs(
            numerical_candidates=numerical_candidates,
            target_col=target_col,
            mentioned_cols=mentioned_cols,
        )

        for column_a, column_b in numerical_pairs[:6]:

            skew_a = abs(
                float(
                    numerical_stats.get(
                        column_a,
                        {},
                    ).get("skewness", 0.0)
                    or 0.0
                )
            )

            skew_b = abs(
                float(
                    numerical_stats.get(
                        column_b,
                        {},
                    ).get("skewness", 0.0)
                    or 0.0
                )
            )

            has_outliers = (
                self._has_outliers(
                    outlier_info,
                    column_a,
                )
                or self._has_outliers(
                    outlier_info,
                    column_b,
                )
            )

            # Method selection only.
            # No statistical calculation is performed.
            if skew_a > 1.0 or skew_b > 1.0 or has_outliers:
                method = "Spearman rank correlation"

                method_reason = (
                    "Spearman rank correlation is planned because "
                    "distributional skewness or potential outliers "
                    "may make a rank-based relationship analysis "
                    "more appropriate."
                )
            else:
                method = "Pearson correlation"

                method_reason = (
                    "Pearson correlation is planned for the "
                    "numerical relationship."
                )

            high_priority = (
                target_col in {column_a, column_b}
                or (
                    column_a in mentioned_cols
                    and column_b in mentioned_cols
                )
            )

            tasks.append(
                InvestigationTask(
                    task_id=next_task_id(),
                    task_type="numerical_vs_numerical",
                    task=(
                        f"Analyze relationship: "
                        f"{column_a} vs {column_b}"
                    ),
                    description=(
                        f"Measure the association between "
                        f"'{column_a}' and '{column_b}'."
                    ),
                    method=method,
                    columns=[column_a, column_b],
                    column_a=column_a,
                    column_b=column_b,
                    reason=(
                        f"Both variables are numerical. "
                        f"{method_reason}"
                    ),
                    priority=(
                        "high"
                        if high_priority
                        else "medium"
                    ),
                    notes=self._pair_notes(
                        missing_info,
                        column_a,
                        column_b,
                    ),
                )
            )

        # --------------------------------------------------------------
        # TASK E — GROUP COMPARISONS
        # --------------------------------------------------------------

        group_tasks_added = 0

        group_target_columns = self._group_target_columns(
            target_col=target_col,
            mentioned_cols=mentioned_cols,
            numerical_candidates=numerical_candidates,
        )

        for categorical_column in categorical_candidates:

            n_unique = int(
                cardinality.get(
                    categorical_column,
                    {},
                ).get("n_unique", 0)
                or 0
            )

            if n_unique < 2 or n_unique > 20:
                continue

            for numerical_column in group_target_columns:

                if numerical_column not in numerical_candidates:
                    continue

                skew = abs(
                    float(
                        numerical_stats.get(
                            numerical_column,
                            {},
                        ).get("skewness", 0.0)
                        or 0.0
                    )
                )

                if n_unique == 2:

                    if skew > 1.0:
                        method = "Mann-Whitney U test"

                        method_reason = (
                            "A rank-based two-group comparison "
                            "is planned because the numerical "
                            "variable is substantially skewed."
                        )
                    else:
                        method = "Welch's t-test"

                        method_reason = (
                            "Welch's t-test is planned for the "
                            "two-group numerical comparison."
                        )

                    task_type = "two_groups_vs_numerical"

                else:

                    if skew > 1.0:
                        method = "Kruskal-Wallis test"

                        method_reason = (
                            "Kruskal-Wallis is planned as a "
                            "non-parametric multi-group comparison."
                        )
                    else:
                        method = "One-way ANOVA"

                        method_reason = (
                            "One-way ANOVA is planned for comparing "
                            "the numerical variable across multiple groups."
                        )

                    task_type = "multiple_groups_vs_numerical"

                high_priority = (
                    categorical_column in mentioned_cols
                    or numerical_column == target_col
                )

                tasks.append(
                    InvestigationTask(
                        task_id=next_task_id(),
                        task_type=task_type,
                        task=(
                            f"Compare {categorical_column} "
                            f"groups on {numerical_column}"
                        ),
                        description=(
                            f"Evaluate whether '{numerical_column}' "
                            f"differs across groups of "
                            f"'{categorical_column}'."
                        ),
                        method=method,
                        columns=[
                            categorical_column,
                            numerical_column,
                        ],
                        group_column=categorical_column,
                        value_column=numerical_column,
                        reason=(
                            f"'{categorical_column}' contains "
                            f"{n_unique} groups. "
                            f"{method_reason}"
                        ),
                        priority=(
                            "high"
                            if high_priority
                            else "medium"
                        ),
                        notes=[
                            f"Grouping levels: {n_unique}."
                        ],
                    )
                )

                group_tasks_added += 1

                # Avoid generating an unnecessarily huge plan.
                if group_tasks_added >= 4:
                    break

            if group_tasks_added >= 4:
                break

        # --------------------------------------------------------------
        # TASK F — CATEGORICAL ASSOCIATIONS
        # --------------------------------------------------------------

        categorical_pairs = self._build_categorical_pairs(
            categorical_candidates=categorical_candidates,
            mentioned_cols=mentioned_cols,
            cardinality=cardinality,
        )

        for column_a, column_b in categorical_pairs[:4]:

            u_a = int(
                cardinality.get(
                    column_a,
                    {},
                ).get("n_unique", 0)
                or 0
            )

            u_b = int(
                cardinality.get(
                    column_b,
                    {},
                ).get("n_unique", 0)
                or 0
            )

            explicitly_mentioned = (
                column_a in mentioned_cols
                and column_b in mentioned_cols
            )

            tasks.append(
                InvestigationTask(
                    task_id=next_task_id(),
                    task_type="categorical_vs_categorical",
                    task=(
                        f"Analyze association: "
                        f"{column_a} vs {column_b}"
                    ),
                    description=(
                        f"Investigate the association between "
                        f"categorical variables '{column_a}' "
                        f"and '{column_b}'."
                    ),
                    method="Chi-square test of independence",
                    columns=[column_a, column_b],
                    column_a=column_a,
                    column_b=column_b,
                    reason=(
                        f"Both variables are categorical with "
                        f"{u_a} and {u_b} levels respectively."
                    ),
                    priority=(
                        "high"
                        if explicitly_mentioned
                        else "low"
                    ),
                    notes=[
                        "Cramér's V can be calculated during Phase 3 "
                        "to measure association effect size."
                    ],
                )
            )

        # --------------------------------------------------------------
        # 6. REMOVE DUPLICATE TASKS
        # --------------------------------------------------------------

        tasks = self._deduplicate_tasks(tasks)

        # --------------------------------------------------------------
        # 7. LIMIT TOTAL TASKS
        # --------------------------------------------------------------

        # Keep plans manageable.
        # Prioritized tasks are retained first.
        tasks = self._limit_tasks(tasks, max_tasks=15)

        # --------------------------------------------------------------
        # 8. INTERPRETED OBJECTIVE
        # --------------------------------------------------------------

        if target_col:
            interpreted_goal = (
                f"Investigate relationships, group differences, "
                f"and relevant analytical factors associated with "
                f"'{target_col}'."
            )

        elif mentioned_cols:
            interpreted_goal = (
                "Investigate statistical relationships and "
                "distributional characteristics involving: "
                f"{', '.join(mentioned_cols)}."
            )

        else:
            interpreted_goal = (
                "Perform exploratory statistical investigation "
                "across suitable numerical distributions, "
                "relationships, and categorical group comparisons."
            )

        # --------------------------------------------------------------
        # 9. PLANNER REASONING
        # --------------------------------------------------------------

        reasoning = [
            f"Dataset dimensions: {n_rows} rows and {n_cols} columns.",
            (
                f"Identified {len(numerical_candidates)} eligible "
                "numerical variable(s) and "
                f"{len(categorical_candidates)} eligible "
                "categorical variable(s)."
            ),
            (
                f"Objective interpretation: {interpreted_goal}"
            ),
            (
                f"Generated {len(tasks)} prioritized investigation "
                "task(s) without executing statistical analysis."
            ),
        ]

        # --------------------------------------------------------------
        # 10. EXPECTED OUTPUTS
        # --------------------------------------------------------------

        expected_outputs = [
            "Validated statistical analysis results from Phase 3.",
            "Relationship measures for selected numerical variables.",
            "Group-comparison results where appropriate.",
            "Categorical association measures where appropriate.",
            "Outlier sensitivity assessments where justified.",
        ]

        # --------------------------------------------------------------
        # 11. CLEAN TASK NOTES
        # --------------------------------------------------------------

        for task in tasks:

            if task.notes:
                task.notes = [
                    note
                    for note in task.notes
                    if note
                ]

        # --------------------------------------------------------------
        # 12. RESPONSE
        # --------------------------------------------------------------

        return PlannerResponse(
            objective=objective,
            interpreted_goal=interpreted_goal,
            target_column=target_col,
            selected_columns=selected_columns,
            analysis_tasks=tasks,
            tasks=tasks,
            reasoning=reasoning,
            warnings=warnings,
            expected_outputs=expected_outputs,
            planner_mode="deterministic",
            planner_status="success",
            planner_model=None,
            planner_provider="deterministic",
            dataset_summary={
                "n_rows": n_rows,
                "n_cols": n_cols,
                "n_numerical": len(numerical_candidates),
                "n_categorical": len(categorical_candidates),
                "total_missing": total_missing,
                "n_duplicates": n_duplicates,
                "planner_version": self.PLANNER_VERSION,
                "planner_mode": "deterministic",
                "planner_status": "success",
                "provider": "deterministic",
            },
        )

    # ------------------------------------------------------------------
    # OBJECTIVE / TARGET DETECTION
    # ------------------------------------------------------------------

    def _match_objective_columns(
        self,
        objective: str,
        column_names: list[str],
    ) -> tuple[Optional[str], list[str]]:
        """
        Identify explicitly mentioned columns and an explicitly
        identified target.

        IMPORTANT FIX:
            A merely mentioned column is NOT automatically considered
            the target.

        Example:

            "Investigate the relationship between Rating and Price"

        produces:

            mentioned = ["Rating", "Price"]
            target = None

        Example:

            "Find factors affecting Selling Price"

        produces:

            target = "Selling Price"
        """

        objective_lower = objective.lower()

        mentioned: list[str] = []

        # --------------------------------------------------------------
        # Match actual column names.
        # --------------------------------------------------------------

        sorted_columns = sorted(
            column_names,
            key=lambda value: len(value),
            reverse=True,
        )

        for column in sorted_columns:

            column_lower = column.lower()

            # Exact phrase / substring match.
            if column_lower in objective_lower:

                if column not in mentioned:
                    mentioned.append(column)

                continue

            # Token-level match.
            tokens = [
                token
                for token in re.split(
                    r"[_\s\-]+",
                    column_lower,
                )
                if len(token) >= 4
            ]

            if any(
                re.search(
                    rf"\b{re.escape(token)}\b",
                    objective_lower,
                )
                for token in tokens
            ):
                if column not in mentioned:
                    mentioned.append(column)

        # --------------------------------------------------------------
        # Explicit target patterns.
        # --------------------------------------------------------------

        target_patterns = [
            # "factors affecting / related to / associated with Selling Price"
            r"(?:factors?\s+(?:affecting|affect|for|related\s+to|associated\s+with|influencing|driving|in))\s+(.+?)(?:\.|$)",

            # "associated with salary" / "related to salary"
            r"(?:related\s+to|associated\s+with)\s+(.+?)(?:\.|$)",

            # "predict Selling Price"
            r"(?:predict|predicting)\s+(.+?)(?:\.|$)",

            # "impact on Selling Price"
            r"(?:impact\s+on)\s+(.+?)(?:\.|$)",

            # "target is Selling Price"
            r"(?:target\s+(?:is|variable\s+is))\s+(.+?)(?:\.|$)",

            # "dependent variable is Selling Price"
            r"(?:dependent\s+variable\s+(?:is|=))\s+(.+?)(?:\.|$)",

            # "focus on Selling Price"
            r"(?:focus\s+on)\s+(.+?)(?:\.|$)",
        ]

        target: Optional[str] = None

        for pattern in target_patterns:

            match = re.search(
                pattern,
                objective_lower,
            )

            if not match:
                continue

            extracted = match.group(1).strip()

            # Find the best matching real column.
            candidates = [
                column
                for column in column_names
                if column.lower() in extracted
                or extracted in column.lower()
            ]

            if candidates:
                target = max(
                    candidates,
                    key=len,
                )
                break

        # --------------------------------------------------------------
        # DO NOT infer target from first mentioned column.
        # --------------------------------------------------------------

        return target, mentioned

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _missing_percentage(
        missing_info: dict[str, Any],
        column: str,
    ) -> float:
        info = missing_info.get(column, {}) or {}

        return float(
            info.get("missing_pct", 0.0) or 0.0
        )

    @staticmethod
    def _is_reasonable_categorical(
        column: str,
        mentioned_cols: list[str],
        high_card_cols: set[str],
        cardinality: dict[str, Any],
    ) -> bool:

        # Explicitly mentioned variables can be retained.
        if column in mentioned_cols:
            return True

        if column in high_card_cols:
            return False

        info = cardinality.get(column, {}) or {}

        if info.get("is_high_cardinality", False):
            return False

        n_unique = int(
            info.get("n_unique", 0) or 0
        )

        return 2 <= n_unique <= 20

    @staticmethod
    def _prioritized_numerical_columns(
        numerical_candidates: list[str],
        target_col: Optional[str],
        mentioned_cols: list[str],
    ) -> list[str]:

        result: list[str] = []

        # Explicit target first.
        if (
            target_col
            and target_col in numerical_candidates
        ):
            result.append(target_col)

        # Explicitly mentioned numerical columns next.
        for column in mentioned_cols:

            if column in numerical_candidates and column not in result:
                result.append(column)

        # Remaining numerical columns.
        for column in numerical_candidates:

            if column not in result:
                result.append(column)

        return result

    @staticmethod
    def _prioritized_outlier_columns(
        outlier_info: dict[str, Any],
        target_col: Optional[str],
        mentioned_cols: list[str],
    ) -> list[str]:

        candidates = [
            column
            for column, info in outlier_info.items()
            if (info.get("n_potential_outliers", 0) or 0) > 0
        ]

        def priority(column: str) -> tuple[int, float]:

            is_target = int(column == target_col)
            is_mentioned = int(column in mentioned_cols)

            outlier_pct = float(
                outlier_info.get(
                    column,
                    {},
                ).get("outlier_pct", 0.0)
                or 0.0
            )

            return (
                is_target * 2 + is_mentioned,
                outlier_pct,
            )

        return sorted(
            candidates,
            key=priority,
            reverse=True,
        )

    @staticmethod
    def _build_numerical_pairs(
        numerical_candidates: list[str],
        target_col: Optional[str],
        mentioned_cols: list[str],
    ) -> list[tuple[str, str]]:

        pairs: list[tuple[str, str]] = []

        seen: set[frozenset[str]] = set()

        # Target-related pairs first.
        if target_col in numerical_candidates:

            for column in numerical_candidates:

                if column == target_col:
                    continue

                pair = frozenset(
                    [target_col, column]
                )

                if pair in seen:
                    continue

                seen.add(pair)

                pairs.append(
                    (target_col, column)
                )

        # Pairs where both variables were mentioned.
        mentioned_numerical = [
            column
            for column in mentioned_cols
            if column in numerical_candidates
        ]

        for index, column_a in enumerate(
            mentioned_numerical
        ):

            for column_b in mentioned_numerical[index + 1:]:

                pair = frozenset(
                    [column_a, column_b]
                )

                if pair in seen:
                    continue

                seen.add(pair)

                pairs.append(
                    (column_a, column_b)
                )

        # Remaining pairs.
        for index, column_a in enumerate(
            numerical_candidates
        ):

            for column_b in numerical_candidates[index + 1:]:

                pair = frozenset(
                    [column_a, column_b]
                )

                if pair in seen:
                    continue

                seen.add(pair)

                pairs.append(
                    (column_a, column_b)
                )

        return pairs

    @staticmethod
    def _group_target_columns(
        target_col: Optional[str],
        mentioned_cols: list[str],
        numerical_candidates: list[str],
    ) -> list[str]:

        result: list[str] = []

        if target_col in numerical_candidates:
            result.append(target_col)

        for column in mentioned_cols:

            if (
                column in numerical_candidates
                and column not in result
            ):
                result.append(column)

        for column in numerical_candidates:

            if column not in result:
                result.append(column)

        return result[:3]

    @staticmethod
    def _build_categorical_pairs(
        categorical_candidates: list[str],
        mentioned_cols: list[str],
        cardinality: dict[str, Any],
    ) -> list[tuple[str, str]]:

        pairs: list[tuple[str, str]] = []

        seen: set[frozenset[str]] = set()

        # Mentioned categorical variables first.
        mentioned = [
            column
            for column in mentioned_cols
            if column in categorical_candidates
        ]

        ordered = mentioned + [
            column
            for column in categorical_candidates
            if column not in mentioned
        ]

        for index, column_a in enumerate(ordered):

            for column_b in ordered[index + 1:]:

                pair = frozenset(
                    [column_a, column_b]
                )

                if pair in seen:
                    continue

                unique_a = int(
                    cardinality.get(
                        column_a,
                        {},
                    ).get("n_unique", 0)
                    or 0
                )

                unique_b = int(
                    cardinality.get(
                        column_b,
                        {},
                    ).get("n_unique", 0)
                    or 0
                )

                # Keep contingency tables reasonably sized.
                if not (
                    2 <= unique_a <= 15
                    and 2 <= unique_b <= 15
                ):
                    continue

                seen.add(pair)

                pairs.append(
                    (column_a, column_b)
                )

        return pairs

    @staticmethod
    def _has_outliers(
        outlier_info: dict[str, Any],
        column: str,
    ) -> bool:

        return bool(
            (
                outlier_info.get(
                    column,
                    {},
                ).get(
                    "n_potential_outliers",
                    0,
                )
                or 0
            )
            > 0
        )

    @staticmethod
    def _pair_notes(
        missing_info: dict[str, Any],
        column_a: str,
        column_b: str,
    ) -> Optional[list[str]]:

        notes: list[str] = []

        missing_a = (
            missing_info.get(
                column_a,
                {},
            ).get(
                "missing_count",
                0,
            )
            or 0
        )

        missing_b = (
            missing_info.get(
                column_b,
                {},
            ).get(
                "missing_count",
                0,
            )
            or 0
        )

        if missing_a > 0 or missing_b > 0:
            notes.append(
                "Missing-value handling may be required."
            )

        return notes or None

    @staticmethod
    def _deduplicate_tasks(
        tasks: list[InvestigationTask],
    ) -> list[InvestigationTask]:

        result: list[InvestigationTask] = []

        seen: set[tuple[Any, ...]] = set()

        for task in tasks:

            columns = tuple(
                sorted(
                    task.columns or []
                )
            )

            key = (
                task.task_type,
                columns,
                task.method,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(task)

        # Re-number task IDs after deduplication.
        for index, task in enumerate(
            result,
            start=1,
        ):
            task.task_id = f"task_{index}"

        return result

    @staticmethod
    def _limit_tasks(
        tasks: list[InvestigationTask],
        max_tasks: int = 15,
    ) -> list[InvestigationTask]:

        priority_order = {
            "high": 0,
            "medium": 1,
            "low": 2,
        }

        tasks = sorted(
            tasks,
            key=lambda task: (
                priority_order.get(
                    task.priority,
                    3,
                ),
                task.task_id,
            ),
        )

        limited = tasks[:max_tasks]

        # Re-number IDs.
        for index, task in enumerate(
            limited,
            start=1,
        ):
            task.task_id = f"task_{index}"

        return limited