"""
backend/app/services/planner_service.py
========================================
Service layer orchestrating the Planner Agent.
Validates inputs, manages planning lifecycle, and returns strongly typed PlannerResponse.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from app.agents.planner_agent import PlannerAgent
from app.schemas.planner_schema import PlannerRequest, PlannerResponse
from app.services.gemini_client import GeminiClient

logger = logging.getLogger("datainvest.planner")


def create_analysis_plan(
    profile: Optional[dict[str, Any]],
    objective: str = "Investigate this dataset.",
    dataset_id: Optional[str] = None,
    preferences: Optional[dict[str, Any]] = None,
    use_llm: Optional[bool] = None,
    gemini_client: Optional[GeminiClient] = None,
    is_research_run: Optional[bool] = None,
) -> PlannerResponse:
    """
    Generate an investigation plan given a dataset profile and user objective.

    Parameters
    ----------
    profile : dict
        The structured Phase 1 dataset profile.
    objective : str
        The user's analytical objective in natural language.
    dataset_id : str, optional
        Optional identifier or filename for provenance tracking.
    preferences : dict, optional
        Optional execution constraints or options. Can specify preferences["research_run"] = True
        or preferences["use_llm"] = True.
    use_llm : bool, optional
        Explicitly enable/disable Gemini LLM planning. Overridden to True if is_research_run is True.
    gemini_client : GeminiClient, optional
        Optional pre-configured GeminiClient instance (e.g., with MockTransport for testing).
    is_research_run : bool, optional
        Explicit research-run marker. When True, forces use_llm=True unconditionally, regardless
        of any conflicting use_llm=False preference or environment setting.

    Returns
    -------
    PlannerResponse
        Validated Pydantic response model containing prioritized tasks.

    Raises
    ------
    ValueError
        If profile data is missing, invalid, or empty.
    GeminiClientError
        If LLM mode is active and Gemini invocation or plan validation fails (no silent fallback).
    """
    if not profile or not isinstance(profile, dict):
        logger.warning("Planner service received empty or invalid profile.")
        raise ValueError("A valid dataset profile dictionary is required to generate an analysis plan.")

    overview = profile.get("overview")
    if not overview or not isinstance(overview, dict) or overview.get("n_rows", 0) <= 0:
        logger.warning("Planner service received profile with 0 rows or missing overview.")
        raise ValueError("Dataset profile contains no valid rows or is missing dataset overview.")

    # 1. Resolve research-run flag: explicit parameter > preferences > env var > default False
    if is_research_run is None:
        if preferences and isinstance(preferences, dict):
            if "research_run" in preferences:
                is_research_run = bool(preferences["research_run"])
            elif "is_research_run" in preferences:
                is_research_run = bool(preferences["is_research_run"])
        if is_research_run is None:
            env_val = os.environ.get("DATAINVEST_RESEARCH_RUN", "").strip().lower()
            is_research_run = env_val in ("true", "1", "yes")

    # 2. Enforce genuine LLM planning
    if is_research_run:
        # Research mode: strictly require LLM planning unconditionally.
        # Never rely on environment variable default or optional preference to enable it.
        if use_llm is False:
            logger.warning(
                "Research run detected with conflicting use_llm=False. "
                "Overriding use_llm to True to enforce genuine LLM planning without fallback."
            )
        use_llm = True
    elif use_llm is None:
        # Normal intended application Planner workflow:
        # Gemini LLM planning is enforced by default. Deterministic planning must never silently substitute.
        if preferences and isinstance(preferences, dict) and "use_llm" in preferences:
            use_llm = bool(preferences["use_llm"])
        else:
            use_llm = True

    logger.info(
        "Creating analysis plan for dataset '%s' with objective: '%s' (use_llm=%s, is_research_run=%s)",
        dataset_id or overview.get("filename") or "unnamed",
        objective,
        use_llm,
        is_research_run,
    )

    agent = PlannerAgent(use_llm=use_llm, gemini_client=gemini_client)
    plan_response = agent.plan(profile=profile, objective=objective)

    logger.info(
        "Generated %d tasks (mode: %s, target: %s)",
        len(plan_response.analysis_tasks),
        plan_response.planner_mode,
        plan_response.target_column,
    )

    return plan_response
