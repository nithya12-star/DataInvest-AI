"""
backend/app/services/demo_service.py
====================================
Offline Demonstration Engine for DataInvest AI.

Responsibilities:
- Provide a 100% offline, reproducible demonstration using synthetic demo data.
- Run deterministic profiling, heuristic task planning, SciPy statistical analysis,
  Critic assumption auditing, controlled re-investigation, and insight synthesis.
- Strictly label all outputs as simulated/deterministic offline demo.
- Never describe demo outputs as genuine LLM responses.
- Never call Gemini, external APIs, or incur API costs.
- Completely isolated from research mode (paired experiments strictly use LLM planning).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from app.modules import profiler
from app.services import dataset_store
from app.agents.planner_agent import PlannerAgent
from app.services.analysis_service import get_analysis_service
from app.services.critic_service import get_critic_service
from app.services.verification_service import get_verification_service
from app.services.insight_service import get_insight_service
from app.services.evaluation_service import evaluation_service
from app.schemas.analysis_schema import PlanAnalysisRequest
from app.schemas.critic_schema import CriticRequest
from app.schemas.verification_schema import VerificationRequest
from app.schemas.insight_schema import InsightRequest

logger = logging.getLogger("datainvest.demo_service")

DEMO_DATASET_ID = "synthetic_demo_data.csv"
DEMO_DISCLAIMER = (
    "OFFLINE DEMONSTRATION ONLY: All outputs in this demo are deterministic and simulated "
    "for offline evaluation without Gemini API calls. Never describe this output as a genuine "
    "LLM response. Research experiments strictly enforce genuine Gemini-based planning."
)


def create_synthetic_demo_dataframe(n_samples: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data designed to safely demonstrate all phases of DataInvest AI:
    - study_hours: continuous, normal
    - exam_score: continuous, linear relation with study_hours + 4 artificial leverage outliers
    - attendance_rate: continuous, highly skewed (to demonstrate normality/skew check)
    - tutoring_group: categorical ('Enrolled', 'Standard')
    - academic_tier: categorical ('Tier_A', 'Tier_B', 'Tier_C')
    """
    rng = np.random.default_rng(seed)

    study_hours = rng.normal(loc=25.0, scale=6.0, size=n_samples).clip(5.0, 50.0)
    # exam_score has strong linear relationship with study_hours
    noise = rng.normal(loc=0.0, scale=4.0, size=n_samples)
    exam_score = (40.0 + 1.2 * study_hours + noise).clip(20.0, 100.0)

    # Inject 4 distinct leverage outliers into exam_score to trigger Critic leverage audit
    exam_score[5] = 98.0
    study_hours[5] = 10.0  # low study, very high score
    exam_score[15] = 25.0
    study_hours[15] = 48.0  # high study, very low score
    exam_score[25] = 99.0
    study_hours[25] = 12.0
    exam_score[35] = 22.0
    study_hours[35] = 46.0

    # attendance_rate: exponential/skewed distribution
    attendance_rate = (100.0 - rng.exponential(scale=12.0, size=n_samples)).clip(30.0, 100.0)

    tutoring = rng.choice(["Enrolled", "Standard"], size=n_samples, p=[0.4, 0.6])
    academic_tier = rng.choice(["Tier_A", "Tier_B", "Tier_C"], size=n_samples, p=[0.25, 0.50, 0.25])

    df = pd.DataFrame({
        "student_id": [f"STU_{i:04d}" for i in range(1, n_samples + 1)],
        "study_hours": np.round(study_hours, 1),
        "exam_score": np.round(exam_score, 1),
        "attendance_rate": np.round(attendance_rate, 1),
        "tutoring_group": tutoring,
        "academic_tier": academic_tier,
    })

    return df


class DemoService:
    """Orchestrates 100% offline, synthetic demonstrations of DataInvest AI."""

    def __init__(self):
        self.dataset_id = DEMO_DATASET_ID

    def get_or_create_demo_dataset(self) -> pd.DataFrame:
        """Retrieve cached synthetic demo dataset or generate and register it."""
        cached = dataset_store.resolve_dataset(self.dataset_id)
        if cached is not None and not cached.empty:
            return cached

        df = create_synthetic_demo_dataframe()
        dataset_store.store_dataset(self.dataset_id, df)
        return df

    def run_offline_demo(
        self,
        objective: str = "Investigate the relationship between study hours, attendance, and exam scores.",
    ) -> Dict[str, Any]:
        """
        Execute full pipeline deterministically in offline demo mode.
        Zero Gemini calls. Clear simulated/mocked markers on all outputs.
        """
        logger.info("Executing offline deterministic demonstration with synthetic dataset.")
        df = self.get_or_create_demo_dataset()

        # Phase 1: Profiling
        profile = profiler.profile_dataset(df)
        profile["filename"] = self.dataset_id
        profile["dataset_id"] = self.dataset_id
        profile["is_offline_demo"] = True

        # Phase 2: Deterministic Planner (Heuristic, no LLM)
        planner = PlannerAgent(use_llm=False)
        plan_response = planner.plan(profile=profile, objective=objective)
        plan_dict = plan_response.model_dump()
        plan_dict["planner_mode"] = "deterministic_simulated"
        plan_dict["is_offline_demo"] = True
        plan_dict["disclaimer"] = DEMO_DISCLAIMER

        # Phase 3: Analysis Execution (SciPy on synthetic data)
        analysis_svc = get_analysis_service()
        analysis_req = PlanAnalysisRequest(
            dataset_id=self.dataset_id,
            tasks=plan_response.analysis_tasks,
        )
        analysis_resp = analysis_svc.execute_plan(analysis_req)
        analysis_dict = analysis_resp.model_dump()
        analysis_dict["is_offline_demo"] = True

        # Phase 4: Critic Review (Deterministic assumption audits)
        critic_svc = get_critic_service()
        critic_req = CriticRequest(
            analysis_response=analysis_resp,
            dataset_profile=profile,
            investigation_plan=plan_response,
            dataset_id=self.dataset_id,
        )
        critic_resp = critic_svc.run_review(critic_req)
        critic_dict = critic_resp.model_dump()
        critic_dict["is_offline_demo"] = True

        # Phase 5: Verification & Controlled Reinvestigation
        verif_svc = get_verification_service()
        verif_req = VerificationRequest(
            analysis_response=analysis_resp,
            critic_response=critic_resp,
            dataset_profile=profile,
            investigation_plan=plan_response,
            dataset_id=self.dataset_id,
            max_reinvestigations=1,
        )
        verif_resp = verif_svc.process_verification(verif_req)
        verif_dict = verif_resp.model_dump()
        verif_dict["is_offline_demo"] = True

        # Phase 6: Insights Synthesis
        insight_svc = get_insight_service()
        insight_req = InsightRequest(
            verification_response=verif_resp,
            dataset_profile=profile,
            investigation_plan=plan_response,
            dataset_id=self.dataset_id,
        )
        insight_resp = insight_svc.generate_insights(insight_req)
        insight_dict = insight_resp.model_dump()
        insight_dict["is_offline_demo"] = True

        # Phase 7: System Evaluation Scorecard
        eval_resp = evaluation_service.evaluate_pipeline(
            profile=profile,
            plan=plan_dict,
            analysis=analysis_dict,
            critic=critic_dict,
            verification=verif_dict,
            insights=insight_dict,
        )
        eval_dict = eval_resp.model_dump()
        eval_dict["is_offline_demo"] = True

        return {
            "is_offline_demo": True,
            "mode": "offline_deterministic_demo",
            "disclaimer": DEMO_DISCLAIMER,
            "dataset_id": self.dataset_id,
            "objective": objective,
            "profile": profile,
            "plan": plan_dict,
            "analysis": analysis_dict,
            "critic": critic_dict,
            "verification": verif_dict,
            "insights": insight_dict,
            "evaluation": eval_dict,
        }


# Singleton instance
demo_service = DemoService()
