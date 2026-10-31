"""
backend/app/services/critic_service.py
======================================
Service layer orchestrating Critic verification reviews.
"""

from __future__ import annotations

from typing import Optional
from app.agents.critic_agent import CriticAgent
from app.schemas.critic_schema import CriticRequest, CriticResponse


class CriticService:
    """Orchestrates statistical verification and finding audits."""

    def __init__(self, agent: Optional[CriticAgent] = None):
        self.agent = agent or CriticAgent()

    def run_review(self, request: CriticRequest) -> CriticResponse:
        """Execute a comprehensive verification review on Phase 3 analysis findings."""
        # Normalize input
        input_data = request.analysis_response or request.results or []
        dataset_id = request.dataset_id or (
            request.analysis_response.dataset_id if request.analysis_response else "active_dataset"
        )

        return self.agent.evaluate_plan(
            analysis_input=input_data,
            profile=request.dataset_profile,
            plan=request.investigation_plan,
            dataset_id=dataset_id,
        )


_CRITIC_SERVICE = CriticService()


def get_critic_service() -> CriticService:
    """Provide the global CriticService instance."""
    return _CRITIC_SERVICE
