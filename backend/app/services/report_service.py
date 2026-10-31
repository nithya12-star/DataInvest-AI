"""
backend/app/services/report_service.py
======================================
Phase 5 / Stage B: Report Generation and Export Service.

Orchestrates retrieval of persistent experiment runs, synthesis of structured
scientific reports via ReportAgent, and safe export to Markdown and JSON files.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from app.agents.report_agent import report_agent
from app.schemas.report_schema import ResearchReport
from app.services.run_store import run_store


class RunNotFoundError(Exception):
    """Raised when the requested experiment run ID does not exist in persistent storage."""
    pass


class InvalidRunIdError(Exception):
    """Raised when the requested run ID contains invalid characters or traversal patterns."""
    pass


def _sanitize_export_slug(text: str, max_len: int = 64) -> str:
    """Sanitize text to create safe alphanumeric filename tokens."""
    cleaned = re.sub(r"[^\w\s-]", "", text).strip().lower()
    cleaned = re.sub(r"[-\s]+", "_", cleaned)
    return cleaned[:max_len] if cleaned else "run"


_SENSITIVE_KEY_PATTERNS = {
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
    "bearer",
    "credential",
    "private_key",
}


def _redact_sensitive_fields(data: Any) -> Any:
    """Recursively redact any sensitive key or token pattern from export dictionaries."""
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            k_lower = str(k).lower()
            if any(term in k_lower for term in _SENSITIVE_KEY_PATTERNS):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = _redact_sensitive_fields(v)
        return sanitized
    elif isinstance(data, list):
        return [_redact_sensitive_fields(item) for item in data]
    return data


class ReportService:
    """Manages report synthesis and safe file export generation for experiment runs."""

    def __init__(self, run_store_instance: Optional[Any] = None):
        self.report_agent = report_agent
        self._run_store = run_store_instance

    @property
    def run_store(self):
        if self._run_store is not None:
            return self._run_store
        import app.services.run_store as rsm
        return rsm.run_store

    def get_report_for_run(self, run_id: str) -> ResearchReport:
        """
        Load a specific experiment run from disk and synthesize its structured ResearchReport.
        Guarantees that report data corresponds strictly to the selected run_id.
        """
        if not run_id or not isinstance(run_id, str):
            raise InvalidRunIdError("A valid run_id string must be provided.")

        clean_id = run_id.strip()
        if "/" in clean_id or "\\" in clean_id or ".." in clean_id:
            raise InvalidRunIdError(f"Unsafe characters detected in run_id: '{run_id}'.")

        run_data = self.run_store.get_run(clean_id)
        if not run_data:
            raise RunNotFoundError(f"Experiment run '{clean_id}' was not found in repository.")

        report = self.report_agent.generate_report(run_data)
        try:
            from app.services.experiment_logger import experiment_logger
            experiment_logger.log_event(
                run_id=clean_id,
                event_type="REPORT_GENERATED",
                phase=6,
                outcome="SUCCESS",
                details={
                    "report_id": report.report_id,
                    "dataset_name": report.dataset_context.dataset_name,
                    "verified_findings_count": len(report.final_verified_findings),
                    "caveated_findings_count": len(report.caveated_or_unresolved_findings),
                },
            )
        except Exception:
            pass

        return report

    def export_markdown(self, run_id: str) -> Tuple[str, str]:
        """
        Synthesize report and render complete GitHub-Flavored Markdown export.
        Returns:
            Tuple of (markdown_text, safe_filename)
        """
        report = self.get_report_for_run(run_id)
        markdown_text = self.report_agent.render_markdown(report)

        dataset_slug = _sanitize_export_slug(report.dataset_context.dataset_name)
        run_slug = _sanitize_export_slug(report.run_id)
        safe_filename = f"report_{dataset_slug}_{run_slug}.md"

        return markdown_text, safe_filename

    def export_json(self, run_id: str) -> Tuple[Dict[str, Any], str]:
        """
        Synthesize report and prepare structured JSON export preserving complete run state.
        Returns:
            Tuple of (export_payload_dict, safe_filename)
        """
        report = self.get_report_for_run(run_id)
        raw_run = self.run_store.get_run(report.run_id) or {}
        sanitized_raw_run = _redact_sensitive_fields(raw_run)

        export_payload: Dict[str, Any] = {
            "export_version": "1.0.0",
            "run_id": report.run_id,
            "generated_at": report.generated_at,
            "report": report.model_dump(),
            "raw_run_state": sanitized_raw_run,
        }

        dataset_slug = _sanitize_export_slug(report.dataset_context.dataset_name)
        run_slug = _sanitize_export_slug(report.run_id)
        safe_filename = f"export_{dataset_slug}_{run_slug}.json"

        return export_payload, safe_filename


# Global singleton service instance
report_service = ReportService()
