"""
backend/app/services/experiment_logger.py
=========================================
Structured, run-specific logging for DataInvest AI.

Responsibilities:
- Record structured audit trail events for Planner, Analysis, Critic,
  Reinvestigation, and Report phases.
- Redact credentials, API keys, and sensitive dataset rows.
- Store logs in backend/data/logs/{run_id}.jsonl without altering historical runs.
- Ensure logging failures NEVER crash experiments or pipelines.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("datainvest.experiment_logger")

APPROVED_EXTERNAL_RESULTS_ROOTS = [
    Path(r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\DataInvest_Pilot_Results").resolve(),
    Path(r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\DataInvest_Research_Results").resolve(),
]
APPROVED_EXTERNAL_RESULTS_ROOT = APPROVED_EXTERNAL_RESULTS_ROOTS[0]


def validate_safe_log_dir(log_dir: Union[str, Path], allow_temp: bool = True) -> Path:
    """
    Validate event log directory safety and protect against path traversal attacks.
    Permits paths within:
      1. The project workspace
      2. The explicitly designated external results roots (Pilot & Research)
      3. Standard system temporary directories (when allow_temp=True, for test isolation)
    Rejects paths that escape permitted roots or traverse outside approved boundaries.
    """
    p = Path(log_dir)
    resolved = p.resolve()

    project_root = Path(__file__).resolve().parents[3].resolve()

    in_project = False
    try:
        resolved.relative_to(project_root)
        in_project = True
    except ValueError:
        pass

    in_approved_external = False
    for approved_root in APPROVED_EXTERNAL_RESULTS_ROOTS:
        try:
            resolved.relative_to(approved_root)
            in_approved_external = True
            break
        except ValueError:
            pass

    in_temp = False
    if allow_temp:
        try:
            resolved.relative_to(Path(tempfile.gettempdir()).resolve())
            in_temp = True
        except ValueError:
            pass

    if not in_project and not in_approved_external and not in_temp:
        raise ValueError(
            f"Path traversal or unauthorized log directory detected: '{log_dir}' "
            f"resolves outside permitted workspace, approved pilot results root, or temp directories: {resolved}"
        )

    return resolved


# Safe run_id validation pattern
_RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

# Sensitive key names to redact
_SENSITIVE_KEY_NAMES = {
    "api_key",
    "apikey",
    "auth",
    "secret",
    "token",
    "password",
    "gemini_api_key",
    "authorization",
    "credentials",
    "bearer",
}

# Regex to detect API keys or bearer tokens in text
_API_KEY_REGEX = re.compile(r"AIza[0-9A-Za-z\-_]{20,}", re.IGNORECASE)
_BEARER_REGEX = re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE)


def sanitize_log_payload(obj: Any, depth: int = 0) -> Any:
    """
    Recursively redact API keys, credentials, and complete sensitive dataset rows.
    """
    if depth > 10:
        return "[MAX_DEPTH_TRUNCATED]"

    if isinstance(obj, str):
        # Mask credentials
        s = _API_KEY_REGEX.sub("[REDACTED_API_KEY]", obj)
        s = _BEARER_REGEX.sub("Bearer [REDACTED_TOKEN]", s)
        return s

    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            k_lower = str(k).lower().strip()
            # Redact known credential field names
            if any(sens in k_lower for sens in _SENSITIVE_KEY_NAMES):
                sanitized[k] = "[REDACTED]"
            # Prevent logging raw dataset rows or complete tables
            elif k_lower in {"raw_rows", "dataset_rows", "records", "data_dump", "full_table"}:
                if isinstance(v, list):
                    sanitized[k] = f"[PROTECTED_DATASET_ROWS: count={len(v)}]"
                else:
                    sanitized[k] = "[PROTECTED_DATASET_CONTENT]"
            elif k_lower == "row" and isinstance(v, (dict, list)):
                sanitized[k] = "[PROTECTED_DATASET_ROW]"
            else:
                sanitized[k] = sanitize_log_payload(v, depth + 1)
        return sanitized

    if isinstance(obj, list):
        # If list contains many items, cap for logging brevity and privacy
        if len(obj) > 50 and any(isinstance(x, (dict, list)) for x in obj[:5]):
            truncated = [sanitize_log_payload(x, depth + 1) for x in obj[:50]]
            truncated.append(f"... [TRUNCATED: {len(obj) - 50} additional items]")
            return truncated
        return [sanitize_log_payload(x, depth + 1) for x in obj]

    return obj


class ExperimentLogger:
    """
    Structured, fail-safe logger for DataInvest AI runs and experiments.
    Writes newline-delimited JSON (JSONL) records to backend/data/logs/{run_id}.jsonl.
    """

    def __init__(self, logs_dir: Optional[Union[Path, str]] = None, allow_temp: bool = True):
        if logs_dir is None:
            # Default to backend/data/logs
            self.logs_dir = Path(__file__).resolve().parent.parent.parent / "data" / "logs"
        else:
            # Explicitly requested log directory: validate containment immediately.
            # Never silently fall back to default historical logs if invalid.
            self.logs_dir = validate_safe_log_dir(logs_dir, allow_temp=allow_temp)

    def _ensure_logs_dir(self) -> None:
        """Create logs directory if it does not exist."""
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Failed to create experiment logs directory %s: %s", self.logs_dir, exc)

    def log_event(
        self,
        run_id: str,
        event_type: str,
        phase: Optional[int] = None,
        outcome: str = "INFO",
        reason_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
    ) -> bool:
        """
        Record a structured audit log event.
        Guaranteed never to raise an unhandled exception or corrupt experiment results.
        """
        if not run_id or not isinstance(run_id, str):
            logger.warning("ExperimentLogger called with invalid run_id: %r", run_id)
            return False

        clean_run_id = run_id.strip()
        if not _RUN_ID_PATTERN.match(clean_run_id):
            logger.warning("ExperimentLogger rejected malformed run_id: %r", clean_run_id)
            return False

        try:
            self._ensure_logs_dir()
            log_path = self.logs_dir / f"{clean_run_id}.jsonl"

            # Prepare event record
            now_iso = datetime.now(timezone.utc).isoformat()
            raw_record = {
                "timestamp": now_iso,
                "run_id": clean_run_id,
                "experiment_id": experiment_id,
                "phase": phase,
                "event_type": str(event_type).upper().strip(),
                "outcome": str(outcome).upper().strip(),
                "reason_code": reason_code,
                "details": details or {},
            }

            # Sanitize for security
            sanitized_record = sanitize_log_payload(raw_record)

            line = json.dumps(sanitized_record, ensure_ascii=False) + "\n"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)

            return True
        except Exception as exc:
            # Logging failure must never fail the calling workflow
            logger.warning(
                "ExperimentLogger failed to write event '%s' for run '%s': %s",
                event_type,
                clean_run_id,
                exc,
            )
            return False

    def get_run_logs(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all structured events for a given run ID.
        Returns empty list if the run has no log file or if run_id is invalid.
        """
        if not run_id or not isinstance(run_id, str):
            return []

        clean_run_id = run_id.strip()
        if not _RUN_ID_PATTERN.match(clean_run_id):
            return []

        log_path = self.logs_dir / f"{clean_run_id}.jsonl"
        if not log_path.exists():
            return []

        events: List[Dict[str, Any]] = []
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:
            logger.warning("Failed to read log file for run '%s': %s", clean_run_id, exc)
            return []

        return events


# Singleton instance
experiment_logger = ExperimentLogger()
