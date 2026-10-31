"""
backend/app/services/run_store.py
==================================
File-based persistence service for DataInvest AI experiment runs.
Stores and retrieves complete pipeline executions under backend/data/runs/.
Guarantees unique IDs per experiment run so repeated uploads of the same dataset
are preserved independently without overwriting previous runs.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.schemas.run_schema import RunSummary, RunPayload


def _get_runs_dir() -> Path:
    """Resolve and ensure the backend/data/runs directory exists."""
    # Anchored relative to this file: backend/app/services -> backend/data/runs
    base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "runs"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _sanitize_slug(name: str) -> str:
    """Convert filename to safe, readable alphanumeric slug."""
    clean = re.sub(r"[^\w\s-]", "", name).strip().lower()
    clean = re.sub(r"[-\s]+", "_", clean)
    return clean[:30] if clean else "dataset"


def _generate_unique_run_id(dataset_name: str) -> str:
    """Generate a collision-proof, timestamped run ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = _sanitize_slug(dataset_name)
    suffix = uuid.uuid4().hex[:6]
    return f"run_{ts}_{slug}_{suffix}"


def _sanitize_run_id(run_id: str) -> Optional[str]:
    """Validate run_id to prevent directory traversal or invalid characters."""
    if not run_id or not isinstance(run_id, str):
        return None
    raw = run_id.strip()
    if "/" in raw or "\\" in raw or ".." in raw:
        return None
    if not re.match(r"^run_[a-zA-Z0-9_-]+$", raw):
        return None
    return raw



class RunStoreService:
    """Manages on-disk persistence of complete experiment pipeline runs."""

    def __init__(self, runs_dir: Optional[Path] = None):
        self._runs_dir = runs_dir or _get_runs_dir()

    @property
    def runs_dir(self) -> Path:
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        return self._runs_dir

    def _extract_summary(self, payload: Dict[str, Any]) -> RunSummary:
        """Extract lightweight metadata summary from complete run payload."""
        run_id = payload.get("run_id", "unknown")
        dataset_name = payload.get("dataset_name") or "untitled_dataset.csv"
        dataset_id = payload.get("dataset_id") or dataset_name
        created_at = payload.get("created_at") or datetime.now(timezone.utc).isoformat()
        updated_at = payload.get("updated_at") or created_at

        # Extract dimensions from profile overview if present
        profile = payload.get("profile") or {}
        overview = profile.get("overview") or {}
        n_rows = overview.get("n_rows")
        n_cols = overview.get("n_cols")

        # Determine completed phases from actual payload contents
        completed = list(payload.get("completed_phases") or [])
        if profile and 1 not in completed:
            completed.append(1)
        if payload.get("plan") and 2 not in completed:
            completed.append(2)
        if payload.get("analysis") and 3 not in completed:
            completed.append(3)
        if payload.get("critic") and 4 not in completed:
            completed.append(4)
        if payload.get("verification") and 5 not in completed:
            completed.append(5)
        if payload.get("insights") and 6 not in completed:
            completed.append(6)
        if payload.get("evaluation") and 7 not in completed:
            completed.append(7)
        completed = sorted(list(set(completed)))

        # Extract evaluation score if available
        evaluation = payload.get("evaluation") or {}
        summary_obj = evaluation.get("summary") or {}
        overall_score = summary_obj.get("overall_score")
        overall_grade = summary_obj.get("overall_grade")

        active_phase = payload.get("active_phase") or (max(completed) if completed else 1)
        mode = payload.get("mode")
        paired_run_id = payload.get("paired_run_id")

        # Determine visual evidence presence across analysis and verification
        has_viz = False
        analysis = payload.get("analysis") or {}
        for r in (analysis.get("results") or []):
            if isinstance(r, dict) and r.get("visual_evidence"):
                has_viz = True
                break
        if not has_viz:
            verification = payload.get("verification") or {}
            for vf in (verification.get("verified_findings") or verification.get("all_findings") or []):
                if isinstance(vf, dict) and vf.get("visual_evidence"):
                    has_viz = True
                    break
            if not has_viz:
                for rec in (verification.get("reinvestigations") or []):
                    if isinstance(rec, dict) and (rec.get("original_visual_evidence") or rec.get("reinvestigated_visual_evidence")):
                        has_viz = True
                        break

        # Extract planner metadata if plan is present
        plan_data = payload.get("plan") or {}
        planner_mode = plan_data.get("planner_mode")
        planner_model = plan_data.get("planner_model") or (plan_data.get("dataset_summary") or {}).get("model")
        planner_provider = plan_data.get("planner_provider")
        if not planner_provider and planner_mode:
            planner_provider = "google" if planner_mode == "llm" else ("deterministic" if planner_mode == "deterministic" else None)

        return RunSummary(
            run_id=run_id,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            created_at=created_at,
            updated_at=updated_at,
            n_rows=n_rows,
            n_cols=n_cols,
            completed_phases=completed,
            active_phase=active_phase,
            overall_score=overall_score,
            overall_grade=overall_grade,
            mode=mode,
            paired_run_id=paired_run_id,
            has_visual_evidence=has_viz,
            planner_mode=planner_mode,
            planner_model=planner_model,
            planner_provider=planner_provider,
        )

    def save_run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save or update an experiment run atomically.
        If run_id is missing or invalid, a new unique run_id is generated.
        Returns dict with status, run_id, updated_at, and summary.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        raw_run_id = data.get("run_id")
        run_id = _sanitize_run_id(raw_run_id) if raw_run_id else None

        dataset_name = data.get("dataset_name") or "dataset.csv"
        if not run_id:
            run_id = _generate_unique_run_id(dataset_name)
            data["run_id"] = run_id
            data["created_at"] = now_iso

        if not data.get("created_at"):
            data["created_at"] = now_iso

        data["updated_at"] = now_iso

        # Build clean summary
        summary = self._extract_summary(data)
        data["completed_phases"] = summary.completed_phases
        data["active_phase"] = summary.active_phase
        if summary.has_visual_evidence:
            data["has_visual_evidence"] = True

        file_path = self.runs_dir / f"{run_id}.json"
        temp_path = self.runs_dir / f"{run_id}.tmp"

        def _json_serial(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "item"):
                return obj.item()
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            return str(obj)

        try:
            # Atomic write via temporary file with resilient serialization
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=_json_serial)
            temp_path.replace(file_path)
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise IOError(f"Failed to save experiment run {run_id}: {str(e)}") from e

        return {
            "status": "saved",
            "run_id": run_id,
            "updated_at": now_iso,
            "summary": summary.model_dump(),
        }

    def list_runs(self) -> List[RunSummary]:
        """List all saved experiment runs sorted newest first."""
        summaries: List[RunSummary] = []
        if not self.runs_dir.exists():
            return summaries

        for entry in self.runs_dir.glob("run_*.json"):
            try:
                with open(entry, "r", encoding="utf-8") as f:
                    content = json.load(f)
                summaries.append(self._extract_summary(content))
            except Exception:
                # Corrupted or unreadable run files are skipped cleanly
                continue

        # Sort descending by updated_at or created_at
        summaries.sort(key=lambda s: s.updated_at or s.created_at, reverse=True)
        return summaries

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full experiment run payload by run_id."""
        clean_id = _sanitize_run_id(run_id)
        if not clean_id:
            return None

        file_path = self.runs_dir / f"{clean_id}.json"
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise IOError(f"Error reading run file {clean_id}: {str(e)}") from e

    def delete_run(self, run_id: str) -> bool:
        """Delete an experiment run file by run_id."""
        clean_id = _sanitize_run_id(run_id)
        if not clean_id:
            return False

        file_path = self.runs_dir / f"{clean_id}.json"
        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            return True
        except Exception:
            return False


# Global singleton instance
run_store = RunStoreService()
