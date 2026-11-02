"""
backend/app/main.py
====================
DataInvest AI — FastAPI Application Entry Point

Endpoints:
    GET  /                    → Root health / version metadata
    GET  /health              → Operational health check
    POST /api/profile         → Secure CSV dataset upload & statistical profiling (Phase 1)
    POST /api/plan            → Structured analysis task planning (Phase 2 Planner Agent)
    POST /api/analyze         → Execute single statistical investigation task (Phase 3 Analysis Agent)
    POST /api/analyze-plan    → Execute complete planned investigation tasks with failure isolation (Phase 3)
    POST /api/critic          → Verify Phase 3 empirical findings and assess validity (Phase 4 Critic Agent)
    POST /api/verify-findings → Controlled re-investigation & verified findings synthesis (Phase 5)
    POST /api/reinvestigate   → Alias endpoint for Phase 5 re-investigation workflow
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from collections import defaultdict
import traceback

# Ensure backend directory is in sys.path so 'app' always resolves to backend/app regardless of cwd
import sys
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from dotenv import load_dotenv

# Explicitly load environment variables from backend/.env regardless of current working directory
_BACKEND_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_BACKEND_ENV_PATH)

from fastapi import FastAPI, File, UploadFile, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse

from app.modules import profiler
from app.services import upload_service
from app.services import planner_service
from app.services import dataset_store
from app.services.analysis_service import get_analysis_service
from app.services.critic_service import get_critic_service
from app.services.verification_service import get_verification_service
from app.services.insight_service import get_insight_service
from app.services.evaluation_service import evaluation_service
from app.services.run_store import run_store
from app.services.report_service import report_service, RunNotFoundError, InvalidRunIdError
from app.services.experiment_logger import experiment_logger
from app.services.demo_service import demo_service, DEMO_DISCLAIMER
from app.agents.report_agent import report_agent
from app.schemas.report_schema import ResearchReport
from app.schemas.profile import (
    ProfileResponse,
    ErrorResponse,
    HealthResponse,
    RootResponse,
)
from app.schemas.planner_schema import PlannerRequest, PlannerResponse
from app.schemas.analysis_schema import (
    SingleAnalysisRequest,
    PlanAnalysisRequest,
    AnalysisResult,
    PlanAnalysisResponse,
)
from app.schemas.critic_schema import CriticRequest, CriticResponse
from app.schemas.verification_schema import (
    VerificationRequest,
    VerificationResponse,
)
from app.schemas.insight_schema import (
    InsightRequest,
    InsightResponse,
)
from app.schemas.evaluation_schema import (
    EvaluationRequest,
    EvaluationResponse,
)
from app.schemas.run_schema import (
    RunSummary,
    RunPayload,
    RunListResponse,
    RunSaveResponse,
)
from app.schemas.experiment_schema import (
    PairedExperimentRequest,
    PairedExperimentResponse,
)
from app.services.experiment_runner import (
    ExperimentPersistenceError,
    ExperimentRunnerError,
    run_paired_experiment,
    _sanitize_error_message,
)
from app.services.gemini_client import (
    GeminiAuthenticationError,
    GeminiClientError,
    GeminiMalformedResponseError,
    GeminiMissingKeyError,
    GeminiNetworkError,
    GeminiRateLimitError,
    GeminiTimeoutError,
)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI Application Configuration
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DataInvest AI",
    description="Automated Insight Discovery — Phase 1 Profiler, Phase 2 Planner, Phase 3 Analysis, Phase 4 Critic, Phase 5 Verification, Phase 6 Insight Generation, Phase 7 System Evaluation",
    version="7.0.0",
)


def get_cors_origins() -> list[str]:
    """
    Build allowed CORS origins combining local development defaults with CORS_ORIGINS env var.
    Explicitly ignores wildcard '*' to protect sensitive research endpoints per security requirements.
    """
    default_origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "https://data-invest-ai.vercel.app",
    ]
    env_origins = os.environ.get("CORS_ORIGINS", "")
    origins = list(default_origins)
    if env_origins:
        for raw in env_origins.split(","):
            cleaned = raw.strip().rstrip("/")
            if cleaned and cleaned != "*" and cleaned not in origins:
                origins.append(cleaned)
    return origins


def is_production_environment() -> bool:
    """Return True if running in a designated production or cloud environment."""
    return (
        os.environ.get("DATAINVEST_PRODUCTION", "").strip().lower() in ("true", "1", "yes")
        or os.environ.get("ENVIRONMENT", "").strip().lower() == "production"
        or os.environ.get("RENDER", "").strip().lower() == "true"
    )


_rate_limits: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(client_ip: str, max_requests: int = 20, window_seconds: int = 60) -> bool:
    """
    Lightweight sliding-window in-memory rate limiter for public abuse prevention.
    Returns True if allowed, False if exceeded.
    """
    now = time.time()
    timestamps = _rate_limits[client_ip]
    cutoff = now - window_seconds
    valid_timestamps = [t for t in timestamps if t > cutoff]
    if len(valid_timestamps) >= max_requests:
        _rate_limits[client_ip] = valid_timestamps
        return False
    valid_timestamps.append(now)
    _rate_limits[client_ip] = valid_timestamps
    return True


def get_benchmark_results_root() -> Path:
    """
    Resolve benchmark results directory using repository relative path,
    falling back to sibling directory in local development structure if present.
    """
    repo_results = Path(__file__).resolve().parent.parent.parent / "DataInvest_Research_Results"
    if (repo_results / "consolidated_5_datasets_20261028").exists():
        return repo_results
    sibling_results = Path(__file__).resolve().parent.parent.parent.parent / "DataInvest_Research_Results"
    if (sibling_results / "consolidated_5_datasets_20261028").exists():
        return sibling_results
    return repo_results


# Robust CORS policy for React/Vite development and production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Health Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/", response_model=RootResponse, tags=["Health"])
async def root():
    """Root metadata endpoint verifying service version and phase."""
    return {"status": "ok", "app": "DataInvest AI", "phase": 7}



@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Standard health check endpoint."""
    return {"status": "healthy"}


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Profiler Endpoint
# ──────────────────────────────────────────────────────────────────────────────


@app.post(
    "/api/profile",
    response_model=ProfileResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation or parsing error in CSV"},
        500: {"model": ErrorResponse, "description": "Internal processing failure"},
    },
    tags=["Profiler"],
)
async def profile_dataset(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Query(None, description="Optional worksheet name for multi-sheet Excel workbooks"),
):
    """
    Accept an uploaded CSV file, run the deterministic Pandas/NumPy profiler,
    store the dataset in the in-memory cache, and return structured profiling data.

    Returns:
    - 422: Malformed, empty, non-CSV, or un-parsable data.
    - 500: Internal error with sanitized user message (no stack traces exposed).
    """

    # 1. Validate file extension
    filename = file.filename or "uploaded_dataset.csv"
    allowed_exts = (".csv", ".txt", ".xlsx")
    if not any(filename.lower().endswith(ext) for ext in allowed_exts):
        return JSONResponse(
            status_code=422,
            content={"error": "Only CSV and TXT tabular files are supported (XLSX also supported). Please upload a .csv, .txt, or .xlsx file."},
        )

    # 2. Read file content safely
    try:
        content = await file.read()
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to read the uploaded file from request."},
        )

    # 3. Parse CSV into DataFrame with multi-encoding support
    df, parse_error = upload_service.parse_csv(content, filename=filename, sheet_name=sheet_name)
    if parse_error:
        return JSONResponse(status_code=422, content={"error": parse_error})

    # 4. Generate statistical profile and store DataFrame
    try:
        # Cache dataset for downstream analysis & critic verification
        dataset_id = dataset_store.store_dataset(filename, df)

        result = profiler.profile_dataset(df)
        result["filename"] = filename
        result["dataset_id"] = dataset_id
        result["preview_rows"] = profiler.prepare_preview_rows(df, n=100)

        return JSONResponse(content=result)
    except Exception:
        # Internal log for server observability; sanitize output for the client
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Dataset profiling failed. Please check the dataset format and try again."},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Planner Agent Endpoint
# ──────────────────────────────────────────────────────────────────────────────


@app.post(
    "/api/plan",
    response_model=PlannerResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid profiler data, objective, or schema validation failure"},
        429: {"model": ErrorResponse, "description": "Gemini API rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Planning service failure"},
        502: {"model": ErrorResponse, "description": "Upstream Gemini service error"},
        503: {"model": ErrorResponse, "description": "Gemini API credentials or service unavailable"},
        504: {"model": ErrorResponse, "description": "Gemini API request timeout"},
    },
    tags=["Planner"],
)
async def generate_analysis_plan(request: PlannerRequest, http_request: Request):
    """
    Accept a Phase 1 dataset profile and natural-language user objective,
    and generate a structured, prioritized investigation plan.

    The Planner Agent decides WHAT analysis should be performed and WHY,
    without executing statistical tests or fabricating results.
    """
    # 0. Check production abuse protection & rate limit
    if is_production_environment():
        client_ip = http_request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (http_request.client.host if http_request.client else "unknown")
        if not check_rate_limit(client_ip, max_requests=20, window_seconds=60):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded for analysis planning. Please wait a minute before making another request."},
            )

    # 1. Validate profile existence
    if not request.profile or not isinstance(request.profile, dict):
        return JSONResponse(
            status_code=422,
            content={"error": "Missing dataset profile. Please provide a valid Phase 1 profile dictionary."},
        )

    # 2. Validate objective length (prevent abuse)
    objective = (request.objective or "").strip()
    if len(objective) > 1000:
        return JSONResponse(
            status_code=422,
            content={"error": "Investigation objective is too long. Please keep it under 1000 characters."},
        )

    if not objective:
        objective = "Investigate this dataset."

    # 3. Execute Planner Service
    try:
        req_use_llm = request.use_llm
        if req_use_llm is None and request.preferences and isinstance(request.preferences, dict):
            if "use_llm" in request.preferences:
                req_use_llm = bool(request.preferences["use_llm"])
        if req_use_llm is None:
            req_use_llm = True

        plan_response = planner_service.create_analysis_plan(
            profile=request.profile,
            objective=objective,
            dataset_id=request.dataset_id,
            preferences=request.preferences,
            use_llm=req_use_llm,
        )
        return plan_response
    except GeminiMissingKeyError:
        return JSONResponse(
            status_code=503,
            content={"error": "Gemini API credentials not configured. Please set the GEMINI_API_KEY environment variable."},
        )
    except GeminiAuthenticationError:
        return JSONResponse(
            status_code=502,
            content={"error": "Gemini API authentication failed. The configured API key is invalid or unauthorized."},
        )
    except GeminiRateLimitError:
        return JSONResponse(
            status_code=429,
            content={"error": "Gemini API rate limit exceeded. Please wait and try again later."},
        )
    except GeminiTimeoutError:
        return JSONResponse(
            status_code=504,
            content={"error": "Gemini API request timed out during planning. Please try again later."},
        )
    except (GeminiNetworkError, GeminiMalformedResponseError):
        return JSONResponse(
            status_code=502,
            content={"error": "Upstream error communicating with Gemini API service."},
        )
    except GeminiClientError:
        return JSONResponse(
            status_code=502,
            content={"error": "Upstream Gemini client error encountered during planning."},
        )
    except ValueError as val_err:
        return JSONResponse(status_code=422, content={"error": str(val_err)})
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Planning service encountered an internal error. Please check inputs and retry."},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Analysis Agent Endpoints (Statistical Execution Engine)
# ──────────────────────────────────────────────────────────────────────────────


@app.post(
    "/api/analyze",
    response_model=AnalysisResult,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid analysis request payload"},
        500: {"model": ErrorResponse, "description": "Analysis execution failure"},
    },
    tags=["Analysis Agent"],
)
async def analyze_single_task(request: SingleAnalysisRequest):
    """
    Execute a single planned investigation task using pure Python/SciPy.
    Returns factual, mathematically verified statistical results.
    """
    if not request.task:
        return JSONResponse(
            status_code=422,
            content={"error": "Missing investigation task in request."},
        )

    try:
        service = get_analysis_service()
        result = service.execute_single_task(request)
        return result
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Statistical analysis execution encountered an internal error."},
        )


@app.post(
    "/api/analyze-plan",
    response_model=PlanAnalysisResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid plan analysis payload"},
        500: {"model": ErrorResponse, "description": "Plan execution failure"},
    },
    tags=["Analysis Agent"],
)
async def analyze_plan(request: PlanAnalysisRequest):
    """
    Execute an entire list of planned investigation tasks with failure isolation.
    Each task is computed using SciPy tools. Any single task failure will not
    affect the execution of subsequent tasks in the plan.
    """
    if not request.tasks:
        return JSONResponse(
            status_code=422,
            content={"error": "No investigation tasks provided in the analysis plan."},
        )

    try:
        service = get_analysis_service()
        plan_response = service.execute_plan(request)
        return plan_response
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Plan analysis execution encountered an internal error."},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: Critic Agent Endpoints (Statistical Verification Layer)
# ──────────────────────────────────────────────────────────────────────────────


@app.post(
    "/api/critic",
    response_model=CriticResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid Critic review payload"},
        500: {"model": ErrorResponse, "description": "Critic review execution failure"},
    },
    tags=["Critic Agent"],
)
async def run_critic_review(request: CriticRequest):
    """
    Evaluate Phase 3 statistical results to verify methodological validity,
    sample size adequacy, outlier sensitivity, assumption stability, and effect size.
    Returns structured verification reviews classified into:
    ACCEPTED, QUESTIONABLE, REJECTED, or REQUIRES_REINVESTIGATION.
    """
    if not request.analysis_response and not request.results:
        return JSONResponse(
            status_code=422,
            content={"error": "Missing analysis results in Critic review request. Please provide analysis_response or results."},
        )

    try:
        service = get_critic_service()
        response = service.run_review(request)
        return response
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Critic verification review encountered an internal error."},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5: Re-investigation & Verified Findings Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@app.post(
    "/api/verify-findings",
    response_model=VerificationResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid verification payload"},
        500: {"model": ErrorResponse, "description": "Verification execution failure"},
    },
    tags=["Verified Findings"],
)
async def verify_findings(request: VerificationRequest):
    """
    Execute controlled re-investigation on eligible findings using the existing
    Analysis Agent and Critic Agent, compare before/after statistical results,
    and synthesize final structured Verified Findings.
    """
    if not request.analysis_response and not request.critic_response:
        return JSONResponse(
            status_code=422,
            content={"error": "Missing analysis or critic results. Please provide analysis_response or critic_response."},
        )

    try:
        service = get_verification_service()
        response = service.process_verification(request)
        return response
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Verification and re-investigation orchestration encountered an internal error."},
        )


@app.post(
    "/api/reinvestigate",
    response_model=VerificationResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid re-investigation payload"},
        500: {"model": ErrorResponse, "description": "Re-investigation execution failure"},
    },
    tags=["Verified Findings"],
)
async def reinvestigate_findings(request: VerificationRequest):
    """Alias endpoint for Phase 5 re-investigation and finding verification."""
    return await verify_findings(request)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 6: Insight Generation Endpoint
# ──────────────────────────────────────────────────────────────────────────────


@app.post(
    "/api/insights",
    response_model=InsightResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid insight generation payload"},
        500: {"model": ErrorResponse, "description": "Insight generation failure"},
    },
    tags=["Insight Generation"],
)
async def generate_insights(request: InsightRequest):
    """
    Convert Phase 5 verified statistical findings into clear, human-readable,
    non-causal analytical insights with full evidence traceability, transparent
    strength classification, and failure isolation.
    """
    if not request.verification_response and not request.verified_findings:
        return JSONResponse(
            status_code=422,
            content={"error": "Missing Phase 5 verification findings. Please provide verification_response or verified_findings."},
        )

    try:
        service = get_insight_service()
        response = service.generate_insights(request)
        return response
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Insight generation encountered an internal processing error."},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 7: System Evaluation & Benchmark Scorecard Endpoint
# ──────────────────────────────────────────────────────────────────────────────


@app.post(
    "/api/evaluate",
    response_model=EvaluationResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid evaluation payload"},
        500: {"model": ErrorResponse, "description": "Evaluation service failure"},
    },
    tags=["System Evaluation"],
)
async def evaluate_system(request: EvaluationRequest):
    """
    Evaluates the complete analytical pipeline across 4 empirical dimensions:
    Execution Reliability, Statistical Rigor, Verification Truth Grounding, and
    Insight Actionability.
    """
    if not request.profile:
        return JSONResponse(
            status_code=422,
            content={"error": "Missing dataset profile in evaluation request."},
        )

    try:
        response = evaluation_service.evaluate_pipeline(
            profile=request.profile,
            plan=request.plan,
            analysis=request.analysis,
            critic=request.critic,
            verification=request.verification,
            insights=request.insights,
        )
        return response
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "System evaluation encountered an internal processing error."},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Experiment Run Repository Endpoints (Persistence & History)
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/api/runs", response_model=RunListResponse, tags=["Experiment Runs"])
async def list_experiment_runs():
    """List all saved experiment runs sorted newest first."""
    try:
        runs = run_store.list_runs()
        return {"total_runs": len(runs), "runs": runs}
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve experiment runs."},
        )


@app.post("/api/runs", response_model=RunSaveResponse, tags=["Experiment Runs"])
async def save_experiment_run(payload: RunPayload):
    """Save or update an experiment run with unique ID and complete pipeline state."""
    try:
        result = run_store.save_run(payload.model_dump())
        saved_run_id = result.get("run_id") or payload.run_id
        try:
            experiment_logger.log_event(
                run_id=saved_run_id,
                event_type="RUN_SAVED",
                outcome="SUCCESS",
                details={
                    "mode": getattr(payload, "mode", None),
                    "active_phase": getattr(payload, "active_phase", None),
                    "dataset_name": getattr(payload, "dataset_name", None),
                },
            )
        except Exception:
            pass
        return result
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to persist experiment run: {str(e)}"},
        )


@app.get("/api/runs/{run_id}", tags=["Experiment Runs"])
async def get_experiment_run(run_id: str):
    """Retrieve complete saved experiment run payload for restoration without re-running."""
    try:
        run_data = run_store.get_run(run_id)
        if not run_data:
            return JSONResponse(
                status_code=404,
                content={"error": f"Experiment run '{run_id}' not found."},
            )
        return run_data
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to load experiment run '{run_id}'."},
        )


@app.delete("/api/runs/{run_id}", tags=["Experiment Runs"])
async def delete_experiment_run(run_id: str, request: Request):
    """
    Delete an experiment run from persistent storage.
    In production environments, this destructive operation is strictly forbidden (HTTP 403)
    unless authenticated with a matching ADMIN_API_TOKEN header ('x-admin-token' or Bearer token).
    """
    if is_production_environment():
        admin_token = os.environ.get("ADMIN_API_TOKEN", "").strip()
        auth_header = request.headers.get("x-admin-token") or request.headers.get("authorization", "")
        token = auth_header.replace("Bearer ", "").strip() if auth_header else ""
        if not admin_token or token != admin_token:
            return JSONResponse(
                status_code=403,
                content={"error": "Run deletion is disabled in production environment."},
            )

    try:
        deleted = run_store.delete_run(run_id)
        if not deleted:
            return JSONResponse(
                status_code=404,
                content={"error": f"Experiment run '{run_id}' not found or could not be deleted."},
            )
        return {"status": "deleted", "run_id": run_id}
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to delete experiment run '{run_id}'."},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Report Agent & Export Endpoints (Phase 5 / Stage B)
# ──────────────────────────────────────────────────────────────────────────────


@app.get(
    "/api/runs/{run_id}/report",
    response_model=ResearchReport,
    responses={
        404: {"model": ErrorResponse, "description": "Experiment run not found"},
        422: {"model": ErrorResponse, "description": "Invalid run ID format"},
        500: {"model": ErrorResponse, "description": "Report synthesis failure"},
    },
    tags=["Experiment Runs", "Report & Export"],
)
async def get_experiment_run_report_endpoint(run_id: str):
    """
    Retrieve a complete structured scientific report synthesized from a saved run.
    Guarantees that report data corresponds strictly to the requested run_id.
    """
    try:
        report = report_service.get_report_for_run(run_id)
        return report
    except RunNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": f"Experiment run '{run_id}' not found."},
        )
    except InvalidRunIdError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": str(exc)},
        )
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to synthesize research report for run '{run_id}'."},
        )


@app.get(
    "/api/runs/{run_id}/export/markdown",
    responses={
        200: {"description": "Formatted GitHub-Flavored Markdown report file download"},
        404: {"model": ErrorResponse, "description": "Experiment run not found"},
        422: {"model": ErrorResponse, "description": "Invalid run ID format"},
        500: {"model": ErrorResponse, "description": "Export generation failure"},
    },
    tags=["Experiment Runs", "Report & Export"],
)
async def export_run_markdown_endpoint(run_id: str):
    """
    Export synthesized research report as a readable, downloadable GitHub-Flavored Markdown file.
    """
    try:
        md_text, filename = report_service.export_markdown(run_id)
        return Response(
            content=md_text,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except RunNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": f"Experiment run '{run_id}' not found."},
        )
    except InvalidRunIdError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": str(exc)},
        )
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to export Markdown report for run '{run_id}'."},
        )


@app.get(
    "/api/runs/{run_id}/export/json",
    responses={
        200: {"description": "Structured JSON report and run state file download"},
        404: {"model": ErrorResponse, "description": "Experiment run not found"},
        422: {"model": ErrorResponse, "description": "Invalid run ID format"},
        500: {"model": ErrorResponse, "description": "Export generation failure"},
    },
    tags=["Experiment Runs", "Report & Export"],
)
async def export_run_json_endpoint(run_id: str):
    """
    Export complete experiment run and synthesized report as a structured JSON download.
    """
    try:
        export_dict, filename = report_service.export_json(run_id)
        json_content = json.dumps(export_dict, indent=2, ensure_ascii=False)
        return Response(
            content=json_content,
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except RunNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": f"Experiment run '{run_id}' not found."},
        )
    except InvalidRunIdError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": str(exc)},
        )
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to export JSON for run '{run_id}'."},
        )


@app.post(
    "/api/report/generate",
    response_model=ResearchReport,
    responses={
        500: {"model": ErrorResponse, "description": "Report synthesis failure"},
    },
    tags=["Report & Export"],
)
async def generate_report_from_payload_endpoint(payload: RunPayload):
    """
    Synthesize a structured research report directly from an in-memory or custom run payload.
    """
    try:
        report = report_agent.generate_report(payload)
        return report
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to synthesize report from provided payload."},
        )


@app.get(
    "/api/runs/{run_id}/logs",
    responses={
        200: {"description": "Structured execution event logs for the specified run"},
        422: {"model": ErrorResponse, "description": "Invalid run ID format"},
    },
    tags=["Experiment Runs", "Audit & Logging"],
)
async def get_run_logs_endpoint(run_id: str):
    """
    Retrieve structured audit event logs for a specific experiment run.
    Safely returns an empty list if no logs exist for this run without rewriting historical data.
    """
    clean_id = (run_id or "").strip()
    if "/" in clean_id or "\\" in clean_id or ".." in clean_id:
        return JSONResponse(status_code=422, content={"error": f"Invalid run ID format: '{run_id}'."})

    logs = experiment_logger.get_run_logs(clean_id)
    return {"run_id": clean_id, "total_events": len(logs), "events": logs}


@app.post(
    "/api/demo/run-offline",
    responses={
        200: {"description": "Complete offline deterministic demonstration response"},
        500: {"model": ErrorResponse, "description": "Offline demo execution failure"},
    },
    tags=["Offline Demo"],
)
async def run_offline_demo_endpoint():
    """
    Execute a 100% offline, reproducible demonstration using synthetic demo data.
    Strictly deterministic. Zero Gemini API calls. All outputs marked as simulated.
    """
    try:
        demo_result = demo_service.run_offline_demo()
        return demo_result
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Offline demonstration execution encountered an internal error."},
        )


@app.get(
    "/api/demo/status",
    tags=["Offline Demo"],
)
async def get_demo_status_endpoint():
    """
    Retrieve metadata regarding offline demo capability, synthetic data, and disclaimers.
    """
    return {
        "offline_demo_available": True,
        "mode": "offline_deterministic_demo",
        "dataset_id": demo_service.dataset_id,
        "disclaimer": DEMO_DISCLAIMER,
        "gemini_api_required": False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Paired Experiments Endpoint (Baseline vs Proposed Benchmark)
# ──────────────────────────────────────────────────────────────────────────────


@app.post(
    "/api/experiments/paired",
    response_model=PairedExperimentResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Dataset not found or unavailable"},
        422: {"model": ErrorResponse, "description": "Invalid experiment request parameters"},
        429: {"model": ErrorResponse, "description": "Gemini API rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Experiment execution or persistence failure"},
        502: {"model": ErrorResponse, "description": "Upstream Gemini service error"},
        503: {"model": ErrorResponse, "description": "Gemini configuration or credentials unavailable"},
    },
    tags=["Paired Experiments"],
)
async def run_paired_experiment_endpoint(request: PairedExperimentRequest, http_request: Request):
    """
    Execute a scientific paired experiment comparing Baseline vs Proposed architectures.

    Research Design:
    1. Validates the incoming request parameters.
    2. Resolves the requested dataset ID and verifies data availability.
    3. Obtains the authoritative dataset profile through the existing profiling pipeline.
    4. Invokes run_paired_experiment(...) exactly once.
    5. Returns the typed PairedExperimentResponse.
    6. Preserves existing research safeguards: one genuine LLM plan, no deterministic fallback,
       shared plan and raw analysis, honest comparison metrics, and safe persistence.
    """
    # 0. Check production abuse protection & rate limit
    if is_production_environment():
        client_ip = http_request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (http_request.client.host if http_request.client else "unknown")
        if not check_rate_limit(client_ip, max_requests=10, window_seconds=60):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded for paired experiments. Please wait a minute before making another request."},
            )

    # 1. Resolve dataset identifier
    dataset_id = (request.dataset_id or "").strip()
    if not dataset_id and request.profile:
        overview = request.profile.get("overview") or {}
        dataset_id = str(request.profile.get("dataset_id") or overview.get("filename") or "").strip()

    if not dataset_id:
        return JSONResponse(
            status_code=422,
            content={"error": "A dataset_id or named dataset profile is required to resolve dataset data."},
        )

    # 2. Resolve DataFrame from dataset_store and verify availability
    df = dataset_store.resolve_dataset(dataset_id)
    if df is None or df.empty:
        return JSONResponse(
            status_code=404,
            content={"error": f"Dataset '{dataset_id}' could not be resolved or is not available. Please upload the dataset first."},
        )

    # 3. Obtain authoritative profile through existing profiling pipeline
    # Do not trust client-supplied profile if application can obtain authoritative profile from stored dataset
    try:
        authoritative_profile = profiler.profile_dataset(df)
        authoritative_profile["filename"] = dataset_id
        authoritative_profile["dataset_id"] = dataset_id
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to compute authoritative profile for dataset '{dataset_id}'."},
        )

    # 4. Invoke run_paired_experiment exactly once with error mapping
    try:
        response = run_paired_experiment(
            profile=authoritative_profile,
            dataset_id=dataset_id,
            objective=request.objective,
            max_reinvestigations=request.max_reinvestigations,
            persist_runs=request.persist_runs,
        )
        return response
    except GeminiMissingKeyError:
        return JSONResponse(
            status_code=503,
            content={"error": "Gemini API credentials not configured. Please set the GEMINI_API_KEY environment variable."},
        )
    except GeminiAuthenticationError:
        return JSONResponse(
            status_code=502,
            content={"error": "Gemini API authentication failed. The configured API key is invalid or unauthorized."},
        )
    except GeminiRateLimitError:
        return JSONResponse(
            status_code=429,
            content={"error": "Gemini API rate limit exceeded. Please wait and try again later."},
        )
    except GeminiTimeoutError:
        return JSONResponse(
            status_code=504,
            content={"error": "Gemini API request timed out during planning. Please try again later."},
        )
    except (GeminiNetworkError, GeminiMalformedResponseError):
        return JSONResponse(
            status_code=502,
            content={"error": "Upstream error communicating with Gemini API service."},
        )
    except GeminiClientError:
        return JSONResponse(
            status_code=502,
            content={"error": "Upstream Gemini client error encountered during planning."},
        )
    except ExperimentPersistenceError as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to persist paired experiment run records.",
                "detail": _sanitize_error_message(str(exc)),
                "partial_persistence": exc.partial_persistence,
                "saved_run_id": exc.saved_run_id,
            },
        )
    except ExperimentRunnerError as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": _sanitize_error_message(str(exc))},
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": _sanitize_error_message(str(exc))},
        )
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Paired experiment execution encountered an internal error."},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Supervised Classification Benchmarks Endpoint (AutoMind Table 3 Reproducibility)
# ──────────────────────────────────────────────────────────────────────────────


@app.get(
    "/api/benchmarks/classification",
    tags=["Benchmarks"],
)
async def get_classification_benchmark_endpoint():
    """
    Retrieve genuine, experimentally measured supervised classification benchmark results
    for the Heart Disease datasets, comparing Baseline (Critic OFF) vs Proposed (Critic ON)
    alongside the published AutoMind Table 3 benchmark values (IEEE Access 2026).
    """
    # 1. Resolve results directories
    results_dir = Path(r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\DataInvest_Research_Results\classification")
    if not results_dir.exists():
        results_dir = Path(__file__).resolve().parent.parent.parent / "DataInvest_Research_Results" / "classification"

    tables_dir = results_dir / "tables"
    config_file = results_dir / "config.json"

    summary_json_path = tables_dir / "heart_classification_benchmark.json"
    automind_json_path = tables_dir / "automind_table3_comparison.json"

    summary_records = []
    if summary_json_path.exists():
        try:
            with open(summary_json_path, "r", encoding="utf-8") as f:
                summary_records = json.load(f)
        except Exception:
            traceback.print_exc()

    automind_records = []
    if automind_json_path.exists():
        try:
            with open(automind_json_path, "r", encoding="utf-8") as f:
                automind_records = json.load(f)
        except Exception:
            traceback.print_exc()

    config_data = {}
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception:
            traceback.print_exc()

    return {
        "status": "success",
        "benchmark_name": "Heart Disease Supervised Classification Benchmark",
        "summary_records": summary_records,
        "automind_comparison": automind_records,
        "config": config_data,
        "citation": {
            "title": "AutoMind: Automated Insight Discovery via Multi-Agent Navigation",
            "journal": "IEEE Access",
            "year": 2026,
            "volume": 14,
            "pages": "23936–23955",
            "doi": "10.1109/ACCESS.2026.3661202",
            "table": "Table 3",
            "reported_f1": "95.52%",
            "reported_model": "GPT-4o Multi-Agent Hub",
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# All 5 Datasets Consolidated Benchmark Endpoint (AutoMind Tables 1-6 Alignment)
# ──────────────────────────────────────────────────────────────────────────────


@app.get(
    "/api/benchmarks/all-datasets",
    tags=["Benchmarks"],
)
async def get_all_datasets_benchmark_endpoint():
    """
    Retrieve consolidated 5-dataset benchmark results aligning with AutoMind (IEEE Access 2026)
    Tables 1 to 6, including all paired Baseline (Critic OFF) vs Proposed (Critic ON) metrics,
    statistical significance tests (McNemar), and Critic agent verification audit trails.
    """
    import csv
    results_root = get_benchmark_results_root()

    cons_dir = results_root / "consolidated_5_datasets_20261028"

    consolidated_summary = []
    if (cons_dir / "CONSOLIDATED_FIVE_DATASET_BENCHMARK.json").exists():
        try:
            with open(cons_dir / "CONSOLIDATED_FIVE_DATASET_BENCHMARK.json", "r", encoding="utf-8") as f:
                consolidated_summary = json.load(f)
        except Exception:
            traceback.print_exc()

    manifest = {}
    if (cons_dir / "DATASET_AND_EXPERIMENT_MANIFEST.json").exists():
        try:
            with open(cons_dir / "DATASET_AND_EXPERIMENT_MANIFEST.json", "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            traceback.print_exc()

    automind_tables = []
    if (cons_dir / "AUTOMIND_ALIGNED_BENCHMARK_TABLES.csv").exists():
        try:
            with open(cons_dir / "AUTOMIND_ALIGNED_BENCHMARK_TABLES.csv", "r", encoding="utf-8") as f:
                automind_tables = list(csv.DictReader(f))
        except Exception:
            traceback.print_exc()

    critic_interventions = []
    if (cons_dir / "CRITIC_INTERVENTION_AUDIT_SUMMARY.csv").exists():
        try:
            with open(cons_dir / "CRITIC_INTERVENTION_AUDIT_SUMMARY.csv", "r", encoding="utf-8") as f:
                critic_interventions = list(csv.DictReader(f))
        except Exception:
            traceback.print_exc()

    # Per-dataset detailed records
    datasets_detail = {}

    # 1. Heart Disease
    heart_dir = results_root / "heart_experiment_candidate_20261028"
    if not heart_dir.exists():
        heart_dir = results_root / "classification"
    heart_run = heart_dir / "runs" / "heart_disease_918_row_logistic_regression.json"
    heart_tables = heart_dir / "tables" / "automind_table3_comparison.json"
    heart_data = {}
    if heart_run.exists():
        try:
            with open(heart_run, "r", encoding="utf-8") as f:
                heart_data = json.load(f)
        except Exception:
            pass
    heart_automind = []
    if heart_tables.exists():
        try:
            with open(heart_tables, "r", encoding="utf-8") as f:
                heart_automind = json.load(f)
        except Exception:
            pass
    datasets_detail["heart"] = {
        "id": "heart",
        "name": "Heart Disease Classification",
        "automind_table": "Table 3",
        "run_metrics": heart_data,
        "automind_comparison": heart_automind,
        "figures": ["figure7_heart_classification_comparison.png", "figure8_automind_table3_benchmark.png"],
    }

    # 2. Fake News
    fake_dir = results_root / "fake_news_experiment_20261028"
    fake_metrics = {}
    fake_profile = {}
    fake_critic = {}
    if (fake_dir / "classification_metrics.json").exists():
        try:
            with open(fake_dir / "classification_metrics.json", "r", encoding="utf-8") as f:
                fake_metrics = json.load(f)
        except Exception:
            pass
    if (fake_dir / "dataset_profile.json").exists():
        try:
            with open(fake_dir / "dataset_profile.json", "r", encoding="utf-8") as f:
                fake_profile = json.load(f)
        except Exception:
            pass
    if (fake_dir / "critic_audits_and_decisions.json").exists():
        try:
            with open(fake_dir / "critic_audits_and_decisions.json", "r", encoding="utf-8") as f:
                fake_critic = json.load(f)
        except Exception:
            pass
    datasets_detail["fake_news"] = {
        "id": "fake_news",
        "name": "Fake News Classification",
        "automind_table": "Table 2",
        "metrics": fake_metrics,
        "profile": fake_profile,
        "critic_audits": fake_critic,
        "figures": ["confusion_matrix_comparison.png"],
    }

    # 3. Customer Segmentation
    cust_dir = results_root / "customer_segmentation_experiment_20261028"
    cust_metrics = {}
    cust_critic = {}
    if (cust_dir / "clustering_metrics.json").exists():
        try:
            with open(cust_dir / "clustering_metrics.json", "r", encoding="utf-8") as f:
                cust_metrics = json.load(f)
        except Exception:
            pass
    if (cust_dir / "critic_audits_and_decisions.json").exists():
        try:
            with open(cust_dir / "critic_audits_and_decisions.json", "r", encoding="utf-8") as f:
                cust_critic = json.load(f)
        except Exception:
            pass
    datasets_detail["customer_segmentation"] = {
        "id": "customer_segmentation",
        "name": "Customer Segmentation (Clustering)",
        "automind_table": "Clustering Benchmark",
        "metrics": cust_metrics,
        "critic_audits": cust_critic,
        "figures": ["customer_segmentation_comparison.png"],
    }

    # 4. Education Survey
    edu_dir = results_root / "education_experiment_20261028"
    edu_metrics = {}
    edu_profile = {}
    edu_cronbach = {}
    edu_critic = {}
    if (edu_dir / "education_metrics.json").exists():
        try:
            with open(edu_dir / "education_metrics.json", "r", encoding="utf-8") as f:
                edu_metrics = json.load(f)
        except Exception:
            pass
    if (edu_dir / "dataset_profile.json").exists():
        try:
            with open(edu_dir / "dataset_profile.json", "r", encoding="utf-8") as f:
                edu_profile = json.load(f)
        except Exception:
            pass
    if (edu_dir / "cronbach_alpha_reliability.json").exists():
        try:
            with open(edu_dir / "cronbach_alpha_reliability.json", "r", encoding="utf-8") as f:
                edu_cronbach = json.load(f)
        except Exception:
            pass
    if (edu_dir / "critic_audits_and_decisions.json").exists():
        try:
            with open(edu_dir / "critic_audits_and_decisions.json", "r", encoding="utf-8") as f:
                edu_critic = json.load(f)
        except Exception:
            pass
    datasets_detail["education"] = {
        "id": "education",
        "name": "Education Survey (Psychometric)",
        "automind_table": "Tables 4 & 5",
        "metrics": edu_metrics,
        "profile": edu_profile,
        "cronbach_alpha": edu_cronbach,
        "critic_audits": edu_critic,
    }

    # 5. ShopMania
    shop_dir = results_root / "shopmania_experiment_20261028"
    shop_metrics = {}
    shop_profile = {}
    shop_critic = {}
    if (shop_dir / "shopmania_metrics.json").exists():
        try:
            with open(shop_dir / "shopmania_metrics.json", "r", encoding="utf-8") as f:
                shop_metrics = json.load(f)
        except Exception:
            pass
    if (shop_dir / "dataset_profile.json").exists():
        try:
            with open(shop_dir / "dataset_profile.json", "r", encoding="utf-8") as f:
                shop_profile = json.load(f)
        except Exception:
            pass
    if (shop_dir / "critic_audits_and_decisions.json").exists():
        try:
            with open(shop_dir / "critic_audits_and_decisions.json", "r", encoding="utf-8") as f:
                shop_critic = json.load(f)
        except Exception:
            pass
    datasets_detail["shopmania"] = {
        "id": "shopmania",
        "name": "ShopMania E-Commerce Taxonomy",
        "automind_table": "Table 6",
        "metrics": shop_metrics,
        "profile": shop_profile,
        "critic_audits": shop_critic,
    }

    # Check for genuine Gemini Live Benchmark results
    gemini_dir = results_root / "gemini_live_5_datasets_20261029"
    gemini_summary = []
    gemini_manifest = None
    if (gemini_dir / "GEMINI_FIVE_DATASET_BENCHMARK.json").exists():
        try:
            with open(gemini_dir / "GEMINI_FIVE_DATASET_BENCHMARK.json", "r", encoding="utf-8") as f:
                gemini_summary = json.load(f)
        except Exception:
            pass
    if (gemini_dir / "GEMINI_EXPERIMENT_MANIFEST.json").exists():
        try:
            with open(gemini_dir / "GEMINI_EXPERIMENT_MANIFEST.json", "r", encoding="utf-8") as f:
                gemini_manifest = json.load(f)
        except Exception:
            pass

    return {
        "status": "success",
        "benchmark_name": "DataInvest AI 5-Dataset Paired Benchmark Suite (AutoMind Tables 1-6 Alignment)",
        "provenance": {
            "mode": "OFFLINE_DETERMINISTIC",
            "gemini_model_configured": "gemini-3.5-flash-lite",
            "api_calls_made": 0,
            "cost_usd": 0.0,
            "random_state": 42,
            "validation_timestamp": "2026-10-29T23:00:00Z",
        },
        "has_gemini_live_results": bool(gemini_summary),
        "gemini_live_summary": gemini_summary,
        "gemini_manifest": gemini_manifest,
        "offline_deterministic_summary": consolidated_summary,
        "consolidated_summary": consolidated_summary,
        "automind_aligned_tables": automind_tables,
        "critic_interventions": critic_interventions,
        "manifest": manifest,
        "datasets": datasets_detail,
        "citation": {
            "title": "AutoMind: Automated Insight Discovery via Multi-Agent Navigation",
            "journal": "IEEE Access",
            "year": 2026,
            "volume": 14,
            "pages": "23936–23955",
            "doi": "10.1109/ACCESS.2026.3661202",
        },
    }


@app.get(
    "/api/benchmarks/figure/{fig_name:path}",
    tags=["Benchmarks"],
)
async def get_benchmark_figure_endpoint(fig_name: str):
    """Serve benchmark PNG figures safely with path-traversal protection."""
    # Sanitize fig_name: extract only the basename to strictly eliminate directory traversal
    safe_fig_name = Path(fig_name).name
    if not safe_fig_name or safe_fig_name != fig_name:
        return JSONResponse(status_code=400, content={"error": "Invalid figure filename."})

    results_root = get_benchmark_results_root()

    candidate_dirs = [
        results_root / "consolidated_5_datasets_20261028",
        results_root / "customer_segmentation_experiment_20261028",
        results_root / "fake_news_experiment_20261028",
        results_root / "heart_experiment_candidate_20261028" / "figures",
        results_root / "classification" / "figures",
        results_root / "figures",
    ]
    for d in candidate_dirs:
        if not d.exists():
            continue
        target = (d / safe_fig_name).resolve()
        # Verify target is strictly within directory d to prevent directory traversal escapes
        try:
            if target.is_relative_to(d.resolve()) and target.is_file():
                return FileResponse(str(target), media_type="image/png")
        except AttributeError:
            if str(target).startswith(str(d.resolve())) and target.is_file():
                return FileResponse(str(target), media_type="image/png")

    return JSONResponse(status_code=404, content={"error": f"Figure '{safe_fig_name}' not found."})
