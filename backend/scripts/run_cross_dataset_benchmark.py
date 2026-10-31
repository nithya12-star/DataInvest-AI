"""
backend/scripts/run_cross_dataset_benchmark.py
=============================================
Reproducible offline cross-dataset benchmark harness for DataInvest AI.
Evaluates paired Baseline vs Proposed pipeline performance across candidate datasets
with runtime duration tracking, rule-level Critic auditing, and structured reporting.

Critical Safety & Integrity Controls:
- 100% OFFLINE BY DEFAULT: Uses deterministic fixture replay or rule-based plans.
- ZERO LIVE API CALLS: Never silently calls Gemini or external LLMs.
- NO OVERWRITING: Does not alter historical saved runs in backend/data/runs/.
- REPORTING: Generates consolidated CSV, Markdown, and JSON benchmark summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure backend directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.modules.profiler import profile_dataset
from app.services.dataset_store import resolve_dataset, store_dataset
from app.services.experiment_runner import run_paired_experiment
from app.agents.planner_agent import PlannerAgent
from app.schemas.planner_schema import InvestigationTask, PlannerResponse, SelectedColumn

logger = logging.getLogger("datainvest.benchmark")

DEFAULT_CANDIDATE_DATASETS: list[str] = [
    "data/sample.csv",
    "data/student-mat.csv",
    "data/winequality-red.csv",
    "data/processed.cleveland.txt",
]

APPROVED_EXTERNAL_RESULTS_ROOTS = [
    Path(r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\DataInvest_Pilot_Results").resolve(),
    Path(r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\DataInvest_Research_Results").resolve(),
]
APPROVED_EXTERNAL_RESULTS_ROOT = APPROVED_EXTERNAL_RESULTS_ROOTS[0]


def validate_safe_output_dir(out_dir: str | Path, allow_temp: bool = True) -> Path:
    """
    Validate output directory safety and protect against path traversal attacks.
    Permits paths within:
      1. The project workspace (PROJECT_ROOT)
      2. The explicitly designated external results roots (APPROVED_EXTERNAL_RESULTS_ROOTS)
      3. Standard system temporary directories (when allow_temp=True, used in testing)
    Rejects paths that escape permitted roots or traverse to arbitrary system locations.
    """
    p = Path(out_dir)
    resolved = p.resolve()

    in_project = False
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
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
            f"Path traversal or unauthorized output path detected: '{out_dir}' "
            f"resolves outside permitted workspace, approved pilot results root, or temp directories: {resolved}"
        )

    return resolved


def build_canonical_fixture_plan(dataset_id: str, profile: dict[str, Any]) -> PlannerResponse:
    """
    Build a deterministic, reproducible fixture plan tailored to the dataset.
    Uses the offline deterministic PlannerAgent engine to generate structured,
    schema-valid investigation tasks covering diverse statistical categories:
    - chi-square contingency & Cochran guideline checks
    - normality & parametric/non-parametric group comparisons
    - correlation, linearity, and outlier/leverage diagnostics
    """
    clean_name = os.path.basename(dataset_id)
    agent = PlannerAgent(use_llm=False)
    objective = (
        f"Conduct offline benchmark investigation of {clean_name}: "
        "analyze statistical associations, group variations, distributions, and assumption integrity."
    )
    plan = agent.plan(profile=profile, objective=objective)
    plan.planner_mode = "deterministic_fixture"
    return plan


def run_single_dataset_benchmark(
    dataset_path: str,
    max_reinvestigations: int = 1,
    use_canonical_fixtures: bool = True,
    is_research_mode: bool = False,
    repetition_id: int = 1,
    gemini_client: Optional[Any] = None,
    log_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    """
    Execute a single dataset paired experiment in benchmark harness.
    Never persists runs to disk (protecting historical runs).
    Returns structured metric dictionary including repetition_id and rule diagnostics.
    """
    dataset_name = os.path.basename(dataset_path)
    planner_mode_desc = (
        "genuine_llm"
        if is_research_mode
        else ("deterministic_fixture" if use_canonical_fixtures else "deterministic_rule")
    )

    res_entry: dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_id": dataset_name,
        "dataset_path": dataset_path,
        "repetition_id": repetition_id,
        "status": "PENDING",
        "n_rows": None,
        "n_cols": None,
        "execution_mode": "research_run" if is_research_mode else "offline_fixture_replay",
        "is_offline_fixture": not is_research_mode,
        "planner_mode": planner_mode_desc,
        "tasks_executed": 0,
        "initial_findings_count": 0,
        "verified_findings_count": 0,
        "findings_requiring_reinvestigation": 0,
        "findings_revised": 0,
        "findings_unresolved": 0,
        "cochran_violations_count": "N/A",
        "normality_violations_count": "N/A",
        "leverage_outlier_violations_count": "N/A",
        "baseline_score": 0.0,
        "proposed_score": 0.0,
        "score_delta": 0.0,
        "total_experiment_duration_seconds": 0.0,
        "shared_duration_seconds": 0.0,
        "planning_duration_seconds": 0.0,
        "analysis_duration_seconds": 0.0,
        "baseline_duration_seconds": 0.0,
        "proposed_duration_seconds": 0.0,
        "critic_duration_seconds": 0.0,
        "verification_duration_seconds": 0.0,
        "proposed_overhead_seconds": 0.0,
        "error_message": "",
    }

    # 1. Resolve dataset
    try:
        df = resolve_dataset(dataset_path)
        if df is None:
            # Try by clean basename
            df = resolve_dataset(dataset_name)

        if df is None or df.empty:
            res_entry["status"] = "FAILED"
            res_entry["error_message"] = f"Dataset could not be loaded or is empty: {dataset_path}"
            return res_entry
    except Exception as exc:
        res_entry["status"] = "FAILED"
        res_entry["error_message"] = f"Resolution error: {str(exc)}"
        return res_entry

    # Cache dataset in store
    store_dataset(dataset_path, df)
    store_dataset(dataset_name, df)

    res_entry["n_rows"] = int(len(df))
    res_entry["n_cols"] = int(len(df.columns))

    # 2. Profile dataset
    try:
        profile = profile_dataset(df)
    except Exception as exc:
        res_entry["status"] = "FAILED"
        res_entry["error_message"] = f"Profiling failed: {str(exc)}"
        return res_entry

    # 3. Setup planning function
    planner_fn = None
    if not is_research_mode:
        if use_canonical_fixtures:
            def fixture_planner(profile, objective, dataset_id, **kwargs):
                return build_canonical_fixture_plan(dataset_id, profile)
            planner_fn = fixture_planner
        else:
            offline_planner_inst = PlannerAgent(use_llm=False)
            def rule_planner(profile, objective, dataset_id, **kwargs):
                return offline_planner_inst.plan(profile=profile, objective=objective)
            planner_fn = rule_planner

    # 4. Run paired experiment
    try:
        resp = run_paired_experiment(
            profile=profile,
            dataset_id=dataset_name,
            objective=f"Cross-dataset benchmark investigation for {dataset_name} (Rep {repetition_id})",
            max_reinvestigations=max_reinvestigations,
            persist_runs=False,  # CRITICAL: Protect historical runs
            gemini_client=gemini_client,
            planner_fn=planner_fn,
            is_offline_fixture=not is_research_mode,
            execution_mode="research_run" if is_research_mode else "offline_fixture_replay",
            log_dir=log_dir,
        )
    except Exception as exc:
        res_entry["status"] = "FAILED"
        res_entry["error_message"] = f"Experiment execution failed: {str(exc)}"
        return res_entry

    # 5. Extract structured metrics
    res_entry["status"] = "SUCCESS"
    res_entry["tasks_executed"] = resp.baseline.total_tasks_executed
    res_entry["initial_findings_count"] = resp.comparison.initial_findings_count or 0
    res_entry["verified_findings_count"] = resp.comparison.verified_findings_count
    res_entry["findings_requiring_reinvestigation"] = resp.comparison.findings_requiring_reinvestigation or 0
    res_entry["findings_revised"] = resp.comparison.findings_revised or 0
    res_entry["findings_unresolved"] = resp.comparison.findings_unresolved or 0

    res_entry["cochran_violations_count"] = (
        resp.comparison.cochran_violations_count
        if resp.comparison.cochran_violations_count is not None
        else "N/A"
    )
    res_entry["normality_violations_count"] = (
        resp.comparison.normality_violations_count
        if resp.comparison.normality_violations_count is not None
        else "N/A"
    )
    res_entry["leverage_outlier_violations_count"] = (
        resp.comparison.leverage_outlier_violations_count
        if resp.comparison.leverage_outlier_violations_count is not None
        else "N/A"
    )

    res_entry["baseline_score"] = resp.comparison.baseline_score
    res_entry["proposed_score"] = resp.comparison.proposed_score
    res_entry["score_delta"] = resp.comparison.overall_score_delta

    res_entry["total_experiment_duration_seconds"] = resp.comparison.total_experiment_duration_seconds or 0.0
    res_entry["shared_duration_seconds"] = resp.comparison.shared_duration_seconds or 0.0
    res_entry["planning_duration_seconds"] = resp.comparison.planning_duration_seconds or 0.0
    res_entry["analysis_duration_seconds"] = resp.comparison.analysis_duration_seconds or 0.0
    res_entry["baseline_duration_seconds"] = resp.comparison.baseline_duration_seconds or 0.0
    res_entry["proposed_duration_seconds"] = resp.comparison.proposed_duration_seconds or 0.0
    res_entry["critic_duration_seconds"] = resp.comparison.critic_duration_seconds or 0.0
    res_entry["verification_duration_seconds"] = resp.comparison.verification_duration_seconds or 0.0
    res_entry["proposed_overhead_seconds"] = resp.comparison.proposed_overhead_seconds or 0.0

    return res_entry


def run_benchmark_suite(
    datasets: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
    repetitions: int = 1,
    use_canonical_fixtures: bool = True,
    is_research_mode: bool = False,
    confirm_live_api: bool = False,
    overwrite: bool = False,
    log_dir: Optional[str] = None,
) -> Tuple[list[dict[str, Any]], dict[str, str]]:
    """
    Run benchmark matrix across candidate datasets with repetition support and save reports.
    """
    # Guard against accidental live external API calls
    if is_research_mode and not confirm_live_api:
        raise RuntimeError(
            "CRITICAL EXECUTION BOUNDARY: Real research-mode runs require explicit opt-in "
            "via confirm_live_api=True (--confirm-live-api). Stopping immediately before making external calls."
        )

    if repetitions < 1:
        raise ValueError(f"Repetitions must be a positive integer >= 1 (got {repetitions}).")

    target_datasets = datasets or list(DEFAULT_CANDIDATE_DATASETS)

    # Route default output directory based on execution mode
    if output_dir is None:
        if is_research_mode:
            resolved_out_dir = PROJECT_ROOT / "reports" / "benchmarks" / "live_research"
        else:
            resolved_out_dir = PROJECT_ROOT / "reports" / "benchmarks"
    else:
        resolved_out_dir = validate_safe_output_dir(output_dir)

    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    resolved_log_dir: Optional[Path] = None
    if log_dir is not None:
        resolved_log_dir = validate_safe_output_dir(log_dir)

    results: list[dict[str, Any]] = []

    print(f"=== Starting Cross-Dataset Benchmark Harness ===")
    print(f"Datasets: {len(target_datasets)}")
    print(f"Repetitions per Dataset: {repetitions} (Total Trials: {len(target_datasets) * repetitions})")
    print(f"Mode: {'RESEARCH (LIVE LLM)' if is_research_mode else 'OFFLINE FIXTURE REPLAY'}")
    print(f"Output Directory: {resolved_out_dir}")
    if resolved_log_dir:
        print(f"Log Directory: {resolved_log_dir}")
    print("-" * 60)

    trial_count = len(target_datasets) * repetitions
    trial_idx = 1
    for rep in range(1, repetitions + 1):
        for ds_path in target_datasets:
            print(
                f"[{trial_idx}/{trial_count}] (Rep {rep}/{repetitions}) Evaluating: {ds_path} ...",
                end=" ",
                flush=True,
            )
            t0 = time.perf_counter()
            entry = run_single_dataset_benchmark(
                dataset_path=ds_path,
                use_canonical_fixtures=use_canonical_fixtures,
                is_research_mode=is_research_mode,
                repetition_id=rep,
                log_dir=resolved_log_dir,
            )
            elapsed = time.perf_counter() - t0
            print(f"{entry['status']} ({elapsed:.2f}s)")
            if entry["status"] == "FAILED":
                print(f"    Error: {entry['error_message']}")
            results.append(entry)
            trial_idx += 1

    # Generate reports
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_filename = f"benchmark_summary_{ts}"

    csv_path = resolved_out_dir / f"{base_filename}.csv"
    md_path = resolved_out_dir / f"{base_filename}.md"
    json_path = resolved_out_dir / f"{base_filename}.json"

    # Handle overwrite protection
    if not overwrite:
        counter = 1
        while csv_path.exists() or md_path.exists() or json_path.exists():
            csv_path = resolved_out_dir / f"{base_filename}_{counter}.csv"
            md_path = resolved_out_dir / f"{base_filename}_{counter}.md"
            json_path = resolved_out_dir / f"{base_filename}_{counter}.json"
            counter += 1

    _write_csv_report(results, csv_path)
    _write_markdown_report(results, md_path, is_research_mode, repetitions)
    _write_json_report(results, json_path, is_research_mode, repetitions)

    print("-" * 60)
    print(f"Benchmark run complete. Generated reports:")
    print(f"  CSV:      {csv_path}")
    print(f"  Markdown: {md_path}")
    print(f"  JSON:     {json_path}")

    return results, {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "json": str(json_path),
    }


def _write_csv_report(results: list[dict[str, Any]], filepath: Path) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def _write_markdown_report(
    results: list[dict[str, Any]],
    filepath: Path,
    is_research_mode: bool,
    repetitions: int,
) -> None:
    mode_desc = (
        "Genuine LLM Research Run (Live Gemini Planner)"
        if is_research_mode
        else "Offline Deterministic Fixture Replay (Verification Validation)"
    )

    lines: list[str] = [
        "# DataInvest AI — Cross-Dataset Benchmark Summary",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Execution Mode:** {mode_desc}",
        f"**Offline Fixture Validation:** {'No (Genuine Research)' if is_research_mode else 'Yes (Deterministic Offline Testing)'}",
        f"**Repetitions per Dataset:** {repetitions}",
        "",
        "> [!NOTE]",
        "> This benchmark harness validates runtime instrumentation, Critic rule-level metrics,",
        "> and pipeline reproducibility across multiple real-world datasets.",
        "> Shared planning/analysis time is isolated from proposed Critic and reinvestigation overhead.",
        "",
        "> [!IMPORTANT]",
        "> **Scientific Notice on Per-Dataset Reporting:**",
        "> Primary outcomes must be evaluated separately for each dataset.",
        "> Results across heterogeneous datasets (e.g. 60-row synthetic `sample.csv` vs 1,599-row continuous `winequality-red.csv`)",
        "> are not pooled into a single composite score. Each dataset possesses unique distributional",
        "> characteristics, feature types, and sample sizes that require isolated statistical interpretation.",
        "",
        "## Consolidated Trial Matrix",
        "",
        "| Dataset | Rep | Dimensions | Status | Tasks | Base Score | Prop Score | Delta | Reinv Req | Revised | Cochran | Normality | Leverage | Shared Time (s) | Overhead (s) | Total Time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        dim_str = f"{r['n_rows']}x{r['n_cols']}" if r["n_rows"] is not None else "N/A"
        base_s = f"{r['baseline_score']:.1f}" if r["status"] == "SUCCESS" else "-"
        prop_s = f"{r['proposed_score']:.1f}" if r["status"] == "SUCCESS" else "-"
        delta_s = f"{r['score_delta']:+.1f}" if r["status"] == "SUCCESS" else "-"
        shared_t = f"{r['shared_duration_seconds']:.3f}" if r["status"] == "SUCCESS" else "-"
        overhead_t = f"{r['proposed_overhead_seconds']:.3f}" if r["status"] == "SUCCESS" else "-"
        total_t = f"{r['total_experiment_duration_seconds']:.3f}" if r["status"] == "SUCCESS" else "-"

        lines.append(
            f"| `{r['dataset_name']}` | {r.get('repetition_id', 1)} | {dim_str} | **{r['status']}** | {r['tasks_executed']} | "
            f"{base_s} | {prop_s} | {delta_s} | {r['findings_requiring_reinvestigation']} | "
            f"{r['findings_revised']} | {r['cochran_violations_count']} | {r['normality_violations_count']} | "
            f"{r['leverage_outlier_violations_count']} | {shared_t} | {overhead_t} | {total_t} |"
        )

    # Per-Dataset Breakdown Section
    lines.extend([
        "",
        "## Per-Dataset Performance Breakdown",
        "",
    ])

    unique_datasets = sorted(list({r["dataset_name"] for r in results}))
    for ds_name in unique_datasets:
        ds_runs = [r for r in results if r["dataset_name"] == ds_name]
        successful_runs = [r for r in ds_runs if r["status"] == "SUCCESS"]

        lines.append(f"### Dataset: `{ds_name}`")
        if successful_runs:
            mean_base = sum(r["baseline_score"] for r in successful_runs) / len(successful_runs)
            mean_prop = sum(r["proposed_score"] for r in successful_runs) / len(successful_runs)
            mean_delta = sum(r["score_delta"] for r in successful_runs) / len(successful_runs)
            mean_shared = sum(r["shared_duration_seconds"] for r in successful_runs) / len(successful_runs)
            mean_overhead = sum(r["proposed_overhead_seconds"] for r in successful_runs) / len(successful_runs)
            total_reinv = sum(r["findings_requiring_reinvestigation"] for r in successful_runs)
            total_revised = sum(r["findings_revised"] for r in successful_runs)

            lines.append(f"- **Completed Trials:** {len(successful_runs)} / {len(ds_runs)}")
            lines.append(f"- **Mean Baseline Score:** {mean_base:.2f}")
            lines.append(f"- **Mean Proposed Score:** {mean_prop:.2f}")
            lines.append(f"- **Mean Score Delta ($\\Delta$):** {mean_delta:+.2f}")
            lines.append(f"- **Reinvestigations:** {total_reinv} flagged, {total_revised} revised")
            lines.append(f"- **Mean Shared Runtime:** {mean_shared:.3f}s")
            lines.append(f"- **Mean Proposed Overhead:** {mean_overhead:.3f}s")
            if not is_research_mode:
                lines.append("- *Note:* In offline fixture replay, repeated runs produce identical deterministic plans and values.")
        else:
            lines.append(f"- **Status:** All trials failed for this dataset.")
            for r in ds_runs:
                lines.append(f"  - Rep {r.get('repetition_id', 1)}: `{r.get('error_message', 'Unknown error')}`")
        lines.append("")

    # Rule-level details & failures section
    lines.extend([
        "## Execution Notes & Observations",
        "",
    ])

    failures = [r for r in results if r["status"] != "SUCCESS"]
    if failures:
        lines.append("### Partial Failures / Unprocessed Datasets")
        for f in failures:
            lines.append(f"- **{f['dataset_name']} (Rep {f.get('repetition_id', 1)})**: `{f['error_message']}`")
        lines.append("")
    else:
        lines.append("All target datasets completed pipeline execution without failure.")
        lines.append("")

    lines.extend([
        "### Statistical Rule Audit Breakdown",
        "- **Cochran (1954) Guidelines**: Evaluates contingency tables for expected counts < 1.0 or >20% cells below 5.0. Recorded as N/A when no categorical tests were evaluated.",
        "- **Normality & Distributional Skewness**: Flags severe skewness (>1.5) or small-group distribution violations, triggering Mann-Whitney or Kruskal-Wallis reinvestigations.",
        "- **Outlier-Influence Diagnostic**: Identifies observations that may disproportionately influence Pearson correlation. This diagnostic alone does not establish that the linearity assumption is violated (flags bivariate outlier points |z| > 3.0).",
        "",
        "### Fairness in Runtime Instrumentation",
        "- **Shared Phase Time**: Measures planning and Phase 3 statistical execution. It is shared across both branches and never attributed to either baseline or proposed execution alone.",
        "- **Proposed Overhead**: Strictly isolates Phase 4 Critic verification and Phase 5 controlled reinvestigation.",
        "",
    ])

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_json_report(
    results: list[dict[str, Any]],
    filepath: Path,
    is_research_mode: bool,
    repetitions: int,
) -> None:
    unique_datasets = sorted(list({r["dataset_name"] for r in results}))
    per_dataset: dict[str, Any] = {}

    for ds_name in unique_datasets:
        ds_runs = [r for r in results if r["dataset_name"] == ds_name]
        successful = [r for r in ds_runs if r["status"] == "SUCCESS"]
        per_dataset[ds_name] = {
            "total_trials": len(ds_runs),
            "successful_trials": len(successful),
            "failed_trials": len(ds_runs) - len(successful),
            "mean_baseline_score": (
                round(sum(r["baseline_score"] for r in successful) / len(successful), 3)
                if successful
                else None
            ),
            "mean_proposed_score": (
                round(sum(r["proposed_score"] for r in successful) / len(successful), 3)
                if successful
                else None
            ),
            "mean_score_delta": (
                round(sum(r["score_delta"] for r in successful) / len(successful), 3)
                if successful
                else None
            ),
            "trials": ds_runs,
        }

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execution_mode": "research_run" if is_research_mode else "offline_fixture_replay",
        "is_offline_fixture": not is_research_mode,
        "repetitions_per_dataset": repetitions,
        "total_trials": len(results),
        "dataset_count": len(unique_datasets),
        "success_count": sum(1 for r in results if r["status"] == "SUCCESS"),
        "failure_count": sum(1 for r in results if r["status"] != "SUCCESS"),
        "per_dataset_summary": per_dataset,
        "results": results,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline cross-dataset benchmark harness for DataInvest AI."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional explicit list of dataset paths (defaults to audit candidate datasets).",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save benchmark reports (defaults to reports/benchmarks).",
        default=None,
    )
    parser.add_argument(
        "--log-dir",
        help="Optional dedicated directory for experiment event logs under permitted roots.",
        default=None,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions per dataset (default: 1). Must be a positive integer >= 1.",
    )
    parser.add_argument(
        "--use-rule-planner",
        action="store_true",
        help="Use dynamic deterministic PlannerAgent instead of canonical fixture plans.",
    )
    parser.add_argument(
        "--research-mode",
        action="store_true",
        help="Opt-in to live LLM planning. Requires --confirm-live-api and GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--confirm-live-api",
        action="store_true",
        help="Confirmation flag required before any real research-mode LLM call is permitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previous benchmark reports with identical names instead of incrementing.",
    )

    args = parser.parse_args()

    # Safety check on research mode
    if args.research_mode and not args.confirm_live_api:
        print(
            "ERROR: --research-mode specified without --confirm-live-api.\n"
            "By safety policy, the benchmark harness runs offline by default and refuses to call external APIs\n"
            "without explicit confirmation. Exiting without making external calls."
        )
        sys.exit(1)

    if args.repetitions < 1:
        print(f"ERROR: --repetitions must be a positive integer >= 1 (got {args.repetitions}).")
        sys.exit(1)

    try:
        run_benchmark_suite(
            datasets=args.datasets,
            output_dir=args.output_dir,
            repetitions=args.repetitions,
            use_canonical_fixtures=not args.use_rule_planner,
            is_research_mode=args.research_mode,
            confirm_live_api=args.confirm_live_api,
            overwrite=args.overwrite,
            log_dir=args.log_dir,
        )
    except Exception as exc:
        print(f"Benchmark execution aborted: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
