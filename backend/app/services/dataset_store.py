"""
backend/app/services/dataset_store.py
======================================
In-memory dataset storage and resolution service for DataInvest AI.
Enables statistical tools and agents to access uploaded datasets by identifier
without re-parsing the file on every analysis task.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional
import pandas as pd

_STORE_LOCK = threading.Lock()
_DATASET_CACHE: dict[str, pd.DataFrame] = {}
_MAX_CACHE_SIZE = 50

SUPPORTED_DATASET_EXTENSIONS = {".csv", ".txt", ".data", ".tsv"}


def store_dataset(dataset_id: str, df: pd.DataFrame) -> str:
    """
    Store a DataFrame in the in-memory cache under dataset_id.
    Returns the stored dataset_id.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("store_dataset requires a valid pandas DataFrame.")

    with _STORE_LOCK:
        # Evict oldest entry if cache exceeds limit
        if len(_DATASET_CACHE) >= _MAX_CACHE_SIZE and dataset_id not in _DATASET_CACHE:
            first_key = next(iter(_DATASET_CACHE))
            del _DATASET_CACHE[first_key]

        _DATASET_CACHE[dataset_id] = df.copy(deep=True)

    return dataset_id


def get_dataset(dataset_id: str) -> Optional[pd.DataFrame]:
    """Retrieve a copy of a stored DataFrame from cache."""
    with _STORE_LOCK:
        df = _DATASET_CACHE.get(dataset_id)
        if df is not None:
            return df.copy(deep=True)
    return None


def _load_from_absolute_path(path_str: str) -> pd.DataFrame:
    """
    Safely load a dataset from an explicitly supplied absolute path.
    Validates file existence, regular file status, and supported tabular extension.
    Does not copy, move, or modify the source file.
    """
    from app.services import upload_service

    p = Path(path_str)
    if not p.is_absolute():
        raise ValueError(f"Path must be absolute: '{path_str}'")

    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found at absolute path: '{path_str}'")

    if not p.is_file():
        raise ValueError(f"Dataset path is not a regular file: '{path_str}'")

    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_DATASET_EXTENSIONS:
        raise ValueError(
            f"Unsupported dataset file extension '{suffix}'. Supported extensions: "
            f"{', '.join(sorted(SUPPORTED_DATASET_EXTENSIONS))}"
        )

    try:
        with open(p, "rb") as f:
            content = f.read()
    except Exception as exc:
        raise IOError(f"Failed to read dataset file at '{path_str}': {exc}") from exc

    parsed_df, parse_err = upload_service.parse_csv(content, filename=p.name)
    if parsed_df is not None and not parsed_df.empty:
        return parsed_df

    # Fallback to direct pd.read_csv if upload_service parser returned empty
    try:
        df = pd.read_csv(p)
        if df is not None and not df.empty:
            df.attrs["dataset_name"] = p.name
            return df
    except Exception:
        pass

    error_msg = parse_err or "File appears empty or could not be parsed as tabular data."
    raise ValueError(f"Failed to parse dataset from '{path_str}': {error_msg}")


def resolve_dataset(dataset_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Resolve dataset by ID from cache, explicit absolute path, or local data directory fallback.
    """
    if not dataset_id:
        with _STORE_LOCK:
            if _DATASET_CACHE:
                # Return most recently added dataset
                last_key = list(_DATASET_CACHE.keys())[-1]
                return _DATASET_CACHE[last_key].copy(deep=True)
        # Disk fallback
        return _load_from_disk("sample.csv")

    # 1. Try cache
    cached = get_dataset(dataset_id)
    if cached is not None:
        return cached

    # 2. Check if dataset_id is an explicit absolute path
    if os.path.isabs(dataset_id):
        df = _load_from_absolute_path(dataset_id)
        with _STORE_LOCK:
            _DATASET_CACHE[dataset_id] = df.copy(deep=True)
        return df

    # 3. Try standard project disk directories
    df = _load_from_disk(dataset_id)
    if df is not None:
        # Cache recovered dataset
        with _STORE_LOCK:
            _DATASET_CACHE[dataset_id] = df.copy(deep=True)
        return df

    return None


def _load_from_disk(filename: str) -> Optional[pd.DataFrame]:
    """
    Load dataset from standard project data directories.
    Uses upload_service.parse_csv for robust delimiter detection,
    headerless benchmark schemas, and multi-encoding resilience.
    """
    from app.services import upload_service

    clean_name = os.path.basename(filename)
    names_to_try = [clean_name]
    if "." not in clean_name:
        names_to_try.extend([f"{clean_name}.csv", f"{clean_name}.txt"])

    dir_candidates = [
        "data",
        os.path.join("..", "data"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"),
        os.path.join(os.path.dirname(__file__), "..", "..", "data"),
    ]

    for target_name in names_to_try:
        for base_dir in dir_candidates:
            path = os.path.join(base_dir, target_name)
            if os.path.exists(path) and os.path.isfile(path):
                try:
                    with open(path, "rb") as f:
                        content = f.read()
                    parsed_df, parse_err = upload_service.parse_csv(content, filename=target_name)
                    if parsed_df is not None and not parsed_df.empty:
                        return parsed_df
                except Exception:
                    pass
                # Secondary fallback: standard pd.read_csv
                try:
                    df = pd.read_csv(path)
                    if df is not None and not df.empty:
                        return df
                except Exception:
                    continue

    return None


def clear_cache() -> None:
    """Clear all cached datasets (primarily for testing)."""
    with _STORE_LOCK:
        _DATASET_CACHE.clear()
