"""
backend/app/services/upload_service.py
=======================================
CSV and tabular file parsing service for DataInvest AI.

Tries multiple encodings, reliably detects delimiters, handles headerless
benchmark datasets, validates content, and returns a clean (DataFrame, None)
or (None, error_message) tuple.
"""

from __future__ import annotations

import io
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

# Verified benchmark schemas for known headerless and benchmark datasets
VERIFIED_BENCHMARK_COLUMNS: dict[str, list[str]] = {
    "adult.txt": [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
    ],
    "adult.data": [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
    ],
    "processed.cleveland.txt": [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
    ],
    "cleveland.data": [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
    ],
}

KNOWN_HEADERLESS_FILES = set(VERIFIED_BENCHMARK_COLUMNS.keys())

# Verified benchmark schemas with semantic column typing and sentinel rules
VERIFIED_BENCHMARK_SCHEMAS: dict[str, dict[str, Any]] = {
    "processed.cleveland.txt": {
        "columns": VERIFIED_BENCHMARK_COLUMNS["processed.cleveland.txt"],
        "column_types": {
            "age": "numerical",
            "sex": "categorical",
            "cp": "categorical",
            "trestbps": "numerical",
            "chol": "numerical",
            "fbs": "categorical",
            "restecg": "categorical",
            "thalach": "numerical",
            "exang": "categorical",
            "oldpeak": "numerical",
            "slope": "categorical",
            "ca": "categorical",
            "thal": "categorical",
            "target": "categorical",
        },
        "categorical_columns": [
            "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"
        ],
        "numerical_columns": [
            "age", "trestbps", "chol", "thalach", "oldpeak"
        ],
    },
    "cleveland.data": {
        "columns": VERIFIED_BENCHMARK_COLUMNS["cleveland.data"],
        "column_types": {
            "age": "numerical",
            "sex": "categorical",
            "cp": "categorical",
            "trestbps": "numerical",
            "chol": "numerical",
            "fbs": "categorical",
            "restecg": "categorical",
            "thalach": "numerical",
            "exang": "categorical",
            "oldpeak": "numerical",
            "slope": "categorical",
            "ca": "categorical",
            "thal": "categorical",
            "target": "categorical",
        },
        "categorical_columns": [
            "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"
        ],
        "numerical_columns": [
            "age", "trestbps", "chol", "thalach", "oldpeak"
        ],
    },
    "bank-full.csv": {
        "missing_sentinels": {
            "job": ["unknown"],
            "education": ["unknown"],
        },
        "numeric_sentinels": {
            "pdays": {
                "sentinel_value": -1,
                "description": "Previous contact unavailable (client not previously contacted)",
            }
        },
        "retained_sentinels": {
            "contact": ["unknown"],
            "poutcome": ["unknown"],
        },
        "derived_columns": {
            "pdays_contacted": {
                "source": "pdays",
                "type": "categorical",
                "description": "Indicator whether client was previously contacted (contacted vs not_contacted)",
            }
        },
    },
    "bank.csv": {
        "missing_sentinels": {
            "job": ["unknown"],
            "education": ["unknown"],
        },
        "numeric_sentinels": {
            "pdays": {
                "sentinel_value": -1,
                "description": "Previous contact unavailable (client not previously contacted)",
            }
        },
        "retained_sentinels": {
            "contact": ["unknown"],
            "poutcome": ["unknown"],
        },
        "derived_columns": {
            "pdays_contacted": {
                "source": "pdays",
                "type": "categorical",
                "description": "Indicator whether client was previously contacted (contacted vs not_contacted)",
            }
        },
    },
    "adult.txt": {
        "columns": VERIFIED_BENCHMARK_COLUMNS["adult.txt"],
        "column_types": {
            "age": "numerical",
            "workclass": "categorical",
            "fnlwgt": "numerical",
            "education": "categorical",
            "education_num": "numerical",
            "marital_status": "categorical",
            "occupation": "categorical",
            "relationship": "categorical",
            "race": "categorical",
            "sex": "categorical",
            "capital_gain": "numerical",
            "capital_loss": "numerical",
            "hours_per_week": "numerical",
            "native_country": "categorical",
            "income": "categorical",
        },
    },
    "adult.data": {
        "columns": VERIFIED_BENCHMARK_COLUMNS["adult.data"],
        "column_types": {
            "age": "numerical",
            "workclass": "categorical",
            "fnlwgt": "numerical",
            "education": "categorical",
            "education_num": "numerical",
            "marital_status": "categorical",
            "occupation": "categorical",
            "relationship": "categorical",
            "race": "categorical",
            "sex": "categorical",
            "capital_gain": "numerical",
            "capital_loss": "numerical",
            "hours_per_week": "numerical",
            "native_country": "categorical",
            "income": "categorical",
        },
    },
}


def _format_clean_categorical_series(series: pd.Series) -> pd.Series:
    """
    Format numeric-encoded categorical values (e.g. 1.0, 0.0) into clean string
    category levels ('1', '0') while strictly preserving missing values as NaN.
    """
    clean_vals = [
        str(int(float(x))) if pd.notna(x) and float(x).is_integer() else (str(x) if pd.notna(x) else np.nan)
        for x in series
    ]
    return pd.Series(clean_vals, index=series.index, dtype="category")


def _detect_delimiter(sample_text: str) -> str:
    """
    Reliably detect delimiter by evaluating candidate counts across non-empty lines.
    Prioritizes standard tabular delimiters: comma, semicolon, tab, pipe.
    """
    candidates = [",", ";", "\t", "|"]
    lines = [ln.strip() for ln in sample_text.splitlines() if ln.strip()]
    if not lines:
        return ","

    sample_lines = lines[:5]
    total_counts = {c: sum(ln.count(c) for ln in sample_lines) for c in candidates}

    best_delim = max(candidates, key=lambda c: total_counts[c])
    return best_delim if total_counts[best_delim] > 0 else ","


def derive_bank_contacted_feature(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Explicit, documented feature-derivation step for Bank Marketing dataset.
    Derives categorical 'pdays_contacted' ('not_contacted' vs 'contacted') from 'pdays' == -1.
    Does NOT mutate the input DataFrame unless inplace=True.
    """
    target = df if inplace else df.copy()
    if "pdays" in target.columns and "pdays_contacted" not in target.columns:
        target["pdays_contacted"] = pd.Series(
            np.where(target["pdays"] == -1, "not_contacted", "contacted"),
            index=target.index,
            dtype="category",
        )
    return target


def derive_benchmark_features(
    df: pd.DataFrame,
    dataset_name: Optional[str] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Explicit, documented feature-derivation helper for benchmark datasets.
    Delegates to dataset-specific feature derivation rules (e.g. Bank Marketing).
    Does NOT mutate the input DataFrame unless inplace=True.
    """
    ds_name = (dataset_name or getattr(df, "attrs", {}).get("dataset_name") or "").lower()
    if ds_name in ("bank-full.csv", "bank.csv") or "pdays" in df.columns:
        return derive_bank_contacted_feature(df, inplace=inplace)
    return df if inplace else df.copy()


def _col_letters_to_index(col_letters: str) -> int:
    """Convert Excel column letters (A, B, ..., Z, AA, ...) to 0-based index."""
    idx = 0
    for char in col_letters.upper():
        if "A" <= char <= "Z":
            idx = idx * 26 + (ord(char) - ord("A") + 1)
    return max(0, idx - 1)


def parse_xlsx(
    content: bytes,
    filename: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Parse Excel (.xlsx) file bytes into a DataFrame using standard library zipfile
    and xml.etree.ElementTree (zero external library dependency like openpyxl).
    Resolves worksheet names through workbook metadata and relationship mappings.
    Correctly aligns sparse rows and concatenates rich-text shared strings.
    """
    import zipfile
    import xml.etree.ElementTree as ET
    import re

    if not content or len(content) == 0:
        return None, "The uploaded file is empty. Please upload a non-empty Excel (.xlsx) file."

    if len(content.strip()) == 0:
        return None, "The file appears to be empty or contains only whitespace."

    try:
        with zipfile.ZipFile(io.BytesIO(content), "r") as z:
            # 1. Load shared strings with full rich-text reconstruction (DEF-07)
            shared_strings: list[str] = []
            if "xl/sharedStrings.xml" in z.namelist():
                ss_tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
                for si in ss_tree.iter():
                    if si.tag.endswith("}si") or si.tag == "si":
                        t_parts = [
                            elem.text
                            for elem in si.iter()
                            if (elem.tag.endswith("}t") or elem.tag == "t") and elem.text is not None
                        ]
                        shared_strings.append("".join(t_parts))

            # 2. Resolve worksheet names through workbook metadata and relationships (DEF-05)
            sheet_map: list[tuple[str, str]] = []  # [(sheet_name, xml_path)]
            if "xl/workbook.xml" in z.namelist():
                wb_tree = ET.fromstring(z.read("xl/workbook.xml"))
                rels_map: dict[str, str] = {}
                if "xl/_rels/workbook.xml.rels" in z.namelist():
                    rels_tree = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
                    for rel in rels_tree.iter():
                        if rel.tag.endswith("}Relationship") or rel.tag == "Relationship":
                            rel_id = rel.attrib.get("Id") or rel.attrib.get("id")
                            target = rel.attrib.get("Target") or rel.attrib.get("target")
                            rel_type = rel.attrib.get("Type") or rel.attrib.get("type") or ""
                            if rel_id and target and ("worksheet" in rel_type.lower() or "sheet" in target.lower()):
                                clean_target = target.replace("\\", "/").lstrip("/")
                                if not clean_target.startswith("xl/"):
                                    clean_target = f"xl/{clean_target}"
                                rels_map[rel_id] = clean_target

                for s in wb_tree.iter():
                    if s.tag.endswith("}sheet") or s.tag == "sheet":
                        s_name = s.attrib.get("name")
                        r_id = (
                            s.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                            or s.attrib.get("r:id")
                            or s.attrib.get("id")
                        )
                        xml_path = rels_map.get(r_id) if r_id else None
                        if s_name:
                            sheet_map.append((s_name, xml_path or ""))

            # Filter valid worksheets present in the zip package
            valid_sheet_map: list[tuple[str, str]] = []
            for s_n, s_p in sheet_map:
                if s_p and s_p in z.namelist():
                    valid_sheet_map.append((s_n, s_p))

            # Fallback if workbook.xml mapping was unavailable
            if not valid_sheet_map:
                all_sheets = [n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
                if not all_sheets:
                    return None, "No valid worksheets found in the Excel workbook."
                valid_sheet_map = [(f"Sheet{i+1}", path) for i, path in enumerate(all_sheets)]

            available_sheet_names = [s[0] for s in valid_sheet_map]

            target_sheet_path: Optional[str] = None
            if sheet_name:
                for s_n, s_p in valid_sheet_map:
                    if s_n.lower() == sheet_name.strip().lower():
                        target_sheet_path = s_p
                        break
                if not target_sheet_path:
                    return None, (
                        f"Worksheet '{sheet_name}' not found in workbook. "
                        f"Available worksheets: {', '.join(repr(n) for n in available_sheet_names)}"
                    )
            else:
                # Default selection: prefer sheet named 'data' if present, otherwise first worksheet
                data_sheet = next((s_p for s_n, s_p in valid_sheet_map if s_n.lower() == "data"), None)
                target_sheet_path = data_sheet if data_sheet else valid_sheet_map[0][1]

            sheet_tree = ET.fromstring(z.read(target_sheet_path))
            ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

            # 3. Parse cell coordinates and handle sparse rows without column shifts (DEF-04)
            parsed_rows: list[dict[int, Any]] = []
            max_col_idx = 0

            for row in sheet_tree.findall(".//ns:row", ns):
                row_dict: dict[int, Any] = {}
                seq_col = 0
                for c in row.findall("ns:c", ns):
                    c_ref = c.attrib.get("r")
                    if c_ref:
                        m = re.match(r"^([A-Za-z]+)", c_ref)
                        col_idx = _col_letters_to_index(m.group(1)) if m else seq_col
                    else:
                        col_idx = seq_col
                    seq_col = col_idx + 1

                    if col_idx > max_col_idx:
                        max_col_idx = col_idx

                    t_attr = c.attrib.get("t")
                    v_tag = c.find("ns:v", ns)
                    val = v_tag.text if v_tag is not None else None

                    if t_attr == "s" and val is not None:
                        try:
                            idx = int(val)
                            val = shared_strings[idx] if idx < len(shared_strings) else val
                        except ValueError:
                            pass
                    elif t_attr == "inlineStr":
                        is_tag = c.find("ns:is", ns)
                        if is_tag is not None:
                            val = "".join(
                                elem.text
                                for elem in is_tag.iter()
                                if (elem.tag.endswith("}t") or elem.tag == "t") and elem.text is not None
                            )

                    if val is not None:
                        row_dict[col_idx] = val

                if row_dict:
                    parsed_rows.append(row_dict)

        if not parsed_rows:
            return None, "The Excel file contains no data."

        num_columns = max_col_idx + 1
        header_dict = parsed_rows[0]

        headers: list[str] = []
        for i in range(num_columns):
            h_val = header_dict.get(i)
            if h_val is not None and str(h_val).strip():
                headers.append(str(h_val).strip())
            else:
                headers.append(f"Column_{i+1}")

        data_rows: list[list[Any]] = []
        for r_dict in parsed_rows[1:]:
            row_vals = [r_dict.get(i, None) for i in range(num_columns)]
            if any(v is not None for v in row_vals):
                data_rows.append(row_vals)

        if not data_rows:
            return None, "The Excel file contains no data rows below the header."

        df = pd.DataFrame(data_rows, columns=headers)

        for col in df.columns:
            try:
                num = pd.to_numeric(df[col], errors="coerce")
                if num.notna().sum() > 0.8 * len(num):
                    df[col] = num
            except Exception:
                pass

        if clean_filename := os.path.basename(filename or "").lower():
            df.attrs["dataset_name"] = clean_filename

        return df, None
    except zipfile.BadZipFile:
        return None, "The uploaded file is not a valid Excel (.xlsx) file or is corrupted."
    except Exception as exc:
        return None, f"Failed to parse Excel file: {str(exc)}"


def parse_csv(
    content: bytes,
    filename: Optional[str] = None,
    apply_benchmark_sentinels: bool = True,
    sheet_name: Optional[str] = None,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Parse CSV, TXT, or XLSX tabular bytes into a DataFrame.

    Parameters
    ----------
    content : bytes — raw file content from the upload.
    filename : str, optional — original filename for benchmark detection.
    apply_benchmark_sentinels : bool, default True — whether to apply benchmark-specific
                                missing-value sentinel mappings (e.g. 'unknown' in
                                Bank Marketing demographic fields). Raw values and
                                full audit provenance are always captured on df.attrs.
    sheet_name : str, optional — worksheet name to parse for multi-sheet Excel workbooks.

    Returns
    -------
    (DataFrame, None)  on success.
    (None, str)        on failure — the string is a user-friendly error message.
                       Never contains a Python traceback.
    """
    clean_filename = os.path.basename(filename or "").lower()
    if clean_filename.endswith(".xlsx"):
        return parse_xlsx(content, filename=clean_filename, sheet_name=sheet_name)

    if not content or len(content) == 0:
        return None, "The uploaded file is empty. Please upload a non-empty CSV file."

    if len(content.strip()) == 0:
        return None, "The file appears to be empty or contains only whitespace."

    is_known_headerless = clean_filename in KNOWN_HEADERLESS_FILES

    encodings = ("utf-8", "utf-8-sig", "latin-1", "cp1252")

    for encoding in encodings:
        try:
            # Inspect delimiter from decoded sample text
            sample_text = content[:8192].decode(encoding, errors="ignore")
            sep = _detect_delimiter(sample_text)

            header_setting: Optional[int] | str = None if is_known_headerless else "infer"

            df = pd.read_csv(
                io.BytesIO(content),
                encoding=encoding,
                sep=sep,
                header=header_setting,
                skipinitialspace=True,
                na_values=["?"],
            )

            if len(df) == 0:
                return None, (
                    "The CSV file was read successfully but contains no data rows. "
                    "Please upload a file with at least one row below the header."
                )

            # Assign verified benchmark columns or neutral string column labels
            if is_known_headerless:
                verified_cols = VERIFIED_BENCHMARK_COLUMNS.get(clean_filename)
                if verified_cols and len(df.columns) == len(verified_cols):
                    df.columns = list(verified_cols)
                else:
                    df.columns = [str(c) for c in df.columns]
            else:
                # Ensure all column labels are clean strings
                df.columns = [str(c).strip() for c in df.columns]

            # Apply verified benchmark schemas and dataset-specific sentinel rules
            if clean_filename in VERIFIED_BENCHMARK_SCHEMAS:
                schema = VERIFIED_BENCHMARK_SCHEMAS[clean_filename]

                # A. Cleveland dataset: format documented nominal/categorical columns
                # as clean category dtypes while keeping continuous variables numeric and NaNs preserved
                for cat_col in schema.get("categorical_columns", []):
                    if cat_col in df.columns:
                        df[cat_col] = _format_clean_categorical_series(df[cat_col])

                # Build detailed provenance distinguishing raw sentinels from interpreted missing values
                missing_provenance: dict[str, Any] = {
                    "rules_version": "2026.1_sentinel_aware",
                    "dataset_name": clean_filename,
                    "raw_dimensions": [int(len(df)), int(len(df.columns))],
                }

                if apply_benchmark_sentinels:
                    sentinel_details: dict[str, Any] = {}
                    interpreted_missing_count = 0

                    # B. Bank Marketing dataset: treat "unknown" as missing (NaN) where dataset semantics justify it (job, education)
                    for col_name, sentinels in schema.get("missing_sentinels", {}).items():
                        if col_name in df.columns:
                            raw_cnt = int(df[col_name].isin(sentinels).sum())
                            sentinel_details[col_name] = {
                                "sentinel_values": list(sentinels),
                                "raw_sentinel_count": raw_cnt,
                                "interpreted_as_missing": True,
                                "rationale": f"UCI documentation notes {sentinels} in demographic field '{col_name}' represents missing data.",
                            }
                            if raw_cnt > 0:
                                df[col_name] = df[col_name].replace({s: np.nan for s in sentinels})
                                interpreted_missing_count += raw_cnt

                    # Retained domain categories (e.g. contact, poutcome) - explicitly documented as preserved
                    retained_details: dict[str, Any] = {}
                    for col_name, sentinels in schema.get("retained_sentinels", {}).items():
                        if col_name in df.columns:
                            raw_cnt = int(df[col_name].isin(sentinels).sum())
                            retained_details[col_name] = {
                                "sentinel_values": list(sentinels),
                                "raw_sentinel_count": raw_cnt,
                                "interpreted_as_missing": False,
                                "rationale": "Retained as distinct domain category per UCI documentation.",
                            }

                    # Numeric sentinels (e.g. pdays = -1) - explicitly documented as preserved
                    numeric_details: dict[str, Any] = {}
                    if "numeric_sentinels" in schema:
                        df.attrs["sentinels"] = schema["numeric_sentinels"]
                        for col_name, s_info in schema["numeric_sentinels"].items():
                            if col_name in df.columns:
                                s_val = s_info.get("sentinel_value", -1)
                                s_cnt = int((df[col_name] == s_val).sum())
                                s_pct = round((s_cnt / len(df)) * 100, 2) if len(df) > 0 else 0.0
                                numeric_details[col_name] = {
                                    "sentinel_value": s_val,
                                    "count": s_cnt,
                                    "percentage": s_pct,
                                    "interpreted_as_missing": False,
                                    "rationale": s_info.get("description", "Numeric sentinel preserved."),
                                }

                    missing_provenance["sentinel_interpretation"] = sentinel_details
                    missing_provenance["retained_domain_categories"] = retained_details
                    missing_provenance["numeric_sentinels"] = numeric_details
                    missing_provenance["summary"] = {
                        "raw_missing_cells": int(df.isna().sum().sum()) - interpreted_missing_count,
                        "interpreted_missing_cells": interpreted_missing_count,
                        "total_missing_cells_after": int(df.isna().sum().sum()),
                    }
                else:
                    # Raw mode: no sentinels replaced
                    missing_provenance["mode"] = "raw_unpreprocessed"
                    missing_provenance["summary"] = {
                        "raw_missing_cells": int(df.isna().sum().sum()),
                        "interpreted_missing_cells": 0,
                        "total_missing_cells_after": int(df.isna().sum().sum()),
                    }

                # Attach provenance to df.attrs without mutating column structure
                df.attrs["missing_value_provenance"] = missing_provenance
                df.attrs["preprocessing_rules"] = missing_provenance
                if "derived_columns" in schema:
                    df.attrs["derived_features_available"] = list(schema["derived_columns"].keys())

            df.attrs["dataset_name"] = clean_filename
            return df, None

        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return None, "The file appears to be empty or contains only whitespace."
        except pd.errors.ParserError as exc:
            return None, f"Could not parse the CSV: {exc}"
        except Exception:
            # Try next encoding before giving up.
            continue

    return None, (
        "Unable to decode the file. Please save it as UTF-8 or Latin-1 encoded CSV and retry."
    )
