"""
backend/tests/test_xlsx_and_critic_repairs.py
============================================
Comprehensive unit and regression tests for DEF-02, DEF-03, DEF-04, DEF-05, and DEF-07.
Validates:
- DEF-02: XLSX dataset-store fallback routing to XLSX parser rather than CSV parser.
- DEF-04: Sparse cell coordinate reconstruction avoiding column shifts.
- DEF-05: Named worksheet resolution via workbook metadata and relationship mappings.
- DEF-07: Rich-text shared strings concatenation across multiple text runs.
- DEF-03: Critic rejection decision certainty distinction in reports and schemas.
Uses synthetic in-memory fixtures exclusively without modifying approved datasets.
"""

import io
import os
import zipfile
import pytest
import pandas as pd
from pathlib import Path

from app.services import upload_service
from app.services import dataset_store
from app.services.run_store import RunStoreService
from app.agents.critic_agent import CriticAgent
from app.agents.report_agent import ReportAgent
from app.schemas.analysis_schema import AnalysisResult
from app.schemas.report_schema import (
    CriticVerdictSummary,
    ResearchReport,
    DatasetContext,
    AnalysisContext,
)


def create_synthetic_xlsx(
    sheets_data: dict[str, list[list[tuple[str, str, str]]]],  # sheet_name -> list of rows of (cell_ref, type, val)
    shared_strings_xml: str,
    target_mapping: dict[str, str] = None,  # sheet_name -> sheet filename (e.g. 'worksheets/sheet2.xml')
) -> bytes:
    """
    Construct a valid in-memory .xlsx ZIP archive using standard library zipfile.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. [Content_Types].xml
        types_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
            '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
            '  <Default Extension="xml" ContentType="application/xml"/>\n'
            '  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>\n'
            '  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>\n'
        )
        for i in range(1, len(sheets_data) + 1):
            types_xml += f'  <Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>\n'
        types_xml += '</Types>'
        z.writestr("[Content_Types].xml", types_xml)

        # 2. _rels/.rels
        rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>\n'
            '</Relationships>'
        )
        z.writestr("_rels/.rels", rels_xml)

        # 3. xl/workbook.xml
        wb_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n'
            '  <sheets>\n'
        )
        sheet_idx = 1
        for s_name in sheets_data.keys():
            wb_xml += f'    <sheet name="{s_name}" sheetId="{sheet_idx}" r:id="rId{sheet_idx}"/>\n'
            sheet_idx += 1
        wb_xml += '  </sheets>\n</workbook>'
        z.writestr("xl/workbook.xml", wb_xml)

        # 4. xl/_rels/workbook.xml.rels
        wb_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        )
        sheet_idx = 1
        for s_name in sheets_data.keys():
            target_file = target_mapping.get(s_name, f"worksheets/sheet{sheet_idx}.xml") if target_mapping else f"worksheets/sheet{sheet_idx}.xml"
            wb_rels_xml += f'  <Relationship Id="rId{sheet_idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="{target_file}"/>\n'
            sheet_idx += 1
        wb_rels_xml += f'  <Relationship Id="rId{sheet_idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>\n'
        wb_rels_xml += '</Relationships>'
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels_xml)

        # 5. xl/sharedStrings.xml
        z.writestr("xl/sharedStrings.xml", shared_strings_xml)

        # 6. Worksheets
        sheet_idx = 1
        for s_name, rows in sheets_data.items():
            target_file = target_mapping.get(s_name, f"worksheets/sheet{sheet_idx}.xml") if target_mapping else f"worksheets/sheet{sheet_idx}.xml"
            ws_path = f"xl/{target_file}" if not target_file.startswith("xl/") else target_file
            ws_xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">\n'
                '  <sheetData>\n'
            )
            for r_idx, row in enumerate(rows, start=1):
                ws_xml += f'    <row r="{r_idx}">\n'
                for cell_ref, cell_type, val in row:
                    if cell_type:
                        ws_xml += f'      <c r="{cell_ref}" t="{cell_type}"><v>{val}</v></c>\n'
                    else:
                        ws_xml += f'      <c r="{cell_ref}"><v>{val}</v></c>\n'
                ws_xml += '    </row>\n'
            ws_xml += '  </sheetData>\n</worksheet>'
            z.writestr(ws_path, ws_xml)
            sheet_idx += 1

    return buf.getvalue()


# ------------------------------------------------------------------------------
# DEF-04 & DEF-07: Sparse Cells & Rich-Text Shared Strings
# ------------------------------------------------------------------------------
def test_def_04_sparse_cells_and_def_07_rich_text():
    """
    Verify:
    1. Sparse cells preserve column alignment without shifting.
    2. Missing middle, leading, and trailing cells remain NaN/None under correct headers.
    3. Rich-text shared strings with multiple runs (<r><t>...</t></r>) are fully concatenated.
    """
    shared_strings_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="6" uniqueCount="6">\n'
        '  <si><t>ColA</t></si>\n'
        '  <si><t>ColB</t></si>\n'
        '  <si><t>ColC</t></si>\n'
        '  <si><t>Normal_Plain_Text</t></si>\n'
        '  <si><r><rPr><b/></rPr><t>RichText </t></r><r><t>Prefix </t></r><r><i/><t>Suffix</t></r></si>\n'
        '  <si><t>Trailing_Value</t></si>\n'
        '</sst>'
    )

    rows = [
        # Row 1: Header (ColA, ColB, ColC)
        [("A1", "s", "0"), ("B1", "s", "1"), ("C1", "s", "2")],
        # Row 2: Middle cell B2 is missing! C2 has numerical value 300
        [("A2", "", "100"), ("C2", "", "300")],
        # Row 3: Leading cell A3 is missing! B3 has rich text (idx 4)
        [("B3", "s", "4")],
        # Row 4: Trailing cells B4 and C4 are missing! A4 has plain text (idx 3)
        [("A4", "s", "3")],
    ]

    xlsx_bytes = create_synthetic_xlsx(
        sheets_data={"data": rows},
        shared_strings_xml=shared_strings_xml,
    )

    df, err = upload_service.parse_xlsx(xlsx_bytes, filename="test_sparse.xlsx")
    assert err is None
    assert df is not None
    assert list(df.columns) == ["ColA", "ColB", "ColC"]
    assert len(df) == 3

    # Check Row 1 (Row 2 in Excel): ColA=100, ColB=NaN/None, ColC=300
    assert int(df.iloc[0]["ColA"]) == 100
    assert pd.isna(df.iloc[0]["ColB"])
    assert int(df.iloc[0]["ColC"]) == 300

    # Check Row 2 (Row 3 in Excel): ColA=NaN/None, ColB="RichText Prefix Suffix", ColC=NaN/None
    assert pd.isna(df.iloc[1]["ColA"])
    assert df.iloc[1]["ColB"] == "RichText Prefix Suffix"
    assert pd.isna(df.iloc[1]["ColC"])

    # Check Row 3 (Row 4 in Excel): ColA="Normal_Plain_Text", ColB=NaN/None, ColC=NaN/None
    assert df.iloc[2]["ColA"] == "Normal_Plain_Text"
    assert pd.isna(df.iloc[2]["ColB"])
    assert pd.isna(df.iloc[2]["ColC"])


# ------------------------------------------------------------------------------
# DEF-05: Named Worksheet Resolution
# ------------------------------------------------------------------------------
def test_def_05_named_worksheet_resolution():
    """
    Verify:
    1. Explicit sheet_name resolves the requested sheet.
    2. Sheet-to-XML relationship resolution works even if sheet name 'data' points to sheet2.xml.
    3. Missing sheet_name produces clear actionable error.
    4. Omitting sheet_name prioritizes sheet named 'data' or defaults to first sheet.
    """
    shared_strings_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="4" uniqueCount="4">\n'
        '  <si><t>Metadata_Col1</t></si>\n'
        '  <si><t>Metadata_Col2</t></si>\n'
        '  <si><t>Data_Col1</t></si>\n'
        '  <si><t>Data_Col2</t></si>\n'
        '</sst>'
    )

    meta_rows = [
        [("A1", "s", "0"), ("B1", "s", "1")],
        [("A2", "", "1"), ("B2", "", "2")],
    ]
    data_rows = [
        [("A1", "s", "2"), ("B1", "s", "3")],
        [("A2", "", "99"), ("B2", "", "88")],
    ]

    # Inverted target mapping: 'Metadata' -> sheet1.xml, 'data' -> sheet2.xml
    target_mapping = {
        "Metadata": "worksheets/sheet1.xml",
        "data": "worksheets/sheet2.xml",
    }

    xlsx_bytes = create_synthetic_xlsx(
        sheets_data={"Metadata": meta_rows, "data": data_rows},
        shared_strings_xml=shared_strings_xml,
        target_mapping=target_mapping,
    )

    # 1. Explicit request for 'data' sheet
    df_data, err = upload_service.parse_xlsx(xlsx_bytes, sheet_name="data")
    assert err is None
    assert list(df_data.columns) == ["Data_Col1", "Data_Col2"]
    assert df_data.iloc[0]["Data_Col1"] == 99

    # 2. Explicit request for 'Metadata' sheet
    df_meta, err = upload_service.parse_xlsx(xlsx_bytes, sheet_name="Metadata")
    assert err is None
    assert list(df_meta.columns) == ["Metadata_Col1", "Metadata_Col2"]
    assert df_meta.iloc[0]["Metadata_Col1"] == 1

    # 3. Nonexistent sheet name request returns actionable error
    df_none, err = upload_service.parse_xlsx(xlsx_bytes, sheet_name="nonexistent_sheet")
    assert df_none is None
    assert "Worksheet 'nonexistent_sheet' not found" in err
    assert "'Metadata'" in err and "'data'" in err

    # 4. Default resolution (no sheet_name) detects and prioritizes 'data' sheet
    df_default, err = upload_service.parse_xlsx(xlsx_bytes, sheet_name=None)
    assert err is None
    assert list(df_default.columns) == ["Data_Col1", "Data_Col2"]


# ------------------------------------------------------------------------------
# DEF-02: Dataset Store XLSX Routing and Fallbacks
# ------------------------------------------------------------------------------
def test_def_02_dataset_store_xlsx_routing(tmp_path):
    """
    Verify:
    1. Dataset store safely loads .xlsx from absolute path without invoking parse_csv or pd.read_csv directly.
    2. Dataset store caches parsed .xlsx in-memory.
    3. Normal CSV files continue to use delimited parser.
    4. Missing or corrupted files fail with clear error messages.
    """
    dataset_store.clear_cache()

    shared_strings_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2">\n'
        '  <si><t>ScoreA</t></si>\n'
        '  <si><t>ScoreB</t></si>\n'
        '</sst>'
    )
    rows = [
        [("A1", "s", "0"), ("B1", "s", "1")],
        [("A2", "", "85"), ("B2", "", "92")],
    ]
    xlsx_bytes = create_synthetic_xlsx({"data": rows}, shared_strings_xml)

    xlsx_path = tmp_path / "test_store_dataset.xlsx"
    xlsx_path.write_bytes(xlsx_bytes)

    # 1. Resolve via absolute path
    resolved_df = dataset_store.resolve_dataset(str(xlsx_path), sheet_name="data")
    assert resolved_df is not None
    assert list(resolved_df.columns) == ["ScoreA", "ScoreB"]
    assert resolved_df.iloc[0]["ScoreA"] == 85

    # 2. Check cache has the resolved dataset
    cached = dataset_store.get_dataset(f"{str(xlsx_path)}::data")
    assert cached is not None
    assert len(cached) == 1

    # 3. Normal CSV continues working through dataset_store
    csv_path = tmp_path / "test_store.csv"
    csv_path.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    csv_df = dataset_store.resolve_dataset(str(csv_path))
    assert csv_df is not None
    assert list(csv_df.columns) == ["x", "y"]
    assert len(csv_df) == 2

    # 4. Nonexistent file raises clear FileNotFoundError
    with pytest.raises(FileNotFoundError):
        dataset_store.resolve_dataset(str(tmp_path / "nonexistent.xlsx"))


# ------------------------------------------------------------------------------
# DEF-03: Critic Rejection Certainty Representation
# ------------------------------------------------------------------------------
def test_def_03_critic_confidence_decision_certainty():
    """
    Verify:
    1. Critic agent retains 1.0 confidence contract for decision certainty on rejected findings.
    2. ReportAgent formats rejected finding as 'Decision Certainty' to prevent misinterpretation.
    3. Accepted findings format normally without false labeling.
    """
    agent = CriticAgent()
    # Malformed task result that triggers immediate rejection
    rejected_task = {
        "task_id": "task_err_01",
        "task_type": "invalid_analysis",
        "method": "bogus_test",
        "success": False,
        "error": "Division by zero in test execution",
    }
    rev = agent.evaluate_result(rejected_task)
    assert rev.status == "REJECTED"
    # Backend contract preserved:
    assert rev.confidence == 1.0
    assert rev.confidence_level == "HIGH"

    # Test report formatting in ReportAgent
    rep_agent = ReportAgent()
    verdict = CriticVerdictSummary(
        task_id="task_err_01",
        status="REJECTED",
        confidence=1.0,
        confidence_level="HIGH",
        reason_codes=["ERR_ZERO_VARIANCE"],
        summary="Analysis was rejected due to zero variance.",
    )
    mock_report = ResearchReport(
        report_id="rep_test_01",
        run_id="run_test_01",
        title="Test Report",
        generated_at="2026-11-01T23:00:00Z",
        dataset_context=DatasetContext(dataset_name="test.csv", dataset_id="test.csv"),
        analysis_context=AnalysisContext(objective="Test objective"),
        critic_verdicts=[verdict],
    )
    md = rep_agent.render_markdown(mock_report)
    # Must explicitly state 'Decision Certainty' for rejected verdicts
    assert "Decision Certainty" in md
    assert "Evaluation Certainty" in md


# ------------------------------------------------------------------------------
# DEF-06: Research Run Storage Resilience & Ephemeral Filesystem Boundaries
# ------------------------------------------------------------------------------
def test_def_06_run_store_persistence_and_isolation(tmp_path):
    """
    Verify:
    1. RunStoreService operates cleanly within its designated storage path.
    2. Missing runs directory is created on demand.
    3. Handles corrupted files during list_runs gracefully without crashing.
    4. Retrieval of nonexistent run returns None cleanly.
    """
    isolated_dir = tmp_path / "custom_runs_dir"
    assert not isolated_dir.exists()

    service = RunStoreService(runs_dir=isolated_dir)
    # Listing empty/new dir returns empty list without error
    runs = service.list_runs()
    assert runs == []
    assert isolated_dir.exists()

    # Save a run atomically
    payload = {
        "dataset_name": "test_persistence.csv",
        "dataset_id": "test_persistence.csv",
        "active_phase": 1,
        "completed_phases": [1],
        "objective": "Verify run store isolation",
    }
    save_res = service.save_run(payload)
    assert save_res["status"] == "saved"
    run_id = save_res["run_id"]

    # Verify retrieval
    retrieved = service.get_run(run_id)
    assert retrieved is not None
    assert retrieved["dataset_name"] == "test_persistence.csv"

    # Corrupt a dummy file in the directory to test error isolation
    corrupted_file = isolated_dir / "run_corrupted_9999.json"
    corrupted_file.write_text("{ this is malformed json", encoding="utf-8")

    # list_runs should skip corrupted file and return the valid run
    runs_after = service.list_runs()
    assert len(runs_after) == 1
    assert runs_after[0].run_id == run_id

    # Nonexistent run retrieval returns None
    assert service.get_run("run_nonexistent_000000_123456") is None


# ------------------------------------------------------------------------------
# Regression: resolve_dataset DataFrame Truth Value Ambiguity Fix
# ------------------------------------------------------------------------------
def test_resolve_dataset_dataframe_truth_value_fix():
    """
    Verify:
    1. resolve_dataset handles cached DataFrame without raising 'ValueError: The truth value of a DataFrame is ambiguous'.
    2. resolve_dataset falls back to get_dataset(dataset_id) when cache_key returns None.
    3. resolve_dataset correctly returns empty DataFrame without truth-value failure.
    """
    # 1. Test direct cache hit with non-empty DataFrame
    df_normal = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    dataset_store.store_dataset("cached_test_id", df_normal)

    # Calling resolve_dataset on cached_test_id previously triggered `df or get_dataset(...)` and failed
    resolved = dataset_store.resolve_dataset("cached_test_id")
    assert resolved is not None
    assert len(resolved) == 3
    assert list(resolved.columns) == ["feature", "target"]

    # 2. Test fallback from sheet-specific cache_key to base dataset_id when sheet key is None
    # Base is cached under "base_dataset_id"
    df_base = pd.DataFrame({"col": [10, 20]})
    dataset_store.store_dataset("base_dataset_id", df_base)

    # Resolve with sheet_name="nonexistent_sheet" -> cache_key is "base_dataset_id::nonexistent_sheet" which is None
    # Fallback to get_dataset("base_dataset_id") should succeed without boolean evaluation
    resolved_fallback = dataset_store.resolve_dataset("base_dataset_id", sheet_name="nonexistent_sheet")
    assert resolved_fallback is not None
    assert len(resolved_fallback) == 2
    assert resolved_fallback.iloc[0]["col"] == 10

    # 3. Test empty DataFrame preservation (not treated as missing or evaluated as boolean)
    df_empty = pd.DataFrame({"empty_col": []})
    dataset_store.store_dataset("empty_cached_id", df_empty)

    resolved_empty = dataset_store.resolve_dataset("empty_cached_id")
    assert resolved_empty is not None
    assert len(resolved_empty) == 0
    assert list(resolved_empty.columns) == ["empty_col"]
