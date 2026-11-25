"""Synthetic portable PIT selection and lineage checks, without FDIC data."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bft65b.panel import compare_pit, prediction
from bft65b.panel.prediction import (
    _portable_chain_contract, _verified_raw_paths, run, select_sources,
)
from bft65b.quality.audit import _sha


SOURCES = {"raw": "source_a", "treatment": "treated_b",
           "panel": "panel_c", "repair": "repair_d"}


def _chain():
    audit = SimpleNamespace(run_id="p2_1_audit_a", status="complete",
        config={"source_attempt": "source_a", "source_vintage_bundle_sha256": "bundle"})
    treatment = SimpleNamespace(config={"source_audit_run": "audit_a",
        "source_audit_findings_sha256": "audit-hash", "source_vintage_bundle_sha256": "bundle"})
    panel = SimpleNamespace(config={"source_treatment_run": "treated_b",
        "source_treatment_summary_sha256": "treatment-hash",
        "source_vintage_bundle_sha256": "bundle"})
    repair = SimpleNamespace(config={"source_panel_run": "panel_c",
        "source_treatment_run": "treated_b", "source_panel_summary_sha256": "panel-hash",
        "source_vintage_bundle_sha256": "bundle"})
    return [audit, treatment, panel, repair,
            {"source_panel_run": "panel_c"}, {"protected_hashes": {"panel": "hash"}}]


def _validate(parts):
    _portable_chain_contract(SOURCES, "bundle", *parts, {"panel": "hash"},
        audit_findings_sha="audit-hash", treatment_summary_sha="treatment-hash",
        panel_summary_sha="panel-hash")


def test_portable_requires_every_explicit_upstream_and_no_accepted_fallback(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/source_vintage_bundle.json").write_text("{}")
    with pytest.raises(ValueError, match="all four explicit"):
        select_sources(tmp_path, **{k + "_run": v for k, v in SOURCES.items() if k != "repair"})
    with pytest.raises(ValueError, match="all four explicit"):
        run(tmp_path, "new_pit")
    assert not (tmp_path / "runs/task_2_5_prediction").exists()
    assert select_sources(tmp_path, **{k + "_run": v for k, v in SOURCES.items()}) == SOURCES
    with pytest.raises(ValueError, match="Unsafe"):
        select_sources(tmp_path, **{k + "_run": "../old" if k == "panel" else v
                                    for k, v in SOURCES.items()})


def test_historical_selection_keeps_original_cli_default(tmp_path):
    assert select_sources(tmp_path)["panel"] == "panel20261115b"


@pytest.mark.parametrize("index,key,value", [
    (0, "source_attempt", "different_vintage"),
    (1, "source_audit_findings_sha256", "stale"),
    (2, "source_treatment_run", "other_treatment"),
    (2, "source_vintage_bundle_sha256", "different_vintage"),
    (3, "source_panel_run", "other_panel"),
    (3, "source_treatment_run", "other_treatment"),
    (3, "source_panel_summary_sha256", "stale"),
])
def test_cross_run_or_cross_vintage_chain_is_rejected(index, key, value):
    parts = _chain()
    _validate(parts)
    broken = deepcopy(parts)
    broken[index].config[key] = value
    with pytest.raises(ValueError, match="lineage differs"):
        _validate(broken)


def test_repair_summary_and_panel_hash_binding_are_required():
    parts = _chain()
    _validate(parts)
    parts[-1]["protected_hashes"] = {"panel": "old"}
    with pytest.raises(ValueError, match="lineage differs"):
        _validate(parts)


def test_relocated_raw_paths_reject_original_absolute_location(tmp_path):
    portable = tmp_path / "portable"
    original = tmp_path / "original" / "raw.parquet"
    mapped = portable / "data_raw" / "raw.parquet"
    original.parent.mkdir()
    mapped.parent.mkdir(parents=True)
    original.write_bytes(b"original remains available")
    mapped.write_bytes(b"mapped vintage")
    raw = SimpleNamespace(status="complete", validations=[{"passed": True}],
        outputs=[{"path": str(mapped), "sha256": _sha(mapped)} for _ in range(172)])
    assert _verified_raw_paths(portable.resolve(), raw, portable=True) == {str(mapped): _sha(mapped)}
    raw.outputs[0] = {"path": str(original), "sha256": _sha(original)}
    with pytest.raises(ValueError, match="escapes the portable source root"):
        _verified_raw_paths(portable.resolve(), raw, portable=True)


def test_full_column_keyed_comparison_detects_value_change(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_pit, "verify_saved", lambda *_: {"status": "verified"})
    roots = [tmp_path / "left", tmp_path / "right"]
    for root, value in zip(roots, [10.0, 10.0]):
        data = root / "candidate.parquet"
        root.mkdir()
        pq.write_table(pa.table({"CERT": [2, 1], "Q": ["2023-06-30"] * 2,
                                 "ASSET": [value, 5.0], "AUDIT": ["a", "b"]}), data)
        evidence = root / "runs/task_2_5_prediction/sample"
        evidence.mkdir(parents=True)
        (evidence / "summary.json").write_text(__import__("json").dumps({
            "outputs": [{"quarter": "20230630", "path": str(data),
                         "sha256": _sha(data)}],
            "overall": {"candidates": 2}, "by_quarter": [{"quarter": "20230630"}],
            "future_perturbation": {"result": "identical"}}))
    equal = compare_pit.compare(roots[0], "sample", roots[1], "sample")
    assert equal["status"] == "PASS" and equal["keyed_rows_compared"] == 2
    right = roots[1] / "candidate.parquet"
    pq.write_table(pa.table({"CERT": [1, 2], "Q": ["2023-06-30"] * 2,
                             "ASSET": [5.0, 10.0], "AUDIT": ["b", "changed"]}), right)
    changed = compare_pit.compare(roots[0], "sample", roots[1], "sample")
    assert changed["status"] == "FAIL"
    assert changed["column_or_evidence_mismatches"] == {"AUDIT": 1}


def test_saved_nondefault_run_rejects_changed_recorded_upstream(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/source_vintage_bundle.json").write_text("{}")
    manifest = SimpleNamespace(run_id="portable_pit", status="complete",
        config={f"source_{key}_run": value for key, value in SOURCES.items()}
            | {"source_vintage_bundle_sha256": "digest"},
        validations=[{"passed": True}], outputs=[{}] * 166, inputs=[])
    summary = {"run_id": "portable_pit", "status": "complete",
        "source_runs": dict(SOURCES), "outputs": [{}] * 166,
        "by_quarter": [{}] * 166}
    metadata = {"status": "complete", "summary_sha256": "digest",
                "report_sha256": "digest", "upstream_hashes": {}}
    monkeypatch.setattr(prediction.RunManifest, "read", lambda *_: manifest)
    monkeypatch.setattr(prediction, "_read_json", lambda path: summary if path.name == "summary.json" else metadata)
    monkeypatch.setattr(prediction, "_sha", lambda *_: "digest")
    manifest.config["source_panel_run"] = "different_panel"
    with pytest.raises(ValueError, match="upstream run selection differs"):
        prediction.verify_saved(tmp_path, "portable_pit", check_upstream=False)
