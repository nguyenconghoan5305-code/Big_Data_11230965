"""Portable source-vintage and explicit repair-upstream contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from bft65b.manifest import RunManifest
from bft65b.panel import repair
from bft65b.panel.build import _config_matches
from bft65b.panel.compare_repair import compare
from bft65b.quality import source_vintage as vintage
from bft65b.quality import treatment


def _write(path: Path, contents: bytes = b"synthetic") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return hashlib.sha256(contents).hexdigest()


def _bundle(root: Path) -> dict:
    evidence = [
        "config/financials_full_v2.json",
        "runs/task_1_5_full_real20261111b.json",
        "runs/task_1_5_full_real20261111b_summary.json",
        "runs/task_1_5_full_real20261111b_acceptance.json",
        "runs/task_1_5_full_real20261111b_summary_unlinked_ids.json",
        *(path.as_posix() for path in vintage.SUPPLEMENTARY),
    ]
    files = []
    for relative in evidence + [f"data_raw/financials/{n:03d}.parquet" for n in range(172)]:
        digest = _write(root / relative)
        files.append({"relative_path": relative, "original_path": f"X:/captured/{relative}",
                      "sha256": digest, "bytes": 9})
    bundle = {"version": "source_vintage_v1", "attempt": vintage.ATTEMPT,
              "files": files,
              "source_manifest_sha256": vintage._sha(root / evidence[1]),
              "source_summary_sha256": vintage._sha(root / evidence[2])}
    path = root / vintage.BUNDLE_FILE
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return bundle


def test_bundle_inventory_rejects_mutation_missing_file_and_unsafe_path(tmp_path):
    bundle = _bundle(tmp_path)
    assert len(vintage.inventory(tmp_path)["files"]) == 180
    target = tmp_path / bundle["files"][8]["relative_path"]
    target.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum/size"):
        vintage.inventory(tmp_path)
    _write(target)
    target.unlink()
    with pytest.raises(ValueError, match="checksum/size"):
        vintage.inventory(tmp_path)
    _write(target)
    bundle["files"][8]["relative_path"] = "../outside.parquet"
    (tmp_path / vintage.BUNDLE_FILE).write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe"):
        vintage.inventory(tmp_path)


def test_relocation_remaps_only_memory_and_rejects_mismatched_lineage(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path)
    old_fin = bundle["files"][8]["original_path"]
    old_fail = bundle["files"][9]["original_path"]
    old_link = next(item["original_path"] for item in bundle["files"]
                    if item["relative_path"].endswith("_summary_unlinked_ids.json"))
    manifest = RunManifest("source", {}, status="complete", outputs=[
        {"dataset": "financials", "partition": "REPDTE=19840331", "path": old_fin},
        {"dataset": "failures", "partition": None, "path": old_fail},
    ])
    summary = {"small_tables": [{"endpoint": "failures", "quarter": None,
                "unresolved_linkage": {"source_ids_path": old_link}}]}
    fin = tmp_path / bundle["files"][8]["relative_path"]
    fail = tmp_path / bundle["files"][9]["relative_path"]
    store = SimpleNamespace(config=object(), target_path=lambda endpoint, _: fin if endpoint == "financials" else fail)
    monkeypatch.setattr(vintage, "full_plan", lambda _: SimpleNamespace(
        jobs=lambda _: [("financials", "19840331", (), 1), ("failures", None, (), 1)]))
    mapped, relocated = vintage.relocated_view(tmp_path, manifest, summary, store)
    assert mapped.outputs[0]["path"] == str(fin) and mapped.outputs[1]["path"] == str(fail)
    assert relocated["small_tables"][0]["unresolved_linkage"]["source_ids_path"].startswith(str(tmp_path))
    assert manifest.outputs[0]["path"] == old_fin and summary["small_tables"][0]["unresolved_linkage"]["source_ids_path"] == old_link
    manifest.outputs[0]["path"] = "X:/unexpected.parquet"
    with pytest.raises(ValueError, match="mapping differs"):
        vintage.relocated_view(tmp_path, manifest, summary, store)


def test_repair_cli_explicit_selection_and_default(monkeypatch, tmp_path, capsys):
    selected = []
    monkeypatch.setattr(repair, "run", lambda root, source, run_id, panel_run=repair.PREVIOUS_RUN:
                        selected.append(panel_run) or {"status": "synthetic"})
    source = tmp_path / "source.json"
    repair.main(["run", "--project-root", str(tmp_path), "--source-json", str(source),
                 "--panel-run", "newpanel", "--run-id", "newrepair"])
    repair.main(["run", "--project-root", str(tmp_path), "--source-json", str(source),
                 "--run-id", "oldrepair"])
    assert selected == ["newpanel", None]
    assert repair._select_panel_run(tmp_path, None) == repair.PREVIOUS_RUN
    _write(tmp_path / vintage.BUNDLE_FILE)
    with pytest.raises(ValueError, match="explicit --panel-run"):
        repair._select_panel_run(tmp_path, None)
    assert capsys.readouterr().out.count('"status": "synthetic"') == 2


def test_repair_selected_panel_missing_does_not_fall_back(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        repair._accepted_panel(tmp_path, "newpanel")
    assert "newpanel" in str(error.value)
    with pytest.raises(ValueError, match="Unsafe panel run ID"):
        repair._accepted_panel(tmp_path, "../accepted")


def test_repair_show_rejects_wrong_upstream_binding_and_preserves_valid_output(tmp_path, monkeypatch, capsys):
    evidence = tmp_path / "runs/task_2_3_transaction_repair/synthetic"
    evidence.mkdir(parents=True)
    summary = {"status": "complete", "source_panel_run": "newpanel",
               "transaction_dispositions": {}, "new_categories": {}}
    report = evidence / "report.md"
    report.write_text("synthetic report", encoding="utf-8")
    (evidence / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (evidence / "metadata.json").write_text(json.dumps({
        "status": "complete", "summary_sha256": repair._sha(evidence / "summary.json"),
        "report_sha256": repair._sha(report), "protected_hashes": {}}), encoding="utf-8")
    manifest = RunManifest("p2_3_repair_synthetic", {"source_panel_run": "wrongpanel"})
    manifest.record_validation("synthetic", True)
    manifest.set_status("complete")
    manifest.write(evidence / "run_manifest.json")
    with pytest.raises(SystemExit) as failed:
        repair.main(["show", "--project-root", str(tmp_path), "--run-id", "synthetic"])
    assert failed.value.code == 2 and "incomplete or inconsistent" in capsys.readouterr().err
    manifest.config["source_panel_run"] = "newpanel"
    manifest.write(evidence / "run_manifest.json")
    monkeypatch.setattr(repair, "_accepted_panel", lambda root, run:
                        ({}, [], [], {}, {}, {}) if run == "newpanel" else (_ for _ in ()).throw(ValueError("wrong run")))
    repair.main(["show", "--project-root", str(tmp_path), "--run-id", "synthetic"])
    assert '"status": "complete"' in capsys.readouterr().out


def test_treatment_rejects_audit_with_different_bundle_lineage(tmp_path):
    config = tmp_path / "config/financials_full_v2.json"
    config.parent.mkdir(parents=True)
    config.write_bytes((Path(__file__).parents[1] / "config/financials_full_v2.json").read_bytes())
    _write(tmp_path / vintage.BUNDLE_FILE)
    evidence = tmp_path / "runs/task_2_1_quality/synthetic_audit"
    evidence.mkdir(parents=True)
    findings = evidence / "findings.json"
    findings.write_text(json.dumps({"status": "complete", "accepted_attempt": vintage.ATTEMPT}), encoding="utf-8")
    (evidence / "metadata.json").write_text(json.dumps({
        "status": "complete", "input_hashes_unchanged": True,
        "findings_sha256": vintage._sha(findings)}), encoding="utf-8")
    manifest = RunManifest("p2_1_synthetic_audit", {}, status="running")
    for n in range(172):
        manifest.record_input(f"synthetic-{n}", "0" * 64)
    manifest.record_validation("synthetic", True)
    manifest.set_status("complete")
    manifest.write(evidence / "run_manifest.json")
    with pytest.raises(ValueError, match="source-vintage lineage differs"):
        treatment.run_treatment(tmp_path, config, "synthetic_audit", "new_treatment")
    assert not (tmp_path / "data_processed/task_2_1_treatment/new_treatment").exists()


def test_json_manifest_config_with_lineage_keeps_verified_field_lists():
    from bft65b.config import ProjectConfig
    config = ProjectConfig()
    serialized = json.loads(json.dumps(config.to_dict()))
    serialized["source_vintage_bundle_sha256"] = "a" * 64
    assert _config_matches(serialized, config)
    serialized["financial_fields"] = ["CERT", "REPDTE"]
    assert not _config_matches(serialized, config)


def _repair_fixture(root: Path, run_id: str, *, reverse: bool = False, changed: bool = False) -> None:
    evidence = root / "runs/task_2_3_transaction_repair" / run_id
    evidence.mkdir(parents=True)
    manifest = RunManifest(f"p2_3_repair_{run_id}", {})
    outputs = []
    rows = {
        "failure_events": pa.table({"ID": ["a", "b"], "CERT": [1, 2],
                                    "RESTYPE": ["FAILURE", "ASSISTANCE"]}),
        "exit_assessments": pa.table({"CERT": [1, 2],
                                     "EXIT_CATEGORY": ["confirmed_failure", "ambiguous_exit"]}),
    }
    if changed:
        rows["exit_assessments"] = pa.table({"CERT": [1, 2],
            "EXIT_CATEGORY": ["ambiguous_exit", "ambiguous_exit"]})
    for dataset, table in rows.items():
        if reverse:
            table = table.take(pa.array([1, 0]))
        path = root / "data_processed/task_2_3_transaction_repair" / run_id / dataset / "part-000.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(table, path)
        digest = vintage._sha(path)
        outputs.append({"dataset": dataset, "path": str(path), "rows": table.num_rows, "sha256": digest})
        manifest.record_output(path=str(path), dataset=dataset, rows=table.num_rows,
                               sha256=digest, compression="ZSTD", schema_version="test",
                               partition=None, source_total=table.num_rows, source_query=None)
    summary = {"status": "complete", "outputs": outputs,
               "new_categories": {"confirmed_failure": 1 if not changed else 0,
                                  "ambiguous_exit": 1 if not changed else 2},
               "transaction_dispositions": {"verified_failure": 1, "assistance_transaction": 1}}
    (evidence / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (evidence / "metadata.json").write_text(json.dumps({"status": "complete",
        "summary_sha256": vintage._sha(evidence / "summary.json")}), encoding="utf-8")
    manifest.record_validation("synthetic", True)
    manifest.set_status("complete")
    manifest.write(evidence / "run_manifest.json")


def test_repaired_result_comparator_checks_keyed_values_not_only_counts(tmp_path):
    original, replay = tmp_path / "original", tmp_path / "relocated"
    _repair_fixture(original, "accepted")
    _repair_fixture(replay, "replayed", reverse=True)
    result = compare(original, "accepted", replay, "replayed")
    assert result["status"] == "PASS"
    assert all(item["keyed_values_equal"] for item in result["outputs"].values())
    changed_root = tmp_path / "changed"
    _repair_fixture(changed_root, "replayed", changed=True)
    failed = compare(original, "accepted", changed_root, "replayed")
    assert failed["status"] == "FAIL"
    assert not failed["outputs"]["exit_assessments"]["keyed_values_equal"]
