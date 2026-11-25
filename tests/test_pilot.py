"""Synthetic integration gates for the bounded FDIC acquisition pilot."""

import json

import pytest

from bft65b.acquisition.client import FDICClient, FDICContractError, FDICRequestError, Page
from bft65b.acquisition.pilot import PILOT_QUARTERS, _verify_completion, run_pilot
from bft65b.acquisition.storage import ParquetStore, StorageConflictError, StorageValidationError
from bft65b.config import ProjectConfig
from bft65b.manifest import RunManifest


def financial(quarter, cert=14, **changes):
    row = {
        "CERT": cert, "REPDTE": quarter, "STNAME": "MA", "ASSET": 100,
        "DEP": "90.0000", "EQ": 10, "NETINC": 1, "ROA": 0.5,
        "NCLNLS": None, "LNATRES": 2, "RBC1AAJ": 12.5,
    }
    row.update(changes)
    return row


def page(endpoint, rows, *, total=None, offset=0, limit=2):
    fields = {
        "financials": ProjectConfig().financial_fields,
        "failures": ProjectConfig().failure_fields,
        "institutions": ProjectConfig().institution_fields,
    }[endpoint]
    return Page(
        endpoint, offset, limit, len(rows) if total is None else total,
        tuple(rows), ("stable", "2026-01-01T00:00:00Z"),
        tuple(tuple(field for field in fields if field not in row) for row in rows),
        tuple(tuple(field for field in fields if field in row and row[field] is None) for row in rows),
    )


class StubClient:
    _row_key = FDICClient._row_key

    def __init__(self, config, pages):
        self.config = config
        self.pages = pages
        self.calls = []
        self.probes = []

    def iter_pages(self, endpoint, *, filters=None, limit=None, require_nonempty=False):
        self.calls.append((endpoint, filters, limit, require_nonempty))
        return iter(self.pages[(endpoint, filters)])

    def fetch_page(self, endpoint, *, filters=None, limit=None):
        self.probes.append((endpoint, filters, limit))
        original = self.pages[(endpoint, filters)][0]
        return Page(endpoint, 0, limit, original.total, original.rows[:1], original.index,
                    original.missing_fields[:1], original.null_fields[:1])


def context(tmp_path, pages):
    config = ProjectConfig(api_page_size=2)
    return (
        config, StubClient(config, pages), ParquetStore(tmp_path / "data_raw", config),
        tmp_path / "runs" / "pilot.json", tmp_path / "runs" / "summary.json",
    )


def good_pages():
    pages = {}
    for quarter in PILOT_QUARTERS:
        pages[("financials", f"REPDTE:{quarter}")] = [
            page("financials", [financial(quarter)], limit=2)
        ]
    pages[("failures", None)] = [page("failures", [
        {"ID": "1", "CERT": 40, "FAILDATE": "1/2/1984", "COST": 5},
        {"ID": "2", "CERT": "40", "FAILDATE": "1/3/1984", "COST": None},
        {"ID": "3", "CERT": None, "FAILDATE": "5/28/1934", "COST": None},
    ], limit=10000)]
    pages[("institutions", None)] = [
        page("institutions", [{"CERT": 4, "ACTIVE": 1}], total=3, limit=2),
        page("institutions", [{"CERT": 5, "ACTIVE": 0}, {"CERT": 6, "ACTIVE": 1}],
             total=3, offset=1, limit=2),
    ]
    return pages


def test_complete_pilot_records_source_provenance_and_verified_files(tmp_path):
    config, client, store, manifest_path, summary_path = context(tmp_path, good_pages())
    summary = run_pilot(config=config, client=client, store=store,
                        manifest_path=manifest_path, summary_path=summary_path)
    manifest = RunManifest.read(manifest_path)
    assert summary["status"] == manifest.status == "complete"
    assert len(summary["quarters"]) == 4 and len(summary["small_tables"]) == 2
    assert len(manifest.sources) == len(manifest.outputs) == 6
    assert json.loads(summary_path.read_text())["status"] == "complete"
    assert all(source["api_total"] == source["downloaded_rows"] for source in manifest.sources)
    assert all(row["completeness"] == row["read_back"] == "pass"
               for row in summary["quarters"] + summary["small_tables"])
    assert all(row["parquet_bytes"] > 0 and len(row["sha256"]) == 64
               for row in summary["quarters"] + summary["small_tables"])
    assert summary["quarters"][0]["null_fields"]["NCLNLS"] == 1
    assert summary["small_tables"][0]["duplicate_keys"] == 0  # Repeated CERT, distinct dates.
    audit = summary["small_tables"][0]["unresolved_linkage"]
    assert audit["count"] == 1
    assert json.loads((tmp_path / "runs/summary_unlinked_ids.json").read_text())["source_ids"] == ["3"]
    assert len(audit["sha256"]) == 64
    assert manifest.outputs[4]["schema_version"] == "2"
    assert client.calls[0] == ("financials", "REPDTE:20230331", 2, True)
    assert client.calls[-2] == ("failures", None, 10000, True)
    assert store.can_skip("financials", manifest_path, quarter="20230331",
                          source_total=1, source_query=summary["quarters"][0]["query"])


def test_count_mismatch_stops_before_any_parquet_promotion(tmp_path):
    pages = good_pages()
    pages[("financials", "REPDTE:20230331")] = [
        page("financials", [financial("20230331")], total=2)
    ]
    config, client, store, manifest_path, summary_path = context(tmp_path, pages)
    with pytest.raises(FDICContractError, match="count/key gate"):
        run_pilot(config=config, client=client, store=store,
                  manifest_path=manifest_path, summary_path=summary_path)
    assert RunManifest.read(manifest_path).status == "failed"
    assert json.loads(summary_path.read_text())["status"] == "failed"
    assert json.loads(summary_path.read_text())["failed_stage"] == "financials:20230331"
    assert not store.target_path("financials", "20230331").exists()


def test_duplicate_bank_quarter_stops_without_promotion(tmp_path):
    pages = good_pages()
    pages[("financials", "REPDTE:20230331")] = [
        page("financials", [financial("20230331"), financial("20230331", cert="014")])
    ]
    config, client, store, manifest_path, summary_path = context(tmp_path, pages)
    with pytest.raises(FDICContractError, match="duplicates=1"):
        run_pilot(config=config, client=client, store=store,
                  manifest_path=manifest_path, summary_path=summary_path)
    assert not store.target_path("financials", "20230331").exists()
    assert RunManifest.read(manifest_path).status == "failed"




def test_invalid_numeric_conversion_stops_without_promotion(tmp_path):
    pages = good_pages()
    pages[("financials", "REPDTE:20230331")] = [
        page("financials", [financial("20230331", ASSET="N/A")])
    ]
    config, client, store, manifest_path, summary_path = context(tmp_path, pages)
    with pytest.raises(StorageValidationError, match="decimal"):
        run_pilot(config=config, client=client, store=store,
                  manifest_path=manifest_path, summary_path=summary_path)
    assert not store.target_path("financials", "20230331").exists()
    assert RunManifest.read(manifest_path).status == "failed"


def test_failures_over_one_page_blocks_before_failure_file(tmp_path):
    pages = good_pages()
    pages[("failures", None)] = [
        page("failures", [{"ID": "1", "CERT": 40, "FAILDATE": "1/2/1984"}], total=10001, limit=10000)
    ]
    config, client, store, manifest_path, summary_path = context(tmp_path, pages)
    with pytest.raises(FDICContractError, match="one API page"):
        run_pilot(config=config, client=client, store=store,
                  manifest_path=manifest_path, summary_path=summary_path)
    assert not store.target_path("failures").exists()
    assert len(RunManifest.read(manifest_path).outputs) == 4
    assert RunManifest.read(manifest_path).status == "failed"


@pytest.mark.parametrize("bad_row,match", [
    ({"CERT": None, "FAILDATE": "5/28/1934"}, "source ID"),
    ({"ID": "1", "CERT": 40, "FAILDATE": "1/4/1984"}, "duplicates=1"),
])
def test_failure_source_identity_gate_stops_before_promotion(tmp_path, bad_row, match):
    pages = good_pages()
    pages[("failures", None)] = [page("failures", [
        {"ID": "1", "CERT": 40, "FAILDATE": "1/2/1984"}, bad_row,
    ], limit=10000)]
    config, client, store, manifest_path, summary_path = context(tmp_path, pages)
    with pytest.raises(FDICContractError, match=match):
        run_pilot(config=config, client=client, store=store,
                  manifest_path=manifest_path, summary_path=summary_path)
    assert not store.target_path("failures").exists()
    assert RunManifest.read(manifest_path).status == "failed"


def test_interrupted_acquisition_resumes_without_refetching_verified_quarter(tmp_path, monkeypatch):
    config, client, store, manifest_path, summary_path = context(tmp_path, good_pages())
    original = client.iter_pages

    def interrupt(endpoint, *, filters=None, limit=None, require_nonempty=False):
        if filters == "REPDTE:20230630":
            raise KeyboardInterrupt("synthetic interruption")
        return original(endpoint, filters=filters, limit=limit,
                        require_nonempty=require_nonempty)

    monkeypatch.setattr(client, "iter_pages", interrupt)
    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        run_pilot(config=config, client=client, store=store,
                  manifest_path=manifest_path, summary_path=summary_path)
    first = store.target_path("financials", "20230331")
    checksum, mtime = first.read_bytes(), first.stat().st_mtime_ns
    assert RunManifest.read(manifest_path).status == "running"
    assert not store.target_path("financials", "20230630").exists()

    monkeypatch.setattr(client, "iter_pages", original)
    client.calls.clear()
    summary = run_pilot(config=config, client=client, store=store,
                        manifest_path=manifest_path, summary_path=summary_path, resume=True)
    assert summary["status"] == "complete"
    assert summary["quarters"][0]["recovery_action"] == "skipped"
    assert client.probes[0] == ("financials", "REPDTE:20230331", 1)
    assert all(filters != "REPDTE:20230331" for _, filters, _, _ in client.calls)
    assert first.read_bytes() == checksum and first.stat().st_mtime_ns == mtime
    manifest = RunManifest.read(manifest_path)
    assert len(manifest.outputs) == 6 and len(summary["quarters"]) == 4
    assert len(summary["small_tables"]) == 2 and all(v["passed"] for v in manifest.validations)


def test_interruption_after_failures_skips_verified_v2_file_on_resume(tmp_path, monkeypatch):
    config, client, store, manifest_path, summary_path = context(tmp_path, good_pages())
    original = client.iter_pages

    def interrupt_institutions(endpoint, *, filters=None, limit=None, require_nonempty=False):
        if endpoint == "institutions":
            raise KeyboardInterrupt("after failures")
        return original(endpoint, filters=filters, limit=limit,
                        require_nonempty=require_nonempty)

    monkeypatch.setattr(client, "iter_pages", interrupt_institutions)
    with pytest.raises(KeyboardInterrupt, match="after failures"):
        run_pilot(config=config, client=client, store=store,
                  manifest_path=manifest_path, summary_path=summary_path)
    failure_file = store.target_path("failures")
    before, mtime = failure_file.read_bytes(), failure_file.stat().st_mtime_ns
    assert len(RunManifest.read(manifest_path).outputs) == 5
    assert RunManifest.read(manifest_path).status == "running"
    monkeypatch.setattr(client, "iter_pages", original)
    client.calls.clear()
    summary = run_pilot(config=config, client=client, store=store,
                        manifest_path=manifest_path, summary_path=summary_path, resume=True)
    assert summary["status"] == "complete"
    assert summary["small_tables"][0]["recovery_action"] == "skipped"
    assert ("failures", None, 1) in client.probes
    assert [endpoint for endpoint, _, _, _ in client.calls] == ["institutions"]
    assert failure_file.read_bytes() == before and failure_file.stat().st_mtime_ns == mtime
    assert len(RunManifest.read(manifest_path).outputs) == 6


@pytest.mark.parametrize("quarter,damage", [
    ("20230630", "truncate"), ("20230930", "missing"),
])
def test_resume_refetches_only_missing_or_corrupt_quarter(tmp_path, quarter, damage):
    config, client, store, manifest_path, summary_path = context(tmp_path, good_pages())
    first = run_pilot(config=config, client=client, store=store,
                      manifest_path=manifest_path, summary_path=summary_path)
    interrupted = RunManifest.read(manifest_path)
    interrupted.set_status("running")  # Synthetic interruption before final run commit.
    interrupted.write(manifest_path)
    intact = store.target_path("financials", "20230331")
    intact_bytes, intact_mtime = intact.read_bytes(), intact.stat().st_mtime_ns
    damaged = store.target_path("financials", quarter)
    if damage == "truncate":
        damaged.write_bytes(damaged.read_bytes()[:40])
    else:
        damaged.unlink()
    assert not store.can_skip("financials", manifest_path, quarter=quarter, source_total=1,
                              source_query=next(row["query"] for row in first["quarters"]
                                                if row["quarter"] == quarter))
    client.calls.clear()
    summary = run_pilot(config=config, client=client, store=store,
                        manifest_path=manifest_path, summary_path=summary_path, resume=True)
    assert summary["status"] == "complete"
    assert [filters for endpoint, filters, _, _ in client.calls] == [f"REPDTE:{quarter}"]
    assert intact.read_bytes() == intact_bytes and intact.stat().st_mtime_ns == intact_mtime
    assert store.validate_file(damaged, "financials", quarter=quarter, expected_rows=1).rows == 1
    if damage == "truncate":
        assert list(damaged.parent.glob("part-000.parquet.corrupt.*"))


@pytest.mark.parametrize("change", ["index", "total"])
def test_source_change_blocks_skip_without_touching_valid_raw(tmp_path, monkeypatch, change):
    config, client, store, manifest_path, summary_path = context(tmp_path, good_pages())
    run_pilot(config=config, client=client, store=store,
              manifest_path=manifest_path, summary_path=summary_path)
    first = store.target_path("financials", "20230331")
    before = first.read_bytes()
    original = client.fetch_page

    def changed(endpoint, *, filters=None, limit=None):
        page = original(endpoint, filters=filters, limit=limit)
        return Page(page.endpoint, page.offset, page.limit,
                    page.total + 1 if change == "total" else page.total, page.rows,
                    ("changed-index", page.index[1]) if change == "index" else page.index,
                    page.missing_fields, page.null_fields)

    monkeypatch.setattr(client, "fetch_page", changed)
    with pytest.raises(FDICContractError, match="Source total or index changed"):
        run_pilot(config=config, client=client, store=store,
                  manifest_path=manifest_path, summary_path=summary_path, resume=True)
    assert first.read_bytes() == before
    assert RunManifest.read(manifest_path).status == "complete"


def test_failed_manifest_requires_fresh_run_and_reconciles_valid_raw(tmp_path, monkeypatch):
    config, client, store, manifest_path, summary_path = context(tmp_path, good_pages())
    original = client.iter_pages

    def exhausted(endpoint, *, filters=None, limit=None, require_nonempty=False):
        if filters == "REPDTE:20230630":
            raise FDICRequestError("HTTP 503 after 4 attempts")
        return original(endpoint, filters=filters, limit=limit,
                        require_nonempty=require_nonempty)

    monkeypatch.setattr(client, "iter_pages", exhausted)
    with pytest.raises(FDICRequestError, match="4 attempts"):
        run_pilot(config=config, client=client, store=store,
                  manifest_path=manifest_path, summary_path=summary_path)
    first = store.target_path("financials", "20230331")
    before, mtime = first.read_bytes(), first.stat().st_mtime_ns
    assert RunManifest.read(manifest_path).status == "failed"
    with pytest.raises(StorageValidationError, match="Failed or pending"):
        run_pilot(config=config, client=client, store=store,
                  manifest_path=manifest_path, summary_path=summary_path, resume=True)
    monkeypatch.setattr(client, "iter_pages", original)
    new_manifest = tmp_path / "runs" / "fresh.json"
    new_summary = tmp_path / "runs" / "fresh_summary.json"
    result = run_pilot(config=config, client=client, store=store,
                       manifest_path=new_manifest, summary_path=new_summary)
    assert result["status"] == "complete"
    assert result["quarters"][0]["recovery_action"] == "reconciled"
    assert first.read_bytes() == before and first.stat().st_mtime_ns == mtime
    assert RunManifest.read(manifest_path).status == "failed"


def test_completed_manifest_and_raw_files_are_read_only_on_repeat(tmp_path):
    config, client, store, manifest_path, summary_path = context(tmp_path, good_pages())
    run_pilot(config=config, client=client, store=store,
              manifest_path=manifest_path, summary_path=summary_path)
    manifest_bytes = manifest_path.read_bytes()
    paths = [store.target_path("financials", quarter) for quarter in PILOT_QUARTERS]
    paths += [store.target_path("failures"), store.target_path("institutions")]
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    client.calls.clear()
    result = run_pilot(config=config, client=client, store=store,
                       manifest_path=manifest_path, summary_path=summary_path, resume=True)
    assert result["status"] == "complete" and len(client.probes) == 6
    assert client.calls == [] and manifest_path.read_bytes() == manifest_bytes
    assert before == [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]


@pytest.mark.parametrize("stale_field,value", [
    ("source_query", "stale-query"), ("source_total", 2),
    ("schema_version", "old-version"),
])
def test_stale_output_metadata_is_refetched_and_reconciled_without_rewrite(
    tmp_path, stale_field, value,
):
    config, client, store, manifest_path, summary_path = context(tmp_path, good_pages())
    run_pilot(config=config, client=client, store=store,
              manifest_path=manifest_path, summary_path=summary_path)
    manifest = RunManifest.read(manifest_path)
    manifest.set_status("running")
    manifest.outputs[0][stale_field] = value
    manifest.write(manifest_path)
    target = store.target_path("financials", "20230331")
    before, mtime = target.read_bytes(), target.stat().st_mtime_ns
    client.calls.clear()
    summary = run_pilot(config=config, client=client, store=store,
                        manifest_path=manifest_path, summary_path=summary_path, resume=True)
    assert summary["status"] == "complete"
    assert summary["quarters"][0]["recovery_action"] == "reconciled"
    assert [filters for _, filters, _, _ in client.calls] == ["REPDTE:20230331"]
    assert target.read_bytes() == before and target.stat().st_mtime_ns == mtime
    assert len(RunManifest.read(manifest_path).outputs) == 7


def test_inconsistent_manifest_checksum_stops_without_overwriting_raw(tmp_path):
    config, client, store, manifest_path, summary_path = context(tmp_path, good_pages())
    run_pilot(config=config, client=client, store=store,
              manifest_path=manifest_path, summary_path=summary_path)
    target = store.target_path("financials", "20230331")
    before = target.read_bytes()
    manifest = RunManifest.read(manifest_path)
    manifest.set_status("running")
    manifest.outputs[0]["sha256"] = "0" * 64
    manifest.write(manifest_path)
    with pytest.raises(StorageConflictError, match="checksum"):
        run_pilot(config=config, client=client, store=store,
                  manifest_path=manifest_path, summary_path=summary_path, resume=True)
    assert target.read_bytes() == before
    assert RunManifest.read(manifest_path).status == "failed"


def test_pilot_recovers_when_first_parquet_promotes_but_manifest_write_fails(
    tmp_path, monkeypatch,
):
    config, client, store, manifest_path, summary_path = context(tmp_path, good_pages())
    original_write = RunManifest.write
    failed = False

    def fail_once_after_promotion(self, path):
        nonlocal failed
        if not failed and len(self.outputs) == 1:
            failed = True
            raise OSError("manifest commit failed after promotion")
        return original_write(self, path)

    monkeypatch.setattr(RunManifest, "write", fail_once_after_promotion)
    with pytest.raises(OSError, match="after promotion"):
        run_pilot(config=config, client=client, store=store,
                  manifest_path=manifest_path, summary_path=summary_path)
    target = store.target_path("financials", "20230331")
    verified = store.validate_file(target, "financials", quarter="20230331", expected_rows=1)
    before, mtime = target.read_bytes(), target.stat().st_mtime_ns
    assert not store.can_skip("financials", manifest_path, quarter="20230331",
                              source_total=1, source_query=next(
                                  item["source_query"] for item in RunManifest.read(manifest_path).outputs
                                  if item["dataset"] == "financials"))
    new_manifest = tmp_path / "runs" / "after_commit_failure.json"
    new_summary = tmp_path / "runs" / "after_commit_failure_summary.json"
    summary = run_pilot(config=config, client=client, store=store,
                        manifest_path=new_manifest, summary_path=new_summary)
    assert summary["status"] == "complete"
    assert summary["quarters"][0]["recovery_action"] == "reconciled"
    assert target.read_bytes() == before and target.stat().st_mtime_ns == mtime
    assert store.validate_file(target, "financials", quarter="20230331",
                               expected_rows=1, expected_sha256=verified.sha256).rows == 1


def test_missing_institutions_cannot_complete_six_output_gate(tmp_path):
    pages = good_pages()
    pages[("institutions", None)] = []
    config, client, store, manifest_path, summary_path = context(tmp_path, pages)
    with pytest.raises(FDICContractError, match="No API pages"):
        run_pilot(config=config, client=client, store=store,
                  manifest_path=manifest_path, summary_path=summary_path)
    assert RunManifest.read(manifest_path).status == "failed"
    assert len(RunManifest.read(manifest_path).outputs) == 5
    assert not store.target_path("institutions").exists()


def test_completion_guard_rejects_missing_output_even_with_passing_validations(tmp_path):
    config, client, store, manifest_path, summary_path = context(tmp_path, good_pages())
    summary = run_pilot(config=config, client=client, store=store,
                        manifest_path=manifest_path, summary_path=summary_path)
    manifest = RunManifest.read(manifest_path)
    manifest.outputs.pop()
    with pytest.raises(StorageValidationError, match="Pilot output missing"):
        _verify_completion(config, store, manifest, summary)
