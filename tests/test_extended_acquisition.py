"""Synthetic Task 1.5A schema, bounded acquisition, and recovery gates."""

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bft65b.acquisition.client import FDICClient, FDICContractError
from bft65b.acquisition.extended import (
    AcquisitionPlan, EXTENDED_PILOT, HISTORICAL_PROFILE, full_plan,
    quarter_ends, run_plan,
)
from bft65b.acquisition.storage import (
    ParquetStore, StorageConflictError, StorageValidationError, arrow_schema,
)
from bft65b.config import (
    BASE_FINANCIAL_FIELDS, EXTENDED_FINANCIAL_FIELDS, ProjectConfig, load_config,
)
from bft65b.manifest import RunManifest


def financial(cert, quarter="19840331", **changes):
    row = {
        "CERT": cert, "REPDTE": quarter, "STNAME": "IOWA",
        "ASSET": "100.0000", "DEP": 80, "EQ": 10,
        "CHBAL": "3.1250", "SC": 5, "BRO": None, "DEPDOM": 70,
        "EINTEXP": "2.5", "LNLSGR": 55, "LNRE": 20, "LNCI": 15,
        "P3ASSET": None, "P9ASSET": "0.25", "NAASSET": 0.1,
        "NTLNLS": -1, "ELNATR": "0.5", "NIMY": "3.25",
        "RBCRWAJ": 12.5,
    }
    row.update(changes)
    return row


class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class Session:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []
        self.headers = {}
        self.index = "synthetic-index"
        self.interrupt_filter = None

    def get(self, url, *, params, timeout):
        params = dict(params)
        self.calls.append((url, params))
        if self.interrupt_filter is not None and self.interrupt_filter == params.get("filters"):
            raise KeyboardInterrupt("synthetic acquisition interruption")
        endpoint = url.rsplit("/", 1)[-1]
        quarter = params.get("filters", "").split(":")[-1] if endpoint == "financials" else None
        all_rows = self.rows[(endpoint, quarter)]
        offset, limit = int(params["offset"]), int(params["limit"])
        page = all_rows[offset:offset + limit]
        return Response({
            "meta": {"total": len(all_rows), "parameters": {
                "fields": params["fields"], "offset": str(offset),
                "limit": str(limit), "filters": params.get("filters", ""),
            }, "index": {"name": self.index, "createTimestamp": "2026-01-01T00:00:00Z"}},
            "data": [{"data": row, "score": 0} for row in page],
        })


def context(tmp_path, rows, *, page_size=2):
    config = ProjectConfig(financial_fields=EXTENDED_FINANCIAL_FIELDS,
                           data_root="data_raw/task_1_5a", api_page_size=page_size)
    session = Session(rows)
    client = FDICClient(config, session=session, sleep=lambda _: None)
    store = ParquetStore(tmp_path / "data_raw/task_1_5a", config)
    return config, client, store, session, tmp_path / "runs/extended.json", tmp_path / "runs/extended_summary.json"


def test_exact_verified_field_sets_and_unverified_rejection():
    assert len(BASE_FINANCIAL_FIELDS) == 11
    assert len(EXTENDED_FINANCIAL_FIELDS) == 26
    assert ProjectConfig(financial_fields=EXTENDED_FINANCIAL_FIELDS).fields_verified
    with pytest.raises(ValueError, match="exact dictionary-verified"):
        ProjectConfig(financial_fields=BASE_FINANCIAL_FIELDS + ("CHBAL",))
    with pytest.raises(ValueError, match="exact dictionary-verified"):
        ProjectConfig(financial_fields=EXTENDED_FINANCIAL_FIELDS + ("UNVERIFIED",))
    with pytest.raises(ValueError, match="tuple types"):
        ProjectConfig(financial_fields=list(EXTENDED_FINANCIAL_FIELDS))
    with pytest.raises(ValueError, match="not been verified"):
        FDICClient(ProjectConfig(financial_fields=EXTENDED_FINANCIAL_FIELDS,
                                 fields_verified=False))


def test_checked_in_extended_config_matches_verified_inventory():
    path = Path(__file__).resolve().parents[1] / "config/financials_extended_v2.json"
    config = load_config(path)
    assert config.financial_fields == EXTENDED_FINANCIAL_FIELDS
    assert config.data_root == "data_raw/task_1_5a"


def test_new_schema_and_paths_cannot_alias_accepted_v1(tmp_path):
    v1 = ParquetStore(tmp_path / "data_raw", ProjectConfig())
    config = ProjectConfig(financial_fields=EXTENDED_FINANCIAL_FIELDS,
                           data_root="data_raw/task_1_5a")
    v2 = ParquetStore(tmp_path / "data_raw/task_1_5a", config)
    assert v1.target_path("financials", "20230331") == (
        tmp_path / "data_raw/fdic/financials/REPDTE=20230331/part-000.parquet"
    )
    assert v2.target_path("financials", "20230331") == (
        tmp_path / "data_raw/task_1_5a/fdic_extended_v2/financials/REPDTE=20230331/part-000.parquet"
    )
    assert v1.schema_version("financials") == "1" and v2.schema_version("financials") == "2"
    assert v2.schema_version("failures") == "2" and v2.schema_version("institutions") == "1"
    assert v2.schema("financials", "20230331").field("CHBAL").type == pa.decimal128(20, 4)
    assert v2.schema("financials", "20230331").field("P3ASSET").type == pa.float64()
    assert v1.schema("financials", "20230331").equals(
        arrow_schema("financials", "20230331"), check_metadata=True)


def test_mixed_types_null_absent_and_manifest_schema(tmp_path):
    config, _, store, _, path, _ = context(tmp_path, {})
    row = financial("0040")
    del row["EINTEXP"]
    manifest = RunManifest.new("extended", config)
    result = store.write("financials", [row], quarter="19840331", source_total=1,
                         source_query="synthetic:financials", manifest=manifest, manifest_path=path)
    table = pq.ParquetFile(result.path).read()
    assert table.schema.equals(store.schema("financials", "19840331"), check_metadata=True)
    record = table.to_pylist()[0]
    assert record["CERT"] == 40
    assert record["CHBAL"] == Decimal("3.1250")
    assert record["P3ASSET"] is None and record["P9ASSET"] == 0.25
    assert record["EINTEXP"] is None and "EINTEXP" in record["SOURCE_MISSING_FIELDS"]
    assert "P3ASSET" not in record["SOURCE_MISSING_FIELDS"]
    assert RunManifest.read(path).outputs[0]["schema_version"] == "2"
    assert store.can_skip("financials", path, quarter="19840331", source_total=1,
                          source_query="synthetic:financials")


@pytest.mark.parametrize("field,value,match", [
    ("CHBAL", "1.00001", "precision"),
    ("BRO", "N/A", "finite decimal"),
    ("NIMY", "NaN", "finite"),
    ("P3ASSET", True, "numeric"),
    ("P3ASSET", 9007199254740993, "exact float64 integer precision"),
    ("P9ASSET", "0.1234567890123456789", "loses source numeric precision"),
])
def test_invalid_extended_numeric_never_promotes(tmp_path, field, value, match):
    config, _, store, _, path, _ = context(tmp_path, {})
    with pytest.raises(StorageValidationError, match=match):
        store.write("financials", [financial(40, **{field: value})], quarter="19840331",
                    source_total=1, source_query="synthetic:financials",
                    manifest=RunManifest.new("bad", config), manifest_path=path)
    assert not store.target_path("financials", "19840331").exists()


def test_multi_page_complete_quarter_and_profile(tmp_path):
    rows = {("financials", "19840331"): [financial(10), financial(20), financial(30)]}
    config, client, store, session, path, summary_path = context(tmp_path, rows)
    plan = AcquisitionPlan("multi_page_test", ("19840331",))
    result = run_plan(plan=plan, config=config, client=client, store=store,
                      manifest_path=path, summary_path=summary_path)
    entry = result["quarters"][0]
    assert result["status"] == "complete"
    assert entry["api_total"] == entry["downloaded_rows"] == 3
    assert entry["page_offsets"] == [0, 2] and entry["page_row_counts"] == [2, 1]
    assert entry["null_fields"]["P3ASSET"] == 3
    assert entry["explicit_null_rates"]["P3ASSET"] == 1.0
    assert entry["missing_fields"]["NCLNLS"] == 3
    assert entry["source_type_counts"]["CHBAL"] == {"str": 3}
    assert entry["conversion_anomalies"] == 0
    assert [call[1]["offset"] for call in session.calls] == [0, 2]
    assert store.validate_file(store.target_path("financials", "19840331"), "financials",
                               quarter="19840331", expected_rows=3,
                               expected_sha256=entry["sha256"]).compression == "ZSTD"


def test_duplicate_bank_quarter_and_zero_row_stop(tmp_path):
    for rows, message in [
        ([financial(10), financial("0010")], "Duplicate"),
        ([], "zero records"),
    ]:
        root = tmp_path / ("empty" if not rows else "duplicate")
        data = {("financials", "19840331"): rows}
        config, client, store, _, path, summary = context(root, data)
        with pytest.raises(FDICContractError, match=message):
            run_plan(plan=AcquisitionPlan("one", ("19840331",)), config=config,
                     client=client, store=store, manifest_path=path, summary_path=summary)
        assert RunManifest.read(path).status == "failed"
        assert not store.target_path("financials", "19840331").exists()


def test_healthy_file_conflict_and_v1_isolation(tmp_path):
    config, client, store, session, path, summary = context(
        tmp_path, {("financials", "19840331"): [financial(10)]}
    )
    plan = AcquisitionPlan("one", ("19840331",))
    run_plan(plan=plan, config=config, client=client, store=store,
             manifest_path=path, summary_path=summary)
    target = store.target_path("financials", "19840331")
    original = target.read_bytes()
    session.rows[("financials", "19840331")] = [financial(10, ASSET=101)]
    with pytest.raises(StorageConflictError, match="differs"):
        run_plan(plan=plan, config=config, client=client, store=store,
                 manifest_path=tmp_path / "runs/new.json",
                 summary_path=tmp_path / "runs/new_summary.json")
    assert target.read_bytes() == original
    assert not (tmp_path / "data_raw/fdic/financials/REPDTE=19840331/part-000.parquet").exists()


def test_interrupted_extended_plan_resumes_and_preserves_bytes(tmp_path):
    rows = {("financials", "20230331"): [financial(10, "20230331")],
            ("financials", "20230630"): [financial(20, "20230630")]}
    config, client, store, session, path, summary = context(tmp_path, rows)
    plan = AcquisitionPlan("two", ("20230331", "20230630"))
    session.interrupt_filter = "REPDTE:20230630"
    with pytest.raises(KeyboardInterrupt, match="synthetic"):
        run_plan(plan=plan, config=config, client=client, store=store,
                 manifest_path=path, summary_path=summary)
    first = store.target_path("financials", "20230331")
    before, mtime = first.read_bytes(), first.stat().st_mtime_ns
    assert RunManifest.read(path).status == "running"
    session.interrupt_filter = None
    session.calls.clear()
    result = run_plan(plan=plan, config=config, client=client, store=store,
                      manifest_path=path, summary_path=summary, resume=True)
    assert result["status"] == "complete" and result["quarters"][0]["recovery_action"] == "skipped"
    assert first.read_bytes() == before and first.stat().st_mtime_ns == mtime
    assert len(RunManifest.read(path).outputs) == 2
    assert [call[1]["limit"] for call in session.calls if call[1].get("filters") == "REPDTE:20230331"] == [1]


def test_source_index_change_blocks_resume(tmp_path):
    data = {("financials", "19840331"): [financial(10)]}
    config, client, store, session, path, summary = context(tmp_path, data)
    plan = AcquisitionPlan("one", ("19840331",))
    run_plan(plan=plan, config=config, client=client, store=store,
             manifest_path=path, summary_path=summary)
    target = store.target_path("financials", "19840331")
    original = target.read_bytes()
    session.index = "changed-index"
    with pytest.raises(FDICContractError, match="Source total or index changed"):
        run_plan(plan=plan, config=config, client=client, store=store,
                 manifest_path=path, summary_path=summary, resume=True)
    assert target.read_bytes() == original and RunManifest.read(path).status == "complete"


def test_extended_schema_corruption_is_refetched_and_rebuilt(tmp_path):
    data = {("financials", "19840331"): [financial(10)]}
    config, client, store, session, path, summary = context(tmp_path, data)
    plan = AcquisitionPlan("one", ("19840331",))
    run_plan(plan=plan, config=config, client=client, store=store,
             manifest_path=path, summary_path=summary)
    target = store.target_path("financials", "19840331")
    old = target.read_bytes()
    manifest = RunManifest.read(path)
    manifest.set_status("running")
    manifest.write(path)
    target.write_bytes(old[:40])
    assert not store.can_skip("financials", path, quarter="19840331", source_total=1,
                              source_query=manifest.outputs[-1]["source_query"])
    session.calls.clear()
    result = run_plan(plan=plan, config=config, client=client, store=store,
                      manifest_path=path, summary_path=summary, resume=True)
    assert result["status"] == "complete"
    assert result["quarters"][0]["recovery_action"] == "written"
    assert target.read_bytes() == old
    assert len(session.calls) == 1 and session.calls[0][1]["limit"] == 2
    assert list(target.parent.glob("part-000.parquet.corrupt.*"))
    assert store.validate_file(target, "financials", quarter="19840331",
                               expected_rows=1, expected_sha256=result["quarters"][0]["sha256"])


def test_stale_extended_schema_manifest_refetches_but_preserves_valid_bytes(tmp_path):
    data = {("financials", "19840331"): [financial(10)]}
    config, client, store, session, path, summary = context(tmp_path, data)
    plan = AcquisitionPlan("one", ("19840331",))
    run_plan(plan=plan, config=config, client=client, store=store,
             manifest_path=path, summary_path=summary)
    target = store.target_path("financials", "19840331")
    before, mtime = target.read_bytes(), target.stat().st_mtime_ns
    manifest = RunManifest.read(path)
    manifest.set_status("running")
    manifest.outputs[-1]["schema_version"] = "1"
    manifest.write(path)
    assert not store.can_skip("financials", path, quarter="19840331", source_total=1,
                              source_query=manifest.outputs[-1]["source_query"])
    session.calls.clear()
    result = run_plan(plan=plan, config=config, client=client, store=store,
                      manifest_path=path, summary_path=summary, resume=True)
    assert result["status"] == "complete"
    assert result["quarters"][0]["recovery_action"] == "reconciled"
    assert target.read_bytes() == before and target.stat().st_mtime_ns == mtime
    assert RunManifest.read(path).outputs[-1]["schema_version"] == "2"
    assert len(session.calls) == 1 and session.calls[0][1]["limit"] == 2


def test_future_small_tables_preserve_null_and_absent_provenance(tmp_path):
    rows = {
        ("financials", "20230331"): [financial(10, "20230331")],
        ("failures", None): [{"ID": "source-1", "CERT": None, "FAILDATE": "5/28/1934", "COST": None}],
        ("institutions", None): [{"CERT": 10, "STNAME": "IOWA", "ACTIVE": 0}],
    }
    config, client, store, _, path, summary = context(tmp_path, rows)
    plan = AcquisitionPlan("small_tables_synthetic", ("20230331",), True)
    result = run_plan(plan=plan, config=config, client=client, store=store,
                      manifest_path=path, summary_path=summary)
    assert result["status"] == "complete" and len(RunManifest.read(path).outputs) == 3
    assert result["small_tables"][0]["unresolved_linkage"]["count"] == 1
    assert pq.ParquetFile(store.target_path("failures")).read(columns=["CERT"])["CERT"].null_count == 1
    record = pq.ParquetFile(store.target_path("institutions")).read().to_pylist()[0]
    assert record["CHANGEC1"] is None and "CHANGEC1" in record["SOURCE_MISSING_FIELDS"]


def test_missing_future_small_table_cannot_complete(tmp_path):
    rows = {
        ("financials", "20230331"): [financial(10, "20230331")],
        ("failures", None): [{"ID": "source-1", "CERT": None, "FAILDATE": "5/28/1934"}],
        ("institutions", None): [],
    }
    config, client, store, _, path, summary = context(tmp_path, rows)
    with pytest.raises(FDICContractError, match="zero records"):
        run_plan(plan=AcquisitionPlan("small_tables_synthetic", ("20230331",), True),
                 config=config, client=client, store=store,
                 manifest_path=path, summary_path=summary)
    assert RunManifest.read(path).status == "failed"
    assert len(RunManifest.read(path).outputs) == 2


def test_full_plan_exact_scope_and_no_cli_authorization(tmp_path):
    config, client, store, _, path, summary = context(tmp_path, {})
    plan = full_plan(config)
    assert len(plan.quarters) == 170
    assert plan.quarters[0] == "19840331" and plan.quarters[-1] == "20260630"
    assert len(plan.jobs(config)) == 172
    assert plan.jobs(config)[-2][0] == "failures"
    assert plan.jobs(config)[-1][0] == "institutions"
    assert HISTORICAL_PROFILE.quarters == ("19840331", "19891231", "20091231", "20260630")
    assert EXTENDED_PILOT.quarters == ("20230331", "20230630", "20230930", "20231231")
    with pytest.raises(ValueError, match="disabled"):
        run_plan(plan=plan, config=config, client=client, store=store,
                 manifest_path=path, summary_path=summary)
    with pytest.raises(ValueError, match="eight approved bounded quarters"):
        run_plan(plan=AcquisitionPlan("renamed_full", plan.quarters, True),
                 config=config, client=client, store=store,
                 manifest_path=path, summary_path=summary)
    assert not path.exists() and not store.target_path("financials", "19840331").exists()
    assert quarter_ends("1984-03-31", "2026-06-30") == plan.quarters
    with pytest.raises(ValueError, match="official assignment"):
        full_plan(ProjectConfig(financial_fields=EXTENDED_FINANCIAL_FIELDS,
                                financial_start="1985-03-31"))
    with pytest.raises(ValueError, match="verified extended"):
        full_plan(ProjectConfig())


def test_accepted_manifest_paths_are_protected(tmp_path):
    config, client, store, _, _, _ = context(tmp_path, {})
    with pytest.raises(ValueError, match="protected"):
        run_plan(plan=AcquisitionPlan("one", ("19840331",)), config=config,
                 client=client, store=store,
                 manifest_path=tmp_path / "runs/task_1_3_pilot_resume3.json",
                 summary_path=tmp_path / "runs/summary.json")
