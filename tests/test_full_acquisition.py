"""Synthetic full-run entrypoint and attempt-versioned recovery tests."""

from pathlib import Path
import json
import multiprocessing

import pytest
import requests

from bft65b.acquisition.client import FDICClient, FDICContractError, FDICRequestError
from bft65b.acquisition.extended import AcquisitionPlan
from bft65b.acquisition.full import dry_run, full_paths, run_attempt
from bft65b.acquisition.storage import ParquetStore, StorageConflictError, StorageValidationError
from bft65b.config import EXTENDED_FINANCIAL_FIELDS, ProjectConfig, load_config
from bft65b.manifest import RunManifest
from bft65b.acquisition import full as full_module


class Response:
    def __init__(self, payload, status_code=200):
        self.payload, self.status_code = payload, status_code

    def json(self):
        return self.payload


class SimulatedProcessStop(BaseException):
    """Mimics process termination without triggering pytest's keyboard handler."""


class Session:
    def __init__(self, rows):
        self.rows = rows
        self.index = "synthetic-index"
        self.total_delta = 0
        self.calls = []
        self.fail_filter = None
        self.failure = None
        self.headers = {}

    def get(self, url, *, params, timeout):
        params = dict(params)
        self.calls.append((url, params))
        if self.fail_filter is not None and params.get("filters") == self.fail_filter:
            if self.failure == "interrupt":
                raise SimulatedProcessStop("synthetic interruption")
            if self.failure == "timeout":
                raise requests.Timeout("synthetic timeout")
            if self.failure in (429, 503):
                return Response({}, self.failure)
        endpoint = url.rsplit("/", 1)[-1]
        quarter = params.get("filters", "").split(":")[-1] if endpoint == "financials" else None
        rows = self.rows[(endpoint, quarter)]
        offset, limit = int(params["offset"]), int(params["limit"])
        return Response({
            "meta": {"total": len(rows) + self.total_delta,
                     "parameters": {"fields": params["fields"], "offset": str(offset),
                                    "limit": str(limit), "filters": params.get("filters", "")},
                     "index": {"name": self.index, "createTimestamp": "2026-01-01T00:00:00Z"}},
            "data": [{"data": row, "score": 0} for row in rows[offset:offset + limit]],
        })


def financial(cert, quarter):
    return {"CERT": cert, "REPDTE": quarter, "STNAME": "SYNTHETIC", "ASSET": "100.0000",
            "DEP": 80, "EQ": 10, "ROA": "1.25", "CHBAL": None, "RBCRWAJ": "12.5"}


def setup(tmp_path, *, count=3, include_small=True):
    config = ProjectConfig(financial_fields=EXTENDED_FINANCIAL_FIELDS,
                           data_root="data_raw/task_1_5_full", api_page_size=2)
    quarters = ("19840331", "19840630", "19840930", "19841231")[:count]
    plan = AcquisitionPlan("synthetic_full", quarters, include_small)
    rows = {("financials", q): [financial(1, q), financial(2, q), financial(3, q)]
            for q in quarters}
    if include_small:
        rows[("failures", None)] = [
            {"ID": "source-2", "CERT": 2, "FAILDATE": "2008-01-01", "COST": "1.0000"},
            {"ID": "source-1", "CERT": None, "FAILDATE": "5/28/1934", "COST": None},
        ]
        rows[("institutions", None)] = [
            {"CERT": 1, "STNAME": "SYNTHETIC", "ACTIVE": 1},
            {"CERT": 2, "STNAME": "SYNTHETIC", "ACTIVE": 0, "CHANGEC1": None},
        ]
    session = Session(rows)
    client = FDICClient(config, session=session, max_attempts=2, backoff_seconds=0, sleep=lambda _: None)
    paths = full_paths(tmp_path, config, "first")
    store = ParquetStore(paths.raw_root, config)
    return config, plan, client, store, session, paths


def run(config, plan, client, store, paths, previous=None):
    return run_attempt(plan=plan, config=config, client=client, store=store,
                       paths=paths, previous_paths=previous,
                       allow_bounded_test_plan=True)


def test_bounded_plan_cannot_masquerade_as_full_run(tmp_path):
    config, plan, client, store, _, paths = setup(tmp_path)
    with pytest.raises(ValueError, match="exact 170-quarter"):
        run_attempt(plan=plan, config=config, client=client, store=store, paths=paths)
    assert not paths.manifest.exists()


def test_exact_dry_run_and_rejected_configuration(tmp_path, monkeypatch):
    cfg = load_config(Path(__file__).resolve().parents[1] / "config/financials_full_v2.json")
    result = dry_run(tmp_path, cfg, "dry")
    assert (result["financial_quarters"], result["total_jobs"], result["network_requests"]) == (170, 172, 0)
    assert result["jobs"][0]["quarter"] == "19840331"
    assert result["jobs"][169]["quarter"] == "20260630"
    assert [j["endpoint"] for j in result["jobs"][-2:]] == ["failures", "institutions"]
    assert not (tmp_path / "runs").exists()
    assert not Path(result["lock_path"]).exists()
    assert not (tmp_path / "data_raw").exists()
    assert result["probe_interval_seconds"] == full_module.FULL_PROBE_INTERVAL_SECONDS
    assert result["api_retry"]["max_attempts"] == full_module.FULL_MAX_ATTEMPTS
    with pytest.raises(ValueError, match="26-field"):
        dry_run(tmp_path, ProjectConfig(data_root="data_raw/task_1_5_full"), "dry")
    with pytest.raises(ValueError, match="dedicated"):
        dry_run(tmp_path, ProjectConfig(financial_fields=EXTENDED_FINANCIAL_FIELDS), "dry")
    with pytest.raises(ValueError, match="new attempt"):
        dry_run(tmp_path, cfg, "dry", "dry")
    monkeypatch.setenv("BFT65B_DATA_ROOT", str(tmp_path / "data_raw/task_1_5a"))
    with pytest.raises(ValueError, match="redirects"):
        dry_run(tmp_path, cfg, "dry")


def test_source_probe_pacing_is_monotonic_and_preserves_request(tmp_path, monkeypatch):
    now = [100.0]
    delays = []
    calls = []

    def sleep(seconds):
        delays.append(seconds)
        now[0] += seconds

    class ProbeClient:
        def fetch_page(self, endpoint, *, filters, limit):
            calls.append((endpoint, filters, limit))
            return "page"

    monkeypatch.setattr(full_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(full_module.time, "sleep", sleep)
    pacer = full_module._ProbePacer(3.2)
    client = ProbeClient()
    assert pacer.fetch(client, "financials", "19840331") == "page"
    assert pacer.fetch(client, "failures", None) == "page"
    assert delays == [pytest.approx(3.2)]
    assert calls == [("financials", "REPDTE:19840331", 1), ("failures", None, 1)]


def test_complete_multi_job_and_linkage(tmp_path):
    config, plan, client, store, _, paths = setup(tmp_path)
    summary = run(config, plan, client, store, paths)
    manifest = RunManifest.read(paths.manifest)
    assert summary["status"] == manifest.status == "complete"
    assert len(summary["quarters"]) == 3 and len(summary["small_tables"]) == 2
    assert len(manifest.outputs) == 5
    assert json.loads(paths.linkage_audit.read_text())["source_ids"] == ["source-1"]
    assert summary["quarters"][0]["page_offsets"] == [0, 2]


@pytest.mark.parametrize("missing", ["financials", "failures", "institutions"])
def test_missing_job_never_completes(tmp_path, missing):
    config, plan, client, store, session, paths = setup(tmp_path)
    key = (missing, plan.quarters[-1] if missing == "financials" else None)
    session.rows[key] = []
    with pytest.raises(FDICContractError):
        run(config, plan, client, store, paths)
    assert RunManifest.read(paths.manifest).status == "failed"
    assert json.loads(paths.summary.read_text())["status"] == "failed"


@pytest.mark.parametrize("failure", ["interrupt", "timeout", 429, 503])
def test_interruption_retry_exhaustion_and_recovery(tmp_path, failure):
    config, plan, client, store, session, paths = setup(tmp_path)
    session.fail_filter = "REPDTE:19840930"
    session.failure = failure
    with pytest.raises((SimulatedProcessStop, FDICRequestError)):
        run(config, plan, client, store, paths)
    before = {q: store.validate_file(store.target_path("financials", q), "financials", quarter=q).sha256
              for q in plan.quarters[:2]}
    old_manifest_bytes = paths.manifest.read_bytes()
    session.fail_filter = None
    recovered = full_paths(tmp_path, config, "recovery")
    result = run(config, plan, client, store, recovered, paths)
    assert result["status"] == "complete"
    assert all(item["recovery_action"] == "inherited" for item in result["quarters"][:2])
    assert paths.manifest.read_bytes() == old_manifest_bytes
    assert before == {q: store.validate_file(store.target_path("financials", q), "financials", quarter=q).sha256
                      for q in plan.quarters[:2]}


def test_changed_source_rejects_recovery(tmp_path):
    config, plan, client, store, session, paths = setup(tmp_path)
    session.fail_filter, session.failure = "REPDTE:19840930", "interrupt"
    with pytest.raises(SimulatedProcessStop):
        run(config, plan, client, store, paths)
    session.fail_filter = None
    session.index = "changed-index"
    with pytest.raises(FDICContractError, match="changed"):
        run(config, plan, client, store, full_paths(tmp_path, config, "next"), paths)


def test_corrupt_recovery_and_healthy_conflict(tmp_path):
    config, plan, client, store, session, paths = setup(tmp_path)
    session.fail_filter, session.failure = "REPDTE:19840930", "interrupt"
    with pytest.raises(SimulatedProcessStop):
        run(config, plan, client, store, paths)
    target = store.target_path("financials", plan.quarters[0])
    target.write_bytes(target.read_bytes()[:20])
    session.fail_filter = None
    repaired = full_paths(tmp_path, config, "repair")
    result = run(config, plan, client, store, repaired, paths)
    assert result["status"] == "complete"
    assert result["quarters"][0]["recovery_action"] == "written"
    assert store.validate_file(target, "financials", quarter=plan.quarters[0]).rows == 3
    session.rows[("financials", plan.quarters[0])][0]["ASSET"] = "999.0000"
    with pytest.raises(StorageConflictError):
        run(config, plan, client, store, full_paths(tmp_path, config, "conflict"))


def test_missing_summary_recovers_by_reconciliation(tmp_path):
    config, plan, client, store, _, paths = setup(tmp_path)
    run(config, plan, client, store, paths)
    original = store.validate_file(store.target_path("financials", plan.quarters[0]),
                                   "financials", quarter=plan.quarters[0]).sha256
    paths.summary.unlink()  # Isolated synthetic fixture only.
    new_paths = full_paths(tmp_path, config, "new")
    result = run(config, plan, client, store, new_paths, paths)
    assert result["status"] == "complete"
    assert result["quarters"][0]["recovery_action"] == "reconciled"
    assert store.validate_file(store.target_path("financials", plan.quarters[0]),
                               "financials", quarter=plan.quarters[0]).sha256 == original


def test_crash_after_parquet_promotion_before_manifest_persistence(tmp_path, monkeypatch):
    config, plan, client, store, _, paths = setup(tmp_path)
    real_write = RunManifest.write
    crashed = False

    def fail_once(self, path):
        nonlocal crashed
        if not crashed and self.outputs:
            crashed = True
            raise OSError("synthetic manifest commit failure")
        return real_write(self, path)

    with monkeypatch.context() as context:
        context.setattr(RunManifest, "write", fail_once)
        with pytest.raises(OSError, match="synthetic manifest"):
            run(config, plan, client, store, paths)
    first = store.target_path("financials", plan.quarters[0])
    before = store.validate_file(first, "financials", quarter=plan.quarters[0]).sha256
    old = paths.manifest.read_bytes()
    recovered = run(config, plan, client, store,
                    full_paths(tmp_path, config, "after-promotion"), paths)
    assert recovered["status"] == "complete"
    assert recovered["quarters"][0]["recovery_action"] == "reconciled"
    assert store.validate_file(first, "financials", quarter=plan.quarters[0]).sha256 == before
    assert paths.manifest.read_bytes() == old


def test_crash_after_manifest_completion_before_summary(tmp_path, monkeypatch):
    config, plan, client, store, _, paths = setup(tmp_path)
    real_atomic = full_module._atomic_json

    def fail_complete(path, value):
        if path == paths.summary and value.get("status") == "complete":
            raise OSError("synthetic summary commit failure")
        return real_atomic(path, value)

    with monkeypatch.context() as context:
        context.setattr(full_module, "_atomic_json", fail_complete)
        with pytest.raises(OSError, match="summary commit"):
            run(config, plan, client, store, paths)
    assert RunManifest.read(paths.manifest).status == "complete"
    assert json.loads(paths.summary.read_text())["status"] == "running"
    original = paths.manifest.read_bytes()
    recovered = run(config, plan, client, store,
                    full_paths(tmp_path, config, "after-summary"), paths)
    assert recovered["status"] == "complete"
    assert all(row["recovery_action"] == "inherited"
               for row in recovered["quarters"] + recovered["small_tables"])
    assert paths.manifest.read_bytes() == original


def test_changed_total_rejects_recovery(tmp_path):
    config, plan, client, store, session, paths = setup(tmp_path)
    session.fail_filter, session.failure = "REPDTE:19840930", "timeout"
    with pytest.raises(FDICRequestError):
        run(config, plan, client, store, paths)
    session.fail_filter = None
    session.total_delta = 1
    with pytest.raises(FDICContractError, match="changed"):
        run(config, plan, client, store, full_paths(tmp_path, config, "total-changed"), paths)


def test_final_source_failure_names_early_financials_without_relabeling_institutions(tmp_path, monkeypatch):
    config, plan, client, store, session, paths = setup(tmp_path)
    original_get = session.get

    def changed_during_sweep(url, *, params, timeout):
        response = original_get(url, params=params, timeout=timeout)
        if (url.endswith("/financials") and params.get("filters") == "REPDTE:19840331"
                and int(params["limit"]) == 1):
            response.payload["meta"]["index"]["name"] = "new-index"
        return response

    with monkeypatch.context() as context:
        context.setattr(session, "get", changed_during_sweep)
        with pytest.raises(FDICContractError, match="Source total/index changed"):
            run(config, plan, client, store, paths)
    manifest = RunManifest.read(paths.manifest)
    summary = json.loads(paths.summary.read_text())
    assert manifest.status == summary["status"] == "failed"
    assert summary["failure"] == {
        "phase": "final_source", "endpoint": "financials", "quarter": "19840331",
        "error_type": "FDICContractError",
        "message": "Source total/index changed for financials:19840331",
    }
    assert manifest.stages["validation:final_source:financials:19840331"] == "failed"
    assert manifest.stages["acquisition:institutions:all"] == "complete"
    assert manifest.stages["storage:institutions:all"] == "complete"
    assert manifest.stages.get("acquisition:institutions:all") != "failed"
    recovered = run(config, plan, client, store,
                    full_paths(tmp_path, config, "after-source-fault"), paths)
    assert recovered["status"] == "complete"


def test_final_parquet_failure_names_early_financials_without_relabeling_institutions(tmp_path, monkeypatch):
    config, plan, client, store, _, paths = setup(tmp_path)
    original_validate = store.validate_file

    def fail_final_readback(path, dataset, **kwargs):
        if (dataset == "financials" and kwargs.get("quarter") == "19840331"
                and store.target_path("institutions").exists()):
            raise StorageValidationError("synthetic final Parquet read-back failure")
        return original_validate(path, dataset, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(store, "validate_file", fail_final_readback)
        with pytest.raises(StorageValidationError, match="synthetic final Parquet"):
            run(config, plan, client, store, paths)
    manifest = RunManifest.read(paths.manifest)
    summary = json.loads(paths.summary.read_text())
    assert manifest.status == summary["status"] == "failed"
    assert summary["failure"] == {
        "phase": "final_parquet", "endpoint": "financials", "quarter": "19840331",
        "error_type": "StorageValidationError",
        "message": "synthetic final Parquet read-back failure",
    }
    assert manifest.stages["validation:final_parquet:financials:19840331"] == "failed"
    assert manifest.stages["acquisition:institutions:all"] == "complete"
    assert manifest.stages["storage:institutions:all"] == "complete"
    recovered = run(config, plan, client, store,
                    full_paths(tmp_path, config, "after-parquet-fault"), paths)
    assert recovered["status"] == "complete"


def test_plan_wide_final_validation_failure_is_plan_scoped(tmp_path, monkeypatch):
    config, plan, client, store, _, paths = setup(tmp_path)

    def fail_plan(*args, **kwargs):
        raise StorageValidationError("synthetic plan-wide mismatch")

    with monkeypatch.context() as context:
        context.setattr(full_module, "_verify_complete", fail_plan)
        with pytest.raises(StorageValidationError, match="plan-wide mismatch"):
            run(config, plan, client, store, paths)
    manifest = RunManifest.read(paths.manifest)
    summary = json.loads(paths.summary.read_text())
    assert summary["failure"]["phase"] == "final_parquet"
    assert summary["failure"]["endpoint"] is None
    assert summary["failure"]["quarter"] is None
    assert summary["failed_stage"] == "validation:final_parquet:plan"
    assert manifest.stages["validation:final_parquet:plan"] == "failed"
    assert manifest.stages["acquisition:institutions:all"] == "complete"


def test_direct_full_namespace_write_requires_ownership(tmp_path):
    config, plan, _, store, _, paths = setup(tmp_path, count=1, include_small=False)
    manifest = RunManifest.new("synthetic-direct-write", config)
    with pytest.raises(StorageValidationError, match="exclusive namespace ownership"):
        store.write("financials", [financial(1, plan.quarters[0])],
                    quarter=plan.quarters[0], source_total=1,
                    manifest=manifest, manifest_path=paths.manifest,
                    source_query="synthetic-source")
    assert not store.target_path("financials", plan.quarters[0]).exists()
    assert not paths.manifest.exists()


def _process_attempt(root, attempt, entered, release, result_queue):
    """Run a normal synthetic attempt in a spawned OS process."""
    config, plan, client, store, session, _ = setup(Path(root), count=1, include_small=False)
    paths = full_paths(Path(root), config, attempt)
    if entered is not None:
        original_get = session.get

        def pause_after_manifest(url, *, params, timeout):
            entered.set()
            if not release.wait(30):
                raise RuntimeError("holder signal timed out")
            return original_get(url, params=params, timeout=timeout)

        session.get = pause_after_manifest
    try:
        summary = run(config, plan, client, store, paths)
        result_queue.put(("complete", summary["status"]))
    except Exception as exc:
        result_queue.put((type(exc).__name__, str(exc)))


def test_two_real_processes_exclude_competing_attempt_and_release_on_termination(tmp_path):
    config, plan, client, store, _, accepted = setup(tmp_path, count=1, include_small=False)
    assert run(config, plan, client, store, accepted)["status"] == "complete"
    target = store.target_path("financials", plan.quarters[0])
    accepted_sha = store.validate_file(target, "financials", quarter=plan.quarters[0]).sha256
    context = multiprocessing.get_context("spawn")
    entered, release, results = context.Event(), context.Event(), context.Queue()
    holder = context.Process(target=_process_attempt,
                             args=(str(tmp_path), "holder", entered, release, results))
    competitor = context.Process(target=_process_attempt,
                                 args=(str(tmp_path), "competitor", None, None, results))
    try:
        holder.start()
        assert entered.wait(20), "holder never reached the explicit acquisition signal"
        holder_paths = full_paths(tmp_path, config, "holder")
        manifest_before = holder_paths.manifest.read_bytes()
        summary_before = holder_paths.summary.read_bytes()
        competitor.start()
        competitor.join(20)
        assert competitor.exitcode == 0
        outcome, message = results.get(timeout=5)
        assert outcome == "FullAcquisitionBusyError"
        assert "owned by another process" in message
        assert not full_paths(tmp_path, config, "competitor").manifest.exists()
        assert not full_paths(tmp_path, config, "competitor").summary.exists()
        assert holder_paths.manifest.read_bytes() == manifest_before
        assert holder_paths.summary.read_bytes() == summary_before
        assert store.validate_file(target, "financials", quarter=plan.quarters[0]).sha256 == accepted_sha
        assert not list(target.parent.glob("*.corrupt.*"))
        holder.terminate()  # Abrupt OS process termination releases the byte-range lock.
        holder.join(20)
        assert holder.exitcode is not None and holder.exitcode != 0
        successor = context.Process(target=_process_attempt,
                                    args=(str(tmp_path), "after-crash", None, None, results))
        try:
            successor.start()
            successor.join(20)
            assert successor.exitcode == 0
            assert results.get(timeout=5) == ("complete", "complete")
        finally:
            if successor.is_alive():
                successor.terminate()
                successor.join(20)
        assert store.validate_file(target, "financials", quarter=plan.quarters[0]).sha256 == accepted_sha
        assert holder_paths.manifest.read_bytes() == manifest_before
        assert holder_paths.summary.read_bytes() == summary_before
        assert not list(target.parent.glob("*.corrupt.*"))
    finally:
        for process in (competitor, holder):
            if process.is_alive():
                process.terminate()
                process.join(20)
