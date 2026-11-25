"""Chronological split feasibility and frozen assignments (Part 5, first gate).

Only outcome counts and source IDs are inspected here. No feature values are
read, preprocessing is fitted, or predictive test score is computed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from bft65b.config import load_config
from bft65b.features.label import _qualifies, _verify_repair
from bft65b.features.leakage_audit import verify_saved as verify_leakage
from bft65b.features.production import verify_saved as verify_labels
from bft65b.features.table import verify_saved as verify_table
from bft65b.manifest import RunManifest
from bft65b.panel.build import _write_verified
from bft65b.panel.retrospective import _root, _run_id, _quarter, verify_saved as verify_membership
from bft65b.quality.audit import _read_json, _sha


VERSION = "chronological_split_v1"
NAMESPACE = "task_5_1_split"
MODEL_RUN = "featuretable20261120a"
REPEAT_RUN = "featuretable20261120b"
LABEL_RUN = "labels20261120a"
MEMBERSHIP_RUN = "membership20261120a"
LEAKAGE_RUN = "p44audit20261120a"
REPAIR_RUN = "repair20261115f"
DEFAULT_RUNS = {"membership": MEMBERSHIP_RUN, "labels": LABEL_RUN,
                "model_table": MODEL_RUN, "model_repeat": REPEAT_RUN,
                "leakage_audit": LEAKAGE_RUN, "repaired_events": REPAIR_RUN}
MIN_TEST_EVENTS = 50
GAP_QUARTERS = 4
MAX_Q = date(2025, 6, 30)
FROZEN_DATES = {"development_start": date(1984, 3, 31),
                "development_end": date(2012, 12, 31),
                "gap_start": date(2013, 3, 31), "gap_end": date(2013, 12, 31),
                "test_start": date(2014, 3, 31), "test_end": date(2025, 6, 30)}
ASSIGNMENT_SCHEMA = pa.schema([
    pa.field("CERT", pa.int64(), nullable=False),
    pa.field("Q", pa.date32(), nullable=False),
    pa.field("PERIOD", pa.string(), nullable=False),
    pa.field("Y", pa.int8(), nullable=False),
], metadata={b"bft65b.split": VERSION.encode()})
LINKAGE_SCHEMA = pa.schema([
    pa.field("CERT", pa.int64(), nullable=False),
    pa.field("Q", pa.date32(), nullable=False),
    pa.field("SOURCE_ID", pa.string(), nullable=False),
    pa.field("FAILDATE", pa.date32(), nullable=False),
    pa.field("PERIOD", pa.string(), nullable=False),
], metadata={b"bft65b.split": VERSION.encode()})


def quarter_index(q: date) -> int:
    if (q.month, q.day) not in {(3, 31), (6, 30), (9, 30), (12, 31)}:
        raise ValueError(f"Not a quarter-end: {q}")
    return q.year * 4 + q.month // 3 - 1


def choose_periods(quarters: list[date], events_by_quarter: dict[date, set[str]],
                   *, min_events: int = MIN_TEST_EVENTS,
                   gap_quarters: int = GAP_QUARTERS) -> dict[str, date | int]:
    """Choose the shortest recent contiguous test block meeting event coverage."""
    if (not quarters or quarters != sorted(set(quarters)) or quarters[-1] > MAX_Q
            or min_events < 1 or gap_quarters < 4
            or any(quarter_index(b) - quarter_index(a) != 1
                   for a, b in zip(quarters, quarters[1:]))):
        raise ValueError("Prediction quarter inventory or split rule is invalid")
    test_events: set[str] = set()
    start = None
    for i in range(len(quarters) - 1, gap_quarters, -1):
        test_events.update(events_by_quarter.get(quarters[i], set()))
        if len(test_events) >= min_events:
            start = i
            break
    if start is None:
        raise ValueError("No chronological split has the required distinct test failures and four-quarter gap")
    dev_end = quarters[start - gap_quarters - 1]
    gap_start = quarters[start - gap_quarters]
    gap_end = quarters[start - 1]
    test_start = quarters[start]
    if (quarter_index(test_start) - quarter_index(dev_end) - 1 != gap_quarters
            or date(dev_end.year + 1, dev_end.month, dev_end.day) >= test_start):
        raise ValueError("Development target window overlaps test prediction start")
    development_events = set().union(*(events_by_quarter.get(q, set())
                                       for q in quarters[:start-gap_quarters]))
    if development_events & test_events:
        raise ValueError("A qualifying failure event appears in both development and test positives")
    return {"development_start": quarters[0], "development_end": dev_end,
            "gap_start": gap_start, "gap_end": gap_end,
            "test_start": test_start, "test_end": quarters[-1],
            "development_target_right_end": date(dev_end.year + 1, dev_end.month, dev_end.day),
            "gap_quarters": gap_quarters, "minimum_test_events": min_events,
            "test_distinct_events": len(test_events)}


def period_for(q: date, bounds: dict) -> str:
    if bounds["development_start"] <= q <= bounds["development_end"]:
        return "development"
    if bounds["gap_start"] <= q <= bounds["gap_end"]:
        return "gap"
    if bounds["test_start"] <= q <= bounds["test_end"]:
        return "test"
    raise ValueError(f"Prediction quarter is outside frozen periods: {q}")


def validate_positive(cert: int, q: date, ids: list[str], event_by_id: dict) -> tuple[str, date]:
    if len(ids) != 1 or ids[0] not in event_by_id:
        raise ValueError(f"Positive key has missing, duplicate or unknown source ID: {cert}, {q}")
    event = event_by_id[ids[0]]
    right = date(q.year + 1, q.month, q.day)
    if event.cert != cert or not _qualifies(event) or not q < event.effective_date <= right:
        raise ValueError(f"Positive source ID fails CERT, verified transaction or target window: {cert}, {q}, {ids[0]}")
    return event.source_id, event.effective_date


def _write_json_atomic(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha(path)


def _accepted(root: Path, selected: dict[str, str] | None = None) -> tuple[dict, dict, dict, dict, dict[str, str]]:
    """Invoke accepted validators and freeze hashes of directly consumed files."""
    selected = dict(DEFAULT_RUNS if selected is None else selected)
    if set(selected) != set(DEFAULT_RUNS) or any(not value for value in selected.values()):
        raise ValueError("Split upstream run selection is incomplete")
    portable = (root / "runs/source_vintage_bundle.json").is_file()
    if portable and any(selected[key] == default for key, default in DEFAULT_RUNS.items()):
        raise ValueError("Portable split requires explicit nonhistorical upstream runs")
    verify_membership(root, selected["membership"])
    verify_labels(root, selected["labels"])
    verify_table(root, selected["model_table"], check_upstream=True)
    verify_table(root, selected["model_repeat"], check_upstream=True)
    verify_leakage(root, selected["leakage_audit"], rehash_inputs=True)
    _, events, repair_hashes = _verify_repair(root, selected["repaired_events"])
    event_by_id = {event.source_id: event for event in events}
    if len(event_by_id) != len(events):
        raise ValueError("Repaired FDIC source IDs are not unique")
    model = _read_json(root / "runs/task_4_3_feature_table" / selected["model_table"] / "summary.json")
    repeat = _read_json(root / "runs/task_4_3_feature_table" / selected["model_repeat"] / "summary.json")
    leakage = _read_json(root / "runs/task_4_4_leakage_audit" / selected["leakage_audit"] / "summary.json")
    labels = _read_json(root / "runs/task_4_2_production_labels" / selected["labels"] / "summary.json")
    if (labels.get("membership_run") != selected["membership"]
            or labels.get("repaired_event_run") != selected["repaired_events"]
            or model.get("source_runs") != repeat.get("source_runs")
            or model.get("source_runs", {}).get("membership") != selected["membership"]
            or model.get("source_runs", {}).get("labels") != selected["labels"]
            or (portable and (leakage.get("source_runs", {}).get("feature_table") != selected["model_table"]
                or leakage.get("source_runs", {}).get("independent_repeat") != selected["model_repeat"]
                or leakage.get("source_runs", {}).get("membership") != selected["membership"]
                or leakage.get("source_runs", {}).get("labels") != selected["labels"]))):
        raise ValueError("Split upstream lineage is inconsistent")
    model_items = {item["quarter"]: item for item in model["outputs"]
                   if item["dataset"] == "model_table"}
    label_items = {item["quarter"]: item for item in labels["outputs"]}
    if (len(model_items) != 166 or set(model_items) != set(label_items)
            or max(model_items) != "20250630"
            or labels.get("outcome_complete_through") != "2026-06-30"):
        raise ValueError("Accepted modelling or label quarter scope differs")
    hashes = {str(Path(item["path"])): item["sha256"]
              for item in list(model_items.values()) + list(label_items.values())}
    hashes.update(repair_hashes)
    for path in (root / "runs/task_4_3_feature_table" / selected["model_table"] / "summary.json",
                 root / "runs/task_4_3_feature_table" / selected["model_repeat"] / "summary.json",
                 root / "runs/task_4_2_production_labels" / selected["labels"] / "summary.json",
                 root / "runs/task_4_2_membership" / selected["membership"] / "summary.json",
                 root / "runs/task_4_4_leakage_audit" / selected["leakage_audit"] / "summary.json"):
        hashes[str(path)] = _sha(path)
    if portable:
        for name in ("source_vintage_bundle.json", "membership_source_bundle_v1.json"):
            path = root / "runs" / name
            hashes[str(path)] = _sha(path)
        if any(not Path(path).resolve().is_relative_to(root) for path in hashes):
            raise ValueError("Portable split input escaped selected root")
    return model_items, label_items, event_by_id, model, hashes


def _quarter_records(model_item: dict, label_item: dict, event_by_id: dict) -> tuple[list[dict], list[dict]]:
    q = _quarter(model_item["quarter"])
    if label_item["quarter"] != model_item["quarter"]:
        raise ValueError("Model and label quarters differ")
    with pq.ParquetFile(model_item["path"]) as file:
        model = file.read(columns=["CERT", "Q", "Y"]).to_pylist()
    with pq.ParquetFile(label_item["path"]) as file:
        labels = file.read(columns=["CERT", "Q", "RIGHT_ENDPOINT", "MEMBERSHIP",
            "FEATURE_HISTORY_ELIGIBLE", "LABEL_DISPOSITION", "Y",
            "QUALIFYING_SOURCE_IDS"]).to_pylist()
    if len(model) != model_item["rows"] or len(labels) != label_item["rows"]:
        raise ValueError(f"Accepted row count differs at {q}")
    label_by_cert = {row["CERT"]: row for row in labels}
    if len(label_by_cert) != len(labels):
        raise ValueError(f"Duplicate label keys at {q}")
    seen = set()
    positives = []
    for row in model:
        cert, y = row["CERT"], row["Y"]
        if cert in seen or row["Q"] != q or y not in (0, 1):
            raise ValueError(f"Duplicate, wrong-quarter or invalid model key at {q}")
        seen.add(cert)
        label = label_by_cert.get(cert)
        expected = "positive" if y == 1 else "negative"
        if (label is None or label["Q"] != q
                or label["RIGHT_ENDPOINT"] != date(q.year + 1, q.month, q.day)
                or label["RIGHT_ENDPOINT"] > date(2026, 6, 30)
                or label["MEMBERSHIP"] != "eligible"
                or label["FEATURE_HISTORY_ELIGIBLE"] is not True
                or label["LABEL_DISPOSITION"] != expected or label["Y"] != y):
            raise ValueError(f"Model/label membership, disposition or follow-up differs: {cert}, {q}")
        if y == 1:
            source_id, faildate = validate_positive(cert, q, label["QUALIFYING_SOURCE_IDS"], event_by_id)
            positives.append({"CERT": cert, "Q": q, "SOURCE_ID": source_id,
                              "FAILDATE": faildate})
        elif label["QUALIFYING_SOURCE_IDS"]:
            raise ValueError(f"Negative has qualifying failure source ID: {cert}, {q}")
    return model, positives


def run(root: Path, run_id: str, *, selected: dict[str, str] | None = None) -> dict:
    root = _root(root)
    _run_id(run_id)
    evidence = root / "runs" / NAMESPACE / run_id
    data = root / "data_processed" / NAMESPACE / run_id
    if evidence.exists() or data.exists():
        raise FileExistsError(f"Split run ID already exists: {run_id}")
    selected = dict(DEFAULT_RUNS if selected is None else selected)
    model_items, label_items, event_by_id, model_summary, input_hashes = _accepted(root, selected)
    quarters = [_quarter(q) for q in sorted(model_items)]
    manifest = RunManifest.new(run_id, load_config(root / "config/financials_full_v2.json"))
    manifest.config["split_policy_version"] = VERSION
    manifest.config["source_runs"] = selected
    bundle_path = root / "runs/source_vintage_bundle.json"
    manifest.config["source_vintage"] = (
        _read_json(bundle_path)["attempt"] if bundle_path.is_file()
        else "real20261111b")
    manifest.config["selection_rule"] = "Shortest recent contiguous test through 2025Q2 with >=50 distinct linked failure source IDs; four intervening quarters"
    for path, digest in sorted(input_hashes.items()):
        manifest.record_input(path, digest)
    manifest.set_status("running")
    manifest.write(evidence / "run_manifest.json")
    try:
        quarter_stats = []
        event_sets = {}
        positive_links = []
        for q in quarters:
            qtext = q.strftime("%Y%m%d")
            model_rows, positives = _quarter_records(model_items[qtext], label_items[qtext], event_by_id)
            ids = {row["SOURCE_ID"] for row in positives}
            event_sets[q] = ids
            positive_links.extend(positives)
            quarter_stats.append({"quarter": qtext, "model_rows": len(model_rows),
                                  "positive_rows": len(positives),
                                  "negative_rows": len(model_rows) - len(positives),
                                  "distinct_positive_banks": len({row["CERT"] for row in positives}),
                                  "distinct_qualifying_events": len(ids)})
        if (sum(item["model_rows"] for item in quarter_stats) !=
                model_summary["overall"]["model_positive"] + model_summary["overall"]["model_negative"]
                or len(positive_links) != model_summary["overall"]["model_positive"]):
            raise ValueError("Model total or positive count differs from accepted P4.3 evidence")
        bounds = choose_periods(quarters, event_sets)
        if any(bounds[key] != value for key, value in FROZEN_DATES.items()):
            raise ValueError("Measured split differs from the accepted frozen 1984Q1–2025Q2 boundaries")
        rows = []
        positive_by_period: dict[str, list[dict]] = defaultdict(list)
        period_banks: dict[str, set[int]] = defaultdict(set)
        period_counts: dict[str, Counter] = defaultdict(Counter)
        for q, stats in zip(quarters, quarter_stats):
            period = period_for(q, bounds)
            stats["period"] = period
            for row in pq.ParquetFile(model_items[stats["quarter"]]["path"]).read(
                    columns=["CERT", "Q", "Y"]).to_pylist():
                rows.append({**row, "PERIOD": period})
                period_banks[period].add(row["CERT"])
                period_counts[period]["positive_rows" if row["Y"] else "negative_rows"] += 1
        for row in positive_links:
            period = period_for(row["Q"], bounds)
            positive_by_period[period].append({**row, "PERIOD": period})
        if len(rows) != sum(item["model_rows"] for item in quarter_stats):
            raise ValueError("Split assignments do not reconcile to input keys")
        period_summary = {}
        for period in ("development", "gap", "test"):
            positives = positive_by_period[period]
            period_summary[period] = {
                **dict(period_counts[period]),
                "model_rows": sum(period_counts[period].values()),
                "distinct_model_banks": len(period_banks[period]),
                "distinct_positive_banks": len({row["CERT"] for row in positives}),
                "distinct_qualifying_events": len({row["SOURCE_ID"] for row in positives}),
                "prediction_quarters": sum(item["period"] == period for item in quarter_stats),
            }
        if (period_summary["test"]["distinct_qualifying_events"] < MIN_TEST_EVENTS
                or set(row["SOURCE_ID"] for row in positive_by_period["development"])
                   & set(row["SOURCE_ID"] for row in positive_by_period["test"])):
            raise ValueError("Test failure gate or development/test event isolation failed")
        assignment = pa.Table.from_pylist(rows, schema=ASSIGNMENT_SCHEMA)
        linkage = pa.Table.from_pylist([
            row for period in ("development", "gap", "test")
            for row in positive_by_period[period]], schema=LINKAGE_SCHEMA)
        outputs = []
        for name, table in (("assignments", assignment), ("positive_linkage", linkage)):
            path = data / f"{name}.parquet"
            digest = _write_verified(path, table)
            item = {"dataset": name, "path": str(path), "rows": table.num_rows,
                    "sha256": digest}
            outputs.append(item)
            manifest.record_output(path=str(path), dataset=name, rows=table.num_rows,
                sha256=digest, compression="ZSTD", schema_version=VERSION,
                partition=None, source_total=table.num_rows, source_query=None)
        if any(_sha(Path(path)) != digest for path, digest in input_hashes.items()):
            raise ValueError("Accepted upstream input changed during split construction")
        summary = {"status": "complete", "run_id": run_id, "version": VERSION,
                   "source_runs": selected,
                   "selection_rule": manifest.config["selection_rule"],
                   "bounds": {key: str(value) if isinstance(value, date) else value
                              for key, value in bounds.items()},
                   "by_quarter": quarter_stats, "periods": period_summary,
                   "outputs": outputs, "input_hashes": input_hashes,
                   "excluded_from_fitting": {"gap_rows": period_summary["gap"]["model_rows"],
                       "outside_accepted_model_table": model_summary["overall"]["label_keys"] -
                           model_summary["overall"]["model_positive"] -
                           model_summary["overall"]["model_negative"],
                       "upstream_exclusion_reasons": model_summary["exclusion_reasons"]},
                   "test_label_access": "Global outcome counts and FDIC source IDs inspected solely for chronological split feasibility; no feature values or predictive test scores inspected",
                   "created_at_utc": datetime.now(timezone.utc).isoformat()}
        _write_json_atomic(evidence / "summary.json", summary)
        _write_json_atomic(evidence / "metadata.json", {
            "status": "complete", "summary_sha256": _sha(evidence / "summary.json"),
            "output_hashes": {item["dataset"]: item["sha256"] for item in outputs}})
        manifest.record_validation("accepted_upstream_manifests_and_hashes", True,
            f"{len(input_hashes)} accepted source/evidence files verified before and after")
        manifest.record_validation("verified_event_linkage_and_target_window", True,
            f"{len(positive_links)} positive bank-quarter rows linked to qualifying FDIC IDs")
        manifest.record_validation("chronological_gap_and_test_failure_count", True,
            f"Four intervening quarters; {period_summary['test']['distinct_qualifying_events']} distinct test events")
        manifest.record_validation("atomic_output_readback_and_reconciliation", True,
            f"{len(rows)} assignments and {len(positive_links)} positive links read back")
        manifest.set_status("complete")
        manifest.write(evidence / "run_manifest.json")
        verify_saved(root, run_id, rehash_inputs=True)
        return {"status": "complete", "run_id": run_id, "bounds": summary["bounds"],
                "periods": period_summary, "evidence": str(evidence)}
    except Exception as exc:
        manifest.record_validation("split_failure", False, str(exc))
        manifest.set_status("failed")
        manifest.write(evidence / "run_manifest.json")
        raise


def verify_saved(root: Path, run_id: str, *, rehash_inputs: bool = True) -> dict:
    root = _root(root)
    _run_id(run_id)
    evidence = root / "runs" / NAMESPACE / run_id
    manifest = RunManifest.read(evidence / "run_manifest.json")
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    if (manifest.status != "complete" or not manifest.validations
            or any(not item["passed"] for item in manifest.validations)
            or summary.get("status") != "complete" or summary.get("run_id") != run_id
            or summary.get("version") != VERSION or metadata.get("status") != "complete"
            or metadata.get("summary_sha256") != _sha(evidence / "summary.json")
            or len(summary["by_quarter"]) != 166 or len(summary["outputs"]) != 2
            or {item["path"]: item["sha256"] for item in manifest.inputs}
               != summary["input_hashes"]):
        raise ValueError("Split manifest, summary or metadata is incomplete")
    selected = summary.get("source_runs")
    if (not isinstance(selected, dict) or set(selected) != set(DEFAULT_RUNS)
            or manifest.config.get("source_runs", selected) != selected):
        raise ValueError("Split saved upstream lineage is inconsistent")
    if (root / "runs/source_vintage_bundle.json").is_file():
        if any(selected[key] == default for key, default in DEFAULT_RUNS.items()):
            raise ValueError("Portable split cannot reference historical default runs")
        if any(not Path(path).resolve().is_relative_to(root)
               for path in summary["input_hashes"]):
            raise ValueError("Portable split saved input escaped root")
    total = Counter()
    keys = set()
    positive_keys = set()
    linked_keys = set()
    bounds = {key: _quarter(value.replace("-", "")) for key, value in summary["bounds"].items()
              if key.endswith(("start", "end")) and isinstance(value, str)}
    for item, schema in zip(summary["outputs"], (ASSIGNMENT_SCHEMA, LINKAGE_SCHEMA)):
        path = Path(item["path"])
        if (not path.resolve().is_relative_to((root / "data_processed" / NAMESPACE / run_id).resolve())
                or _sha(path) != item["sha256"]
                or metadata["output_hashes"].get(item["dataset"]) != item["sha256"]):
            raise ValueError(f"Split output checksum or path differs: {path}")
        with pq.ParquetFile(path) as file:
            if file.schema_arrow != schema or file.metadata.num_rows != item["rows"]:
                raise ValueError(f"Split output schema or row count differs: {path}")
            if any(file.metadata.row_group(g).column(c).compression != "ZSTD"
                   for g in range(file.metadata.num_row_groups)
                   for c in range(file.metadata.row_group(g).num_columns)):
                raise ValueError(f"Split output codec differs: {path}")
            rows = file.read().to_pylist()
        for row in rows:
            total[item["dataset"]] += 1
            if item["dataset"] == "assignments":
                key = (row["CERT"], row["Q"])
                if (key in keys or row["Y"] not in (0, 1)
                        or row["PERIOD"] != period_for(row["Q"], bounds)):
                    raise ValueError("Duplicate or inconsistent saved split key")
                keys.add(key)
                if row["Y"] == 1:
                    positive_keys.add(key)
            else:
                key = (row["CERT"], row["Q"])
                if (key in linked_keys or row["PERIOD"] != period_for(row["Q"], bounds)
                        or not row["Q"] < row["FAILDATE"] <=
                            date(row["Q"].year + 1, row["Q"].month, row["Q"].day)):
                    raise ValueError("Duplicate or inconsistent saved positive link")
                linked_keys.add(key)
        if total[item["dataset"]] != item["rows"]:
            raise ValueError("Split output count differs on read-back")
    if (positive_keys != linked_keys
            or total["assignments"] != sum(row["model_rows"] for row in summary["by_quarter"])
            or total["positive_linkage"] != sum(row["positive_rows"] for row in summary["by_quarter"])
            or sum(p["model_rows"] for p in summary["periods"].values()) != total["assignments"]
            or summary["periods"]["test"]["distinct_qualifying_events"] < MIN_TEST_EVENTS):
        raise ValueError("Split period totals or test failure gate differ")
    if rehash_inputs and any(_sha(Path(path)) != digest
                             for path, digest in summary["input_hashes"].items()):
        raise ValueError("Accepted upstream checksum changed after split")
    if rehash_inputs and (root / "runs/source_vintage_bundle.json").is_file():
        *_, current_hashes = _accepted(root, selected)
        if current_hashes != summary["input_hashes"]:
            raise ValueError("Portable split upstream lineage or file inventory changed")
    return {"status": "complete", "run_id": run_id, "bounds": summary["bounds"],
            "periods": summary["periods"], "outputs": len(summary["outputs"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a chronological FDIC failure split")
    parser.add_argument("mode", choices=("run", "verify", "show"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    for key in DEFAULT_RUNS:
        parser.add_argument("--" + key.replace("_", "-"))
    args = parser.parse_args()
    if args.mode == "run":
        selected = {key: getattr(args, key) for key in DEFAULT_RUNS}
        if any(selected.values()) and not all(selected.values()):
            parser.error("All split upstream run IDs are required when selecting a replay chain")
        result = run(args.project_root, args.run_id,
                     selected=selected if any(selected.values()) else None)
    elif args.mode == "verify":
        result = verify_saved(args.project_root, args.run_id)
    else:
        result = verify_saved(args.project_root, args.run_id, rehash_inputs=False)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
