"""Development-only FDIC estimated-loss audit; no threshold or test scoring.

Amounts remain in nominal thousands of US dollars. Scenario parameters are
declared assumptions, separate from observed FDIC event-level quantities.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from bft65b.config import load_config
from bft65b.features.label import _verify_repair
from bft65b.manifest import RunManifest
from bft65b.modelling.development import verify_saved as verify_development
from bft65b.modelling.split import _write_json_atomic, verify_saved as verify_split
from bft65b.panel.build import _write_verified
from bft65b.panel.retrospective import _root, _run_id
from bft65b.quality.audit import _read_json, _sha


VERSION = "development_cost_audit_v1"
NAMESPACE = "task_5_4_cost"
SPLIT_RUN = "split20261120b"
DEVELOPMENT_RUNS = ("dev20261121a", "dev20261121b")
REPAIR_RUN = "repair20261115f"
DEFAULT_RUNS = {"split": SPLIT_RUN, "development_primary": DEVELOPMENT_RUNS[0],
                "development_repeat": DEVELOPMENT_RUNS[1], "repair": REPAIR_RUN}
SELECTED_CANDIDATE = "bounded_hist_gradient_boosting"
SOURCE_UNIT = "nominal_thousands_usd"
EVENT_SCHEMA = pa.schema([
    pa.field("SOURCE_ID", pa.string(), nullable=False),
    pa.field("CERT", pa.int64(), nullable=False),
    pa.field("FAILDATE", pa.date32(), nullable=False),
    pa.field("COST", pa.float64()),
    pa.field("COST_STATUS", pa.string(), nullable=False),
    pa.field("SOURCE_FIELD_ABSENT", pa.bool_(), nullable=False),
    pa.field("DEVELOPMENT_POSITIVE_ROWS", pa.int32(), nullable=False),
    pa.field("OOF_POSITIVE_ROWS", pa.int32(), nullable=False),
], metadata={b"bft65b.cost_audit": VERSION.encode(), b"cost_unit": SOURCE_UNIT.encode()})


def thousand_usd_to_usd(value: int | float | Decimal) -> Decimal:
    """Convert a finite source amount without rounding or guessing its unit."""
    amount = Decimal(str(value))
    if not amount.is_finite():
        raise ValueError("Nonfinite COST cannot be converted")
    return amount * Decimal(1000)


def cost_status(value: object, *, absent: bool = False) -> str:
    if absent:
        return "source_absent"
    if value is None:
        return "missing"
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return "invalid_or_special"
    number = float(value)
    if not math.isfinite(number):
        return "invalid_or_special"
    return "negative" if number < 0 else "zero" if number == 0 else "positive"


def _event_period(day: date) -> str:
    if day.year < 1986:
        return "1984-1985"
    if day.year <= 1993:
        return "1986-1993"
    if day.year <= 2007:
        return "1994-2007"
    return "2008-2013"


def scenario_spec() -> dict:
    """Freeze dimensionless assumptions before any threshold selection."""
    return {
        "version": "cost_scenarios_v1",
        "decision_unit": "one bank-quarter alarm",
        "false_alarm_policy": "one investigation charge per alarmed negative bank-quarter; repeated alarms each incur a charge",
        "miss_policy": "row-level surrogate per missed positive bank-quarter; repeated positives for one event repeat this surrogate and must not be summed as literal FDIC loss",
        "reference": {"preventable_fraction": 0.10, "false_alarm_fraction_of_training_median": 0.01},
        "sensitivity": {"preventable_fractions": [0.05, 0.10, 0.25],
                        "false_alarm_fractions_of_training_median": [0.001, 0.01, 0.05]},
        "reference_amount_source": "median finite nonnegative COST from unique verified training events only; nominal thousands USD",
        "missing_cost_policy": "keep event and labels; do not impute raw COST; a training-only representative event cost supplies the simplified loss surrogate",
        "negative_cost_policy": "retain and report; exclude from nonnegative representative statistic without changing raw value",
        "not_observed": ["preventable_fraction", "false_alarm_investigation_cost", "causal_avoided_loss"],
        "threshold_selected": False,
    }


def validate_scenarios(spec: dict) -> None:
    fractions = spec["sensitivity"]["preventable_fractions"]
    alarm = spec["sensitivity"]["false_alarm_fractions_of_training_median"]
    reference = spec["reference"]
    if (not fractions or not alarm or not all(0 <= x <= 1 for x in fractions)
            or not all(x > 0 for x in alarm)
            or reference["preventable_fraction"] not in fractions
            or reference["false_alarm_fraction_of_training_median"] not in alarm
            or spec["threshold_selected"] is not False):
        raise ValueError("Invalid or threshold-dependent cost scenario")


def representative_training_cost(training_ids: set[str], event_rows: list[dict]) -> float:
    """Fit a representative cost using unique training source IDs only."""
    if not training_ids:
        raise ValueError("No training failure events")
    by_id = {row["SOURCE_ID"]: row for row in event_rows}
    if len(by_id) != len(event_rows) or training_ids - by_id.keys():
        raise ValueError("Duplicate or unmatched training source ID")
    usable = [float(by_id[source_id]["COST"]) for source_id in training_ids
              if cost_status(by_id[source_id]["COST"],
                             absent=by_id[source_id].get("SOURCE_FIELD_ABSENT", False))
              in ("zero", "positive")]
    if not usable:
        raise ValueError("Training events have no usable FDIC COST")
    return float(statistics.median(usable))


def fold_training_costs(links: list[dict], event_rows: list[dict], folds: list[dict]) -> list[dict]:
    """Keep descriptive all-development coverage distinct from fold fits."""
    results = []
    for fold in folds:
        start, end = date.fromisoformat(fold["train_start"]), date.fromisoformat(fold["train_end"])
        validation_start = date.fromisoformat(fold["validation_start"])
        training = [row for row in links if row["PERIOD"] == "development"
                    and start <= row["Q"] <= end]
        validation = [row for row in links if row["PERIOD"] == "development"
                      and validation_start <= row["Q"] <= date.fromisoformat(fold["validation_end"])]
        ids = {row["SOURCE_ID"] for row in training}
        if (len(training) != fold["training_positive_rows"]
                or len(ids) != fold["training_distinct_events"]
                or ids & {row["SOURCE_ID"] for row in validation}):
            raise ValueError(f"Frozen fold training counts or event isolation differ: {fold['id']}")
        subset = [row for row in event_rows if row["SOURCE_ID"] in ids]
        if len(subset) != len(ids):
            raise ValueError(f"Missing fold-training cost event: {fold['id']}")
        stats = _describe(subset)
        results.append({"fold_id": fold["id"], "training_end": str(end),
                        "training_positive_rows": len(training),
                        "training_distinct_events": len(ids),
                        "usable_training_events": stats["usable_nonnegative_events"],
                        "training_only_median_nominal_thousands_usd": representative_training_cost(ids, event_rows)})
    return results


def _quantile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    x = (len(sorted_values) - 1) * fraction
    lo = math.floor(x)
    hi = math.ceil(x)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (x - lo)


def _describe(rows: list[dict]) -> dict:
    counts = Counter(row["COST_STATUS"] for row in rows)
    usable = sorted(float(row["COST"]) for row in rows
                    if row["COST_STATUS"] in ("zero", "positive"))
    total = math.fsum(usable)
    positives = sorted((x for x in usable if x > 0), reverse=True)
    positive_total = math.fsum(positives)
    concentration = {}
    for fraction in (0.01, 0.05, 0.10):
        n = math.ceil(len(positives) * fraction)
        concentration[str(fraction)] = (math.fsum(positives[:n]) / positive_total
                                         if positive_total else None)
    return {
        "distinct_events": len(rows), "cost_status_events": dict(sorted(counts.items())),
        "usable_nonnegative_events": len(usable),
        "usable_coverage": len(usable) / len(rows) if rows else None,
        "nominal_thousands_usd": {
            "sum": total if usable else None,
            "mean": total / len(usable) if usable else None,
            "min": usable[0] if usable else None,
            "p25": _quantile(usable, .25), "median": _quantile(usable, .5),
            "p75": _quantile(usable, .75), "p90": _quantile(usable, .90),
            "p95": _quantile(usable, .95), "p99": _quantile(usable, .99),
            "max": usable[-1] if usable else None,
        },
        "positive_cost_concentration_top_fraction": concentration,
        "negative_values_nominal_thousands_usd": sorted(
            float(row["COST"]) for row in rows if row["COST_STATUS"] == "negative"),
    }


def reconcile_costs(links: list[dict], repaired: list[dict], raw: list[dict],
                    folds: list[dict]) -> tuple[list[dict], dict]:
    """Validate identity/period before touching an event's COST for analysis."""
    link_keys = set()
    dev = defaultdict(list)
    nondev = defaultdict(set)
    for row in links:
        key = (row["CERT"], row["Q"])
        if key in link_keys or not row["SOURCE_ID"] or row["PERIOD"] not in {"development", "gap", "test"}:
            raise ValueError("Duplicate or invalid positive linkage key")
        link_keys.add(key)
        if row["PERIOD"] == "development":
            dev[row["SOURCE_ID"]].append(row)
        else:
            nondev[row["PERIOD"]].add(row["SOURCE_ID"])
    if not dev:
        raise ValueError("No development events")
    if set(dev) & nondev["test"]:
        raise ValueError("Development and test share a failure event")
    repair_by_id = {row["ID"]: row for row in repaired}
    raw_by_id = {row["ID"]: row for row in raw}
    if len(repair_by_id) != len(repaired) or len(raw_by_id) != len(raw):
        raise ValueError("Duplicate FDIC source ID")
    if set(dev) - repair_by_id.keys() or set(dev) - raw_by_id.keys():
        raise ValueError("Unmatched development source ID")
    validation_by_q = {}
    for fold in folds:
        start, end = date.fromisoformat(fold["validation_start"]), date.fromisoformat(fold["validation_end"])
        if start > end:
            raise ValueError("Invalid frozen fold")
        validation_by_q[fold["id"]] = (start, end)
    output = []
    for source_id in sorted(dev):
        event, source = repair_by_id[source_id], raw_by_id[source_id]
        if (event["CERT"] != source["CERT"] or event["FAILDATE"] != source["FAILDATE"]
                or event["COST"] != source["COST"]
                or event["TRANSACTION_DISPOSITION"] != "verified_failure"
                or event["BANK_LINK_DISPOSITION"] != "chronological_candidate"
                or event["SUPPLEMENT_MATCH"] != "matched"):
            raise ValueError(f"Conflicting source identity or transaction: {source_id}")
        rows = dev[source_id]
        for link in rows:
            q = link["Q"]
            if (link["CERT"] != source["CERT"] or link["FAILDATE"] != source["FAILDATE"]
                    or not q < source["FAILDATE"] <= date(q.year + 1, q.month, q.day)):
                raise ValueError(f"Source ID, CERT or four-quarter window differs: {source_id}")
        missing_fields = source.get("SOURCE_MISSING_FIELDS") or []
        absent = "COST" in missing_fields
        value = source["COST"]
        status = cost_status(value, absent=absent)
        oof_rows = sum(any(start <= link["Q"] <= end for start, end in validation_by_q.values())
                       for link in rows)
        output.append({"SOURCE_ID": source_id, "CERT": source["CERT"],
                       "FAILDATE": source["FAILDATE"], "COST": value,
                       "COST_STATUS": status, "SOURCE_FIELD_ABSENT": absent,
                       "DEVELOPMENT_POSITIVE_ROWS": len(rows), "OOF_POSITIVE_ROWS": oof_rows})
    if set(raw_by_id) != set(dev):
        raise ValueError("Raw COST selection includes non-development events")
    by_period = {}
    for period in sorted({_event_period(row["FAILDATE"]) for row in output}):
        by_period[period] = _describe([row for row in output if _event_period(row["FAILDATE"]) == period])
    oof = [row for row in output if row["OOF_POSITIVE_ROWS"]]
    return output, {
        "development": _describe(output),
        "by_failure_period": by_period,
        "oof_validation": {**_describe(oof), "positive_bank_quarter_rows": sum(r["OOF_POSITIVE_ROWS"] for r in oof)},
        "positive_bank_quarter_rows": sum(r["DEVELOPMENT_POSITIVE_ROWS"] for r in output),
        "distinct_positive_banks": len({r["CERT"] for r in output}),
        "development_events_with_faildate_after_dev_q_end": sum(r["FAILDATE"] > date(2012, 12, 31) for r in output),
        "gap_only_distinct_events": len(nondev["gap"] - set(dev)),
        "test_only_distinct_events": len(nondev["test"] - set(dev)),
    }


def _paths(root: Path, run_id: str) -> tuple[Path, Path]:
    _run_id(run_id)
    return root / "runs" / NAMESPACE / run_id, root / "data_processed" / NAMESPACE / run_id


def _preflight(root: Path, selected: dict[str, str]) -> tuple[dict, list[dict], list[dict], list[dict], list[dict], dict]:
    portable = (root / "runs/source_vintage_bundle.json").is_file()
    if set(selected) != set(DEFAULT_RUNS) or any(not value for value in selected.values()):
        raise ValueError("All COST upstream run IDs are required")
    if portable and any(selected[key] == default for key, default in DEFAULT_RUNS.items()):
        raise ValueError("Portable COST audit cannot use historical default upstream runs")
    verify_split(root, selected["split"], rehash_inputs=True)
    for run_id in (selected["development_primary"], selected["development_repeat"]):
        result = verify_development(root, run_id, rehash_inputs=True)
        if result["selected_candidate"] != SELECTED_CANDIDATE:
            raise ValueError("Accepted development selection changed")
    _verify_repair(root, selected["repair"])
    split_root = root / "runs/task_5_1_split" / selected["split"]
    dev_root = root / "runs/task_5_2_development" / selected["development_primary"]
    split = _read_json(split_root / "summary.json")
    frozen = _read_json(dev_root / "frozen_plan.json")
    repeat_frozen = _read_json(root / "runs/task_5_2_development" /
                               selected["development_repeat"] / "frozen_plan.json")
    if (frozen.get("split_run") != selected["split"]
            or repeat_frozen.get("split_run") != selected["split"]
            or frozen.get("folds") != repeat_frozen.get("folds")
            or frozen.get("candidates") != repeat_frozen.get("candidates")
            or (portable and split.get("source_runs", {}).get("repaired_events") != selected["repair"])):
        raise ValueError("COST split, development and repair lineage is inconsistent")
    if (split["bounds"]["development_end"] != "2012-12-31"
            or split["bounds"]["development_target_right_end"] != "2013-12-31"
            or split["bounds"]["gap_start"] != "2013-03-31"
            or split["bounds"]["test_start"] != "2014-03-31"):
        raise ValueError("Accepted split bounds changed")
    link_path = next(Path(x["path"]) for x in split["outputs"] if x["dataset"] == "positive_linkage")
    links = pq.read_table(link_path).to_pylist()
    ids = {row["SOURCE_ID"] for row in links if row["PERIOD"] == "development"}
    repair_path = root / "data_processed/task_2_3_transaction_repair" / selected["repair"] / "failure_events/part-000.parquet"
    raw_path = root / "data_raw/task_1_5_full/fdic_extended_v2/failures.parquet"
    fields = ["ID", "CERT", "FAILDATE", "COST", "TRANSACTION_DISPOSITION", "BANK_LINK_DISPOSITION", "SUPPLEMENT_MATCH"]
    repaired = ds.dataset(repair_path).to_table(columns=fields, filter=pc.field("ID").isin(sorted(ids))).to_pylist()
    raw = ds.dataset(raw_path).to_table(columns=["ID", "CERT", "FAILDATE", "COST", "SOURCE_MISSING_FIELDS"],
                                       filter=pc.field("ID").isin(sorted(ids))).to_pylist()
    capture_path = root / "runs/task_2_3_transaction_repair/source_capture.json"
    capture = _read_json(capture_path)
    dictionary_path = root / "runs/task_2_3_transaction_repair/failure_properties_20261115.yaml"
    dictionary = dictionary_path.read_text(encoding="utf-8")
    if (_sha(dictionary_path) != capture["dictionary_sha256"]
            or "x-number-unit: Thousands of US Dollars" not in dictionary
            or "title: Estimated Loss" not in dictionary):
        raise ValueError("Official saved COST definition or checksum changed")
    input_paths = [link_path, repair_path, raw_path, capture_path, dictionary_path,
                   split_root / "summary.json", split_root / "run_manifest.json",
                   dev_root / "frozen_plan.json", dev_root / "summary.json"]
    for run_id in (selected["development_primary"], selected["development_repeat"]):
        input_paths.extend([root / "runs/task_5_2_development" / run_id / "metadata.json",
                            root / "runs/task_5_2_development" / run_id / "run_manifest.json"])
    if portable:
        input_paths.extend([root / "runs/source_vintage_bundle.json",
                            root / "runs/task_2_3_transaction_repair" / selected["repair"] / "summary.json",
                            root / "runs/task_5_2_development" / selected["development_repeat"] / "frozen_plan.json"])
        if any(not path.resolve().is_relative_to(root) for path in input_paths):
            raise ValueError("Portable COST input path escaped the selected root")
    hashes = {str(path): _sha(path) for path in input_paths}
    return split, frozen["folds"], links, repaired, raw, {"hashes": hashes, "capture": capture,
                                                          "source_runs": selected}


def _report(summary: dict) -> str:
    rows = ["# Development FDIC COST audit", "", "Unit: nominal thousands of US dollars. Event-level counts use unique FDIC source IDs; observations are bank-quarters.", "", "| Population | Positive bank-quarters | Distinct events | Usable nonnegative COST | Coverage | Median COST (nominal $000) |", "|---|---:|---:|---:|---:|---:|"]
    for name, stats, count in (("All development", summary["measured"]["development"], summary["measured"]["positive_bank_quarter_rows"]),
                               ("Selected-model OOF validation", summary["measured"]["oof_validation"], summary["measured"]["oof_validation"]["positive_bank_quarter_rows"])):
        rows.append(f"| {name} | {count:,} | {stats['distinct_events']:,} | {stats['usable_nonnegative_events']:,} | {100*stats['usable_coverage']:.2f}% | {stats['nominal_thousands_usd']['median']:,.3f} |")
    rows.extend(["", "| Failure-date period | Distinct events | Missing | Zero | Positive | Negative | Invalid/special | Usable coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for period, stats in summary["measured"]["by_failure_period"].items():
        counts = stats["cost_status_events"]
        rows.append(f"| {period} | {stats['distinct_events']:,} | {counts.get('missing',0)+counts.get('source_absent',0):,} | {counts.get('zero',0):,} | {counts.get('positive',0):,} | {counts.get('negative',0):,} | {counts.get('invalid_or_special',0):,} | {100*stats['usable_coverage']:.2f}% |")
    rows.extend(["", "FDIC defines COST as estimated DIF disbursements to insured depositors less expected receivership recoveries. The published dictionary notes incomplete estimated-loss data before 1986 and for FSLIC-insured failures in 1934–1988. Estimates can change with recoveries; these are neither final realized losses nor total social loss. No inflation adjustment is applied.", "", "The reference preventable fraction (0.10) and investigation-cost ratio (0.01 of the training-event median) are illustrative assumptions, not observed monetary costs. The full grid is frozen in scenario.json. Each bank-quarter alarm incurs an assumed investigation charge. A missed positive bank-quarter receives a simplified cost surrogate; repeated quarters for one event therefore must not be summed as literal FDIC losses. No threshold is selected here.", "", "Raw missing COST remains null; negative and invalid/special values remain visible and are excluded only from the nonnegative representative statistic. Fold-level cost statistics must be fitted on unique training events. Current-vintage COST is retrospective outcome evidence and never a predictor available at historical q.", ""])
    return "\n".join(rows)


def _write_text_atomic(path: Path, value: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha(path)


def run(root: Path, run_id: str, selected: dict[str, str] | None = None) -> dict:
    root = _root(root)
    evidence, data = _paths(root, run_id)
    if evidence.exists() or data.exists():
        raise FileExistsError(f"COST audit run already exists: {run_id}")
    if selected is None:
        selected = DEFAULT_RUNS.copy()
    split, folds, links, repaired, raw, provenance = _preflight(root, selected)
    events, measured = reconcile_costs(links, repaired, raw, folds)
    measured["fold_training_costs"] = fold_training_costs(links, events, folds)
    expected = split["periods"]["development"]
    if (measured["development"]["distinct_events"] != expected["distinct_qualifying_events"]
            or measured["positive_bank_quarter_rows"] != expected["positive_rows"]):
        raise ValueError("Development event/observation counts do not match accepted split")
    spec = scenario_spec()
    validate_scenarios(spec)
    manifest = RunManifest.new(run_id, load_config(root / "config/financials_full_v2.json"))
    manifest.config.update({"cost_audit_version": VERSION, "source_unit": SOURCE_UNIT,
                            "source_runs": selected,
                            "source_vintage": _read_json(root / "runs/source_vintage_bundle.json")["attempt"]
                                if (root / "runs/source_vintage_bundle.json").is_file() else None,
                            "selected_candidate": SELECTED_CANDIDATE})
    for path, digest in sorted(provenance["hashes"].items()):
        manifest.record_input(path, digest)
    manifest.set_status("running")
    manifest.write(evidence / "run_manifest.json")
    try:
        output_path = data / "development_events.parquet"
        table = pa.Table.from_pylist(events, schema=EVENT_SCHEMA)
        output_hash = _write_verified(output_path, table)
        manifest.record_output(path=str(output_path), dataset="development_cost_events",
                               rows=len(events), sha256=output_hash, compression="zstd",
                               schema_version=VERSION, partition=None, source_total=len(events), source_query=None)
        scenario_path = evidence / "scenario.json"
        scenario_hash = _write_json_atomic(scenario_path, spec)
        summary = {"status": "complete", "version": VERSION, "run_id": run_id,
                   "source": {"definition": "Estimated Loss", "unit": SOURCE_UNIT,
                              "dictionary_url": provenance["capture"]["dictionary_url"],
                              "dictionary_sha256": provenance["capture"]["dictionary_sha256"],
                              "retrieved_utc": provenance["capture"]["retrieved_utc"],
                              "source_index": provenance["capture"]["index"]},
                   "periods": split["bounds"], "measured": measured,
                   "scenario_sha256": scenario_hash,
                   "output": {"path": str(output_path), "rows": len(events), "sha256": output_hash},
                   "input_hashes": provenance["hashes"],
                   "source_runs": selected,
                   "environment": {"python": platform.python_version(), "os": platform.platform(),
                                   "pyarrow": pa.__version__},
                   "test_predictors_opened": 0, "test_cost_events_used": 0,
                   "threshold_selected": False}
        summary_hash = _write_json_atomic(evidence / "summary.json", summary)
        report_path = evidence / "report.md"
        report_hash = _write_text_atomic(report_path, _report(summary))
        _write_json_atomic(evidence / "metadata.json", {"status": "complete",
             "summary_sha256": summary_hash, "report_sha256": report_hash,
             "scenario_sha256": scenario_hash, "output_sha256": output_hash})
        manifest.record_validation("development_only_event_identity", True,
                                   f"{len(events)} distinct development source IDs")
        manifest.record_validation("raw_repaired_cost_parity", True)
        manifest.record_validation("source_dictionary_and_units", True)
        manifest.record_validation("atomic_output_readback", True)
        manifest.record_validation("upstream_hashes_unchanged", all(
            _sha(Path(path)) == digest for path, digest in provenance["hashes"].items()))
        manifest.set_status("complete")
        manifest.write(evidence / "run_manifest.json")
        return verify_saved(root, run_id)
    except Exception as exc:
        manifest.record_validation("cost_audit_failure", False, str(exc))
        manifest.set_status("failed")
        manifest.write(evidence / "run_manifest.json")
        raise


def verify_saved(root: Path, run_id: str, *, rehash_inputs: bool = True) -> dict:
    root = _root(root)
    evidence, data = _paths(root, run_id)
    manifest = RunManifest.read(evidence / "run_manifest.json")
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    selected = summary.get("source_runs", DEFAULT_RUNS)
    if (set(selected) != set(DEFAULT_RUNS)
            or manifest.config.get("source_runs", selected) != selected
            or ((root / "runs/source_vintage_bundle.json").is_file()
                and any(selected[key] == default for key, default in DEFAULT_RUNS.items()))):
        raise ValueError("Saved COST upstream lineage is inconsistent")
    spec_path = evidence / "scenario.json"
    spec = _read_json(spec_path)
    validate_scenarios(spec)
    if spec != scenario_spec():
        raise ValueError("Saved COST scenario differs from frozen version")
    output = summary["output"]
    path = Path(output["path"])
    if (manifest.status != "complete" or not manifest.validations
            or not all(check["passed"] for check in manifest.validations)
            or summary["status"] != "complete" or summary["version"] != VERSION
            or summary["run_id"] != run_id or metadata["status"] != "complete"
            or path.resolve() != (data / "development_events.parquet").resolve()
            or _sha(path) != output["sha256"] or metadata["output_sha256"] != output["sha256"]
            or _sha(evidence / "summary.json") != metadata["summary_sha256"]
            or _sha(evidence / "report.md") != metadata["report_sha256"]
            or _sha(spec_path) != metadata["scenario_sha256"]
            or _sha(spec_path) != summary["scenario_sha256"]
            or len(manifest.outputs) != 1 or manifest.outputs[0]["sha256"] != output["sha256"]
            or summary["test_predictors_opened"] != 0 or summary["test_cost_events_used"] != 0
            or summary["threshold_selected"] is not False):
        raise ValueError("COST audit manifest, outputs or completion evidence differs")
    with pq.ParquetFile(path) as file:
        if (file.schema_arrow != EVENT_SCHEMA or file.metadata.num_rows != output["rows"]
                or any(file.metadata.row_group(g).column(c).compression != "ZSTD"
                       for g in range(file.metadata.num_row_groups)
                       for c in range(file.metadata.row_group(g).num_columns))):
            raise ValueError("COST audit Parquet schema, count or codec differs")
        rows = file.read().to_pylist()
    if (len({row["SOURCE_ID"] for row in rows}) != len(rows)
            or _describe(rows) != summary["measured"]["development"]
            or sum(row["DEVELOPMENT_POSITIVE_ROWS"] for row in rows)
               != summary["measured"]["positive_bank_quarter_rows"]):
        raise ValueError("COST audit event-level read-back differs")
    if rehash_inputs and any(_sha(Path(path)) != digest
                             for path, digest in summary["input_hashes"].items()):
        raise ValueError("Accepted COST audit input hash changed")
    if rehash_inputs and (root / "runs/source_vintage_bundle.json").is_file():
        split_summary = _read_json(root / "runs/task_5_1_split" / selected["split"] / "summary.json")
        primary_plan = _read_json(root / "runs/task_5_2_development" /
                                  selected["development_primary"] / "frozen_plan.json")
        repeat_plan = _read_json(root / "runs/task_5_2_development" /
                                 selected["development_repeat"] / "frozen_plan.json")
        if (split_summary.get("source_runs", {}).get("repaired_events") != selected["repair"]
                or primary_plan.get("split_run") != selected["split"]
                or repeat_plan.get("split_run") != selected["split"]
                or primary_plan.get("folds") != repeat_plan.get("folds")
                or primary_plan.get("candidates") != repeat_plan.get("candidates")
                or any(not Path(path).resolve().is_relative_to(root)
                       for path in summary["input_hashes"])):
            raise ValueError("Portable COST upstream lineage changed")
    return {"status": "complete", "run_id": run_id,
            "development_events": len(rows),
            "development_positive_rows": summary["measured"]["positive_bank_quarter_rows"],
            "usable_cost_events": summary["measured"]["development"]["usable_nonnegative_events"],
            "oof_events": summary["measured"]["oof_validation"]["distinct_events"],
            "evidence": str(evidence), "data": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit FDIC COST on unique development failure events")
    parser.add_argument("mode", choices=("run", "verify", "show"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    for key in DEFAULT_RUNS:
        parser.add_argument("--" + key.replace("_", "-"))
    args = parser.parse_args()
    if args.mode == "run":
        selected = {key: getattr(args, key) for key in DEFAULT_RUNS}
        if any(selected.values()) and not all(selected.values()):
            parser.error("All COST upstream run IDs are required when selecting a replay chain")
        result = run(args.project_root, args.run_id, selected if all(selected.values()) else None)
    else:
        result = verify_saved(args.project_root, args.run_id, rehash_inputs=args.mode == "verify")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
