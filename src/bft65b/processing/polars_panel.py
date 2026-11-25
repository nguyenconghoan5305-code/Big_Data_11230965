"""P3.1 observed-panel workload with Polars lazy scans and exact P2 parity.

This module does not construct a prediction population, labels, or features.
Its three output relations are a frozen contract for later pandas and Spark
implementations. Timings here are not comparative benchmark measurements.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import platform
import re
from typing import Any, Mapping, Sequence

import polars as pl
import pyarrow.parquet as pq

from bft65b.acquisition.extended import full_plan
from bft65b.config import load_config
from bft65b.manifest import RunManifest, dependency_versions
from bft65b.panel.build import _accepted_inputs, _write_verified
from bft65b.quality.audit import _read_json, _sha, _write_json


WORKLOAD_ID = "observed_panel_quality_gaps_v1"
TREATMENT_RUN = "treatment20261114a"
PANEL_RUN = "panel20261115b"
AUDIT_RUN = "audit20261114b"
RUN_NAMESPACE = "task_3_1_polars"
INPUT_COLUMNS = ("CERT", "REPDTE", "ID", "P3ASSET", "SOURCE_MISSING_FIELDS", "DQ_FLAGS")
GAP_COLUMNS = ("CERT", "PREVIOUS_REPDTE", "NEXT_REPDTE", "MISSING_INTERVENING_QUARTERS")
QUALITY_COLUMNS = ("REPDTE", "ROW_COUNT", "DISTINCT_KEY_COUNT", "P3ASSET_NULL_ROWS",
                   "P3ASSET_SOURCE_ABSENT_ROWS", "P3ASSET_EXPLICIT_NULL_ROWS",
                   "UNMATCHED_INSTITUTIONS_SNAPSHOT_ROWS")
UNMATCHED_FLAG = "financial_cert_absent_from_institutions:CERT"
EXPECTED_TYPES = {"CERT": pl.Int64, "REPDTE": pl.Date, "ID": pl.String,
                  "P3ASSET": pl.Float64, "SOURCE_MISSING_FIELDS": pl.List(pl.String),
                  "DQ_FLAGS": pl.List(pl.String)}


def lazy_queries(paths: Sequence[Path]) -> dict[str, pl.LazyFrame]:
    """Define every transformation lazily from the same explicit file list."""
    if not paths or len(set(map(str, paths))) != len(paths):
        raise ValueError("Input file list is empty or contains duplicate paths")
    base = pl.scan_parquet([str(path) for path in paths], hive_partitioning=False,
                           glob=False).select(INPUT_COLUMNS)
    observed = base.sort(["REPDTE", "CERT"])
    ordered = base.select(["CERT", "REPDTE"]).sort(["CERT", "REPDTE"])
    previous = ordered.with_columns(
        pl.col("REPDTE").shift(1).over("CERT").alias("PREVIOUS_REPDTE"))
    current_index = pl.col("REPDTE").dt.year() * 4 + pl.col("REPDTE").dt.month() // 3 - 1
    prior_index = (pl.col("PREVIOUS_REPDTE").dt.year() * 4
                   + pl.col("PREVIOUS_REPDTE").dt.month() // 3 - 1)
    gaps = (previous.with_columns((current_index - prior_index - 1).cast(pl.Int32)
             .alias("MISSING_INTERVENING_QUARTERS"))
            .filter(pl.col("MISSING_INTERVENING_QUARTERS") > 0)
            .select([pl.col("CERT"), pl.col("PREVIOUS_REPDTE"),
                     pl.col("REPDTE").alias("NEXT_REPDTE"),
                     pl.col("MISSING_INTERVENING_QUARTERS")])
            .sort(["CERT", "NEXT_REPDTE"]))
    absent = pl.col("SOURCE_MISSING_FIELDS").list.contains("P3ASSET")
    null = pl.col("P3ASSET").is_null()
    quality = (base.group_by("REPDTE").agg([
        pl.len().alias("ROW_COUNT"),
        pl.col("CERT").n_unique().alias("DISTINCT_KEY_COUNT"),
        null.sum().alias("P3ASSET_NULL_ROWS"),
        absent.sum().alias("P3ASSET_SOURCE_ABSENT_ROWS"),
        (null & ~absent).sum().alias("P3ASSET_EXPLICIT_NULL_ROWS"),
        pl.col("DQ_FLAGS").list.contains(UNMATCHED_FLAG).sum()
          .alias("UNMATCHED_INSTITUTIONS_SNAPSHOT_ROWS"),
    ]).sort("REPDTE"))
    return {"observed": observed, "gaps": gaps, "quality": quality}


def materialize(queries: Mapping[str, pl.LazyFrame],
                quarters: Sequence[str]) -> dict[str, pl.DataFrame]:
    """Collect only at the documented sink and reject duplicate bank-quarters."""
    expected = {date(int(q[:4]), int(q[4:6]), int(q[6:])) for q in quarters}
    observed = queries["observed"].collect()
    gaps = queries["gaps"].collect()
    quality = queries["quality"].collect()
    if (tuple(observed.columns) != INPUT_COLUMNS
            or any(observed.schema[name] != dtype for name, dtype in EXPECTED_TYPES.items())
            or tuple(gaps.columns) != GAP_COLUMNS
            or tuple(quality.columns) != QUALITY_COLUMNS):
        raise ValueError("Workload output schema or column order differs")
    if (set(quality["REPDTE"].to_list()) != expected or len(quality) != len(expected)
            or observed["CERT"].null_count() or observed["REPDTE"].null_count()
            or (quality["ROW_COUNT"] != quality["DISTINCT_KEY_COUNT"]).any()):
        raise ValueError("Quarter inventory, required key, or duplicate bank-quarter differs")
    if (quality["P3ASSET_NULL_ROWS"] != quality["P3ASSET_SOURCE_ABSENT_ROWS"]
            + quality["P3ASSET_EXPLICIT_NULL_ROWS"]).any():
        raise ValueError("Source absence and explicit null counts do not reconcile")
    if (quality["ROW_COUNT"].sum() != len(observed)
            or gaps["MISSING_INTERVENING_QUARTERS"].min() is not None
            and gaps["MISSING_INTERVENING_QUARTERS"].min() < 1):
        raise ValueError("Workload rows or gap values do not reconcile")
    return {"observed": observed, "gaps": gaps, "quality": quality}


def _reference(root: Path, quarters: Sequence[str], panel_run: str = PANEL_RUN) -> tuple[dict[str, Path], Path, dict[str, str], dict[str, Any]]:
    evidence = root / "runs" / "task_2_2_panel" / panel_run
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    manifest = RunManifest.read(evidence / "run_manifest.json")
    if (manifest.status != "complete" or summary.get("status") != "complete"
            or metadata.get("status") != "complete"
            or metadata.get("summary_sha256") != _sha(evidence / "summary.json")
            or len(summary.get("outputs", [])) != 173 or len(manifest.outputs) != 173
            or any(not row.get("passed") for row in manifest.validations)):
        raise ValueError("Accepted P2.2 panel evidence is incomplete")
    base = (root / "data_processed" / "task_2_2_panel" / panel_run).resolve()
    outputs = {row["path"]: row for row in manifest.outputs}
    paths: dict[str, Path] = {}
    protected = {str(evidence / name): _sha(evidence / name) for name in
                 ("summary.json", "metadata.json", "run_manifest.json", "report.md")}
    gaps_path: Path | None = None
    for item in summary["outputs"]:
        if item["dataset"] not in {"financials", "reporting_gaps"}:
            continue
        relative = Path(item["relative_path"])
        path = (base / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(base)
                or outputs.get(str(path), {}).get("sha256") != item["sha256"]
                or _sha(path) != item["sha256"]):
            raise ValueError(f"Accepted panel reference differs: {relative}")
        protected[str(path)] = item["sha256"]
        if item["dataset"] == "financials":
            if item["quarter"] in paths:
                raise ValueError("Duplicate reference quarter")
            paths[item["quarter"]] = path
        else:
            gaps_path = path
    if set(paths) != set(quarters) or gaps_path is None:
        raise ValueError("Accepted panel reference quarter/gap inventory differs")
    return paths, gaps_path, protected, summary


def _check_parity(outputs: Mapping[str, pl.DataFrame], reference_paths: Mapping[str, Path],
                  reference_gaps: Path, audit: Mapping[str, Any]) -> dict[str, Any]:
    """Compare every key and projected value, and independently audit counts."""
    observed, gaps, quality = (outputs[name] for name in ("observed", "gaps", "quality"))
    audit_periods = {row["period"]: row for row in audit["periods"]
                     if row["dataset"] == "financials"}
    by_q = []
    offset = 0
    for stat in quality.iter_rows(named=True):
        q = stat["REPDTE"].strftime("%Y%m%d")
        n = stat["ROW_COUNT"]
        actual = observed.slice(offset, n)
        with pq.ParquetFile(reference_paths[q]) as file:
            expected = pl.from_arrow(file.read(columns=list(INPUT_COLUMNS))).sort(["REPDTE", "CERT"])
        if not actual.equals(expected, null_equal=True):
            raise ValueError(f"Projected bank-quarter keys or values differ from P2.2: {q}")
        period = audit_periods.get(q)
        if period is None or (n != period["rows"]
                or stat["DISTINCT_KEY_COUNT"] != period["distinct_cert_count"]
                or stat["P3ASSET_SOURCE_ABSENT_ROWS"] != period["field_counts"]["P3ASSET"]["absent"]
                or stat["P3ASSET_EXPLICIT_NULL_ROWS"] != period["field_counts"]["P3ASSET"]["explicit_null"]
                or stat["UNMATCHED_INSTITUTIONS_SNAPSHOT_ROWS"] !=
                period["institution_snapshot_join"]["unmatched_records"]):
            raise ValueError(f"Quarter quality counts differ from P2.1: {q}")
        by_q.append({"quarter": q, **{key: value for key, value in stat.items()
                                      if key != "REPDTE"}})
        offset += n
    if offset != len(observed):
        raise ValueError("Quarter slices do not cover observed relation")
    with pq.ParquetFile(reference_gaps) as file:
        expected_gaps = pl.from_arrow(file.read()).sort(["CERT", "NEXT_REPDTE"])
    if not gaps.equals(expected_gaps, null_equal=True):
        raise ValueError("Resumed gap records differ from P2.2")
    return {"quarters": len(by_q), "projected_rows_exact": len(observed),
            "projected_value_mismatches": 0, "gap_events_exact": len(gaps),
            "gap_distinct_banks": gaps["CERT"].n_unique(),
            "missing_intervening_quarters": gaps["MISSING_INTERVENING_QUARTERS"].sum(),
            "by_quarter": by_q}


def _paths(root: Path, run_id: str) -> tuple[Path, Path]:
    return (root / "data_processed" / RUN_NAMESPACE / run_id,
            root / "runs" / RUN_NAMESPACE / run_id)


def run(root: Path, run_id: str, *, treatment_run: str = TREATMENT_RUN,
        panel_run: str = PANEL_RUN, audit_run: str = AUDIT_RUN) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}", value)
           for value in (treatment_run, panel_run, audit_run)):
        raise ValueError("Unsafe upstream run ID")
    if (root / "runs/source_vintage_bundle.json").is_file() and (
            treatment_run == TREATMENT_RUN or panel_run == PANEL_RUN or audit_run == AUDIT_RUN):
        raise ValueError("Portable processing requires explicit portable upstream runs")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}", run_id):
        raise ValueError("Unsafe run ID")
    data_root, evidence = _paths(root, run_id)
    if data_root.exists() or evidence.exists():
        raise FileExistsError("P3.1 run already exists")
    config = load_config(root / "config" / "financials_full_v2.json")
    quarters = tuple(full_plan(config).quarters)
    treatment_summary, inventory, treatment_manifest = _accepted_inputs(root, config, treatment_run)
    indexed = {row["quarter"]: row for row in inventory if row["dataset"] == "financials"}
    if len(quarters) != 170 or set(indexed) != set(quarters):
        raise ValueError("Accepted treatment Financials inventory differs")
    reference_paths, reference_gaps, reference_hashes, reference_summary = _reference(root, quarters, panel_run)
    if (reference_summary.get("source_treatment_run") != treatment_run
            or treatment_manifest.config.get("source_audit_run") != audit_run):
        raise ValueError("Processing panel, treatment and audit lineage differs")
    audit_path = root / "runs" / "task_2_1_quality" / audit_run / "findings.json"
    audit = _read_json(audit_path)
    if (_sha(audit_path) != treatment_summary["audit_findings_sha256"]
            or audit.get("status") != "complete"):
        raise ValueError("P2.1 audit findings changed")
    input_files = [indexed[q] for q in quarters]
    input_hashes = {item["path"]: item["derived_sha256"] for item in input_files}
    queries = lazy_queries([Path(item["path"]) for item in input_files])
    evidence.mkdir(parents=True)
    data_root.mkdir(parents=True)
    manifest = RunManifest.new(run_id, config)
    manifest.config.update({"workload_id": WORKLOAD_ID, "treatment_run": treatment_run,
                            "panel_reference_run": panel_run, "audit_run": audit_run})
    manifest.set_status("running")
    manifest.write(evidence / "run_manifest.json")
    try:
        plans = {}
        for name, query in queries.items():
            plan_path = evidence / f"{name}_optimized_plan.txt"
            plan_path.write_text(query.explain(optimized=True), encoding="utf-8")
            plans[name] = {"path": str(plan_path), "sha256": _sha(plan_path)}
        frames = materialize(queries, quarters)
        parity = _check_parity(frames, reference_paths, reference_gaps, audit)
        if (parity["projected_rows_exact"] != reference_summary["panel_overall"]["output_rows"]
                or parity["gap_events_exact"] != reference_summary["gaps"]["gap_events"]
                or parity["gap_distinct_banks"] != reference_summary["gaps"]["distinct_banks"]
                or parity["missing_intervening_quarters"] != reference_summary["gaps"]["missing_quarters"]):
            raise ValueError("Overall panel/gap evidence differs")
        artifacts = {}
        for name, frame in frames.items():
            path = data_root / f"{name}.parquet"
            digest = _write_verified(path, frame.to_arrow())
            artifacts[name] = {"path": str(path), "rows": len(frame), "sha256": digest,
                               "schema": {k: str(v) for k, v in frame.schema.items()}}
            manifest.record_output(path=str(path), dataset=name, rows=len(frame), sha256=digest,
                compression="ZSTD", schema_version=WORKLOAD_ID, partition=None,
                source_total=len(frame), source_query=None)
        for path, digest in {**input_hashes, **reference_hashes,
                             str(audit_path): treatment_summary["audit_findings_sha256"]}.items():
            if _sha(Path(path)) != digest:
                raise ValueError(f"Protected input/reference changed: {path}")
        overall = {"observed_bank_quarters": len(frames["observed"]),
                   "unique_bank_quarter_keys": int(frames["quality"]["DISTINCT_KEY_COUNT"].sum()),
                   "resumed_gap_events": len(frames["gaps"]),
                   "resumed_gap_banks": parity["gap_distinct_banks"],
                   "missing_intervening_quarters": parity["missing_intervening_quarters"],
                   "p3asset_null_rows": int(frames["quality"]["P3ASSET_NULL_ROWS"].sum()),
                   "p3asset_source_absent_rows": int(frames["quality"]["P3ASSET_SOURCE_ABSENT_ROWS"].sum()),
                   "p3asset_explicit_null_rows": int(frames["quality"]["P3ASSET_EXPLICIT_NULL_ROWS"].sum()),
                   "unmatched_institutions_snapshot_rows": int(frames["quality"]["UNMATCHED_INSTITUTIONS_SNAPSHOT_ROWS"].sum())}
        contract = {"workload_id": WORKLOAD_ID, "status": "frozen_after_parity",
            "source_run": treatment_run, "reference_run": panel_run, "audit_run": audit_run,
            "input_files": [{"quarter": q, "path": indexed[q]["path"],
                "sha256": indexed[q]["derived_sha256"], "rows": indexed[q]["rows"]} for q in quarters],
            "input_columns": list(INPUT_COLUMNS),
            "input_types": {k: str(v) for k, v in EXPECTED_TYPES.items()},
            "key": ["CERT", "REPDTE"], "observed_order": ["REPDTE", "CERT"],
            "gap_order": ["CERT", "NEXT_REPDTE"], "quality_order": ["REPDTE"],
            "gap_rule": "quarter_index(next)-quarter_index(previous)-1 > 0 for consecutive observed CERT reports",
            "null_rule": "P3ASSET null is split by SOURCE_MISSING_FIELDS contains P3ASSET; no imputation",
            "unmatched_flag": UNMATCHED_FLAG,
            "materialization": "Polars LazyFrame.collect after optimized plans are captured",
            "output_schemas": {name: row["schema"] for name, row in artifacts.items()},
            "expected_outputs": {name: {"rows": row["rows"], "sha256": row["sha256"]}
                                 for name, row in artifacts.items()}}
        _write_json(evidence / "contract.json", contract)
        summary = {"status": "complete", "run_id": run_id, "workload_id": WORKLOAD_ID,
                   "overall": overall, "parity": {key: value for key, value in parity.items()
                                                  if key != "by_quarter"},
                   "by_quarter": parity["by_quarter"], "artifacts": artifacts,
                   "plans": plans, "contract_path": str(evidence / "contract.json")}
        _write_json(evidence / "summary.json", summary)
        metadata = {"status": "complete", "at_utc": datetime.now(timezone.utc).isoformat(),
            "os": platform.platform(), "python": platform.python_version(),
            "dependencies": dependency_versions(), "config": config.to_dict(),
            "input_hashes": input_hashes, "reference_hashes": reference_hashes,
            "audit_findings_sha256": treatment_summary["audit_findings_sha256"],
            "contract_sha256": _sha(evidence / "contract.json"),
            "summary_sha256": _sha(evidence / "summary.json")}
        _write_json(evidence / "metadata.json", metadata)
        for item in input_files:
            manifest.record_input(item["path"], item["derived_sha256"], item["rows"])
        manifest.record_validation("accepted_input_and_reference_hashes", True,
            "170 treatment Financials and 171 P2.2 reference Parquet files verified before and after")
        manifest.record_validation("all_projected_rows_and_values", True,
            "170 quarter key sets and all six projected columns match P2.2 exactly")
        manifest.record_validation("resumed_gaps", True,
            "Every gap record matches P2.2 exactly")
        manifest.record_validation("per_quarter_quality", True,
            "170 P3ASSET and snapshot-unmatched counts match P2.1 audit")
        manifest.record_validation("output_readback_and_contract", True,
            "Three ZSTD outputs passed atomic write/read-back; optimized plans and contract saved")
        manifest.set_status("complete")
        manifest.write(evidence / "run_manifest.json")
        return summary
    except Exception as exc:
        manifest.record_validation("p3_1_run", False, f"{type(exc).__name__}: {exc}")
        manifest.set_status("failed")
        manifest.write(evidence / "run_manifest.json")
        raise


def _accepted_result(root: Path, run_id: str) -> dict[str, Any]:
    _, evidence = _paths(root, run_id)
    manifest = RunManifest.read(evidence / "run_manifest.json")
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    if (manifest.run_id != run_id or manifest.status != "complete"
            or summary.get("status") != "complete" or summary.get("run_id") != run_id
            or metadata.get("status") != "complete"
            or _sha(evidence / "summary.json") != metadata["summary_sha256"]
            or _sha(evidence / "contract.json") != metadata["contract_sha256"]
            or len(manifest.outputs) != 3 or any(not x["passed"] for x in manifest.validations)):
        raise ValueError("P3.1 saved evidence is incomplete")
    by_name = {row["dataset"]: row for row in manifest.outputs}
    if (set(by_name) != {"observed", "gaps", "quality"}
            or any(by_name[name]["path"] != summary["artifacts"][name]["path"]
                   or by_name[name]["sha256"] != summary["artifacts"][name]["sha256"]
                   or by_name[name]["rows"] != summary["artifacts"][name]["rows"]
                   for name in by_name)):
        raise ValueError("P3.1 manifest and output inventory differ")
    return summary


def compare(root: Path, run_id: str, candidate_dir: Path) -> dict[str, Any]:
    """Executable full-output parity gate for later pandas and Spark work."""
    summary = _accepted_result(root, run_id)
    reference_dir, _ = _paths(root, run_id)
    compared = {}
    for name in ("observed", "gaps", "quality"):
        path = (reference_dir / f"{name}.parquet").resolve()
        expected = summary["artifacts"][name]
        if (path != Path(expected["path"]).resolve()
                or not path.is_relative_to(reference_dir.resolve())
                or _sha(path) != expected["sha256"]):
            raise ValueError(f"Saved P3.1 {name} reference checksum/path differs")
        reference = pl.read_parquet(path, hive_partitioning=False)
        candidate = pl.read_parquet(candidate_dir / f"{name}.parquet", hive_partitioning=False)
        if (candidate.columns != reference.columns or candidate.schema != reference.schema
                or not candidate.equals(reference, null_equal=True)):
            raise ValueError(f"Candidate output differs from frozen P3.1 {name} relation")
        compared[name] = len(reference)
    return {"status": "exact_parity", "run_id": run_id, "rows_by_relation": compared,
            "reference_overall": summary["overall"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="P3.1 lazy Polars observed-panel workload")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "show", "plans", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--project-root", type=Path, default=Path.cwd())
        p.add_argument("--run-id", required=True)
        if name == "run":
            p.add_argument("--treatment-run", default=TREATMENT_RUN)
            p.add_argument("--panel-run", default=PANEL_RUN)
            p.add_argument("--audit-run", default=AUDIT_RUN)
        if name == "compare":
            p.add_argument("--candidate-dir", required=True, type=Path)
        if name == "plans":
            p.add_argument("--relation", choices=("observed", "gaps", "quality"),
                           default="gaps")
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run(args.project_root, args.run_id, treatment_run=args.treatment_run,
                     panel_run=args.panel_run, audit_run=args.audit_run)
        print(json.dumps({"status": result["status"], "run_id": args.run_id,
                          "overall": result["overall"], "plans": result["plans"]}, indent=2))
    elif args.command == "compare":
        print(json.dumps(compare(args.project_root, args.run_id, args.candidate_dir), indent=2))
    else:
        try:
            result = _accepted_result(args.project_root, args.run_id)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.exit(2, f"P3.1 run {args.run_id}: {exc}\n")
        if args.command == "plans":
            info = result["plans"][args.relation]
            path = Path(info["path"])
            if _sha(path) != info["sha256"]:
                parser.exit(2, f"P3.1 run {args.run_id}: optimized plan checksum differs\n")
            print(path.read_text(encoding="utf-8"))
        else:
            print(json.dumps({"status": result["status"], "run_id": args.run_id,
                              "overall": result["overall"], "plans": result["plans"]}, indent=2))


if __name__ == "__main__":
    main()
