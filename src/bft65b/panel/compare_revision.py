"""Exact keyed comparison of two saved P2.4 prediction runs.

This is read-only for both runs. It compares the accepted prediction columns
without converting decimal or floating values to text or applying tolerances.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from bft65b.manifest import RunManifest
from bft65b.panel.asof import PREDICTOR_FIELDS
from bft65b.quality.audit import _read_json, _sha, _write_json


COMPARISON_FIELDS = (
    "ELIGIBILITY_DISPOSITION", "SELECTED_REPDTE", "AVAILABILITY_DATE",
    "REPORT_AGE_DAYS", *PREDICTOR_FIELDS,
)


def compare_tables(previous: pa.Table, current: pa.Table) -> dict[str, Any]:
    """Count changes by (CERT, Q), including null and value differences."""
    needed = ("CERT", "Q", *COMPARISON_FIELDS)
    if any(name not in table.column_names for table in (previous, current)
           for name in needed):
        raise ValueError("Prediction comparison lacks required columns")
    old = previous.select(needed).sort_by([("Q", "ascending"), ("CERT", "ascending")])
    new = current.select(needed).sort_by([("Q", "ascending"), ("CERT", "ascending")])
    old_keys = list(zip(old["CERT"].to_pylist(), old["Q"].to_pylist()))
    new_keys = list(zip(new["CERT"].to_pylist(), new["Q"].to_pylist()))
    if len(old_keys) != len(set(old_keys)) or len(new_keys) != len(set(new_keys)):
        raise ValueError("Prediction comparison contains duplicate CERT/q keys")
    old_set, new_set = set(old_keys), set(new_keys)
    differences = {name: 0 for name in COMPARISON_FIELDS}
    if old_keys == new_keys:
        for name in COMPARISON_FIELDS:
            if not old[name].equals(new[name]):
                differences[name] = sum(
                    a != b for a, b in zip(old[name].to_pylist(), new[name].to_pylist()))
    else:
        old_index = {key: i for i, key in enumerate(old_keys)}
        new_index = {key: i for i, key in enumerate(new_keys)}
        shared = sorted(old_set & new_set)
        for name in COMPARISON_FIELDS:
            old_values, new_values = old[name].to_pylist(), new[name].to_pylist()
            differences[name] = sum(old_values[old_index[key]] != new_values[new_index[key]]
                                    for key in shared)
    return {"previous_rows": len(old_keys), "current_rows": len(new_keys),
            "shared_keys": len(old_set & new_set),
            "previous_only_keys": len(old_set - new_set),
            "current_only_keys": len(new_set - old_set),
            "field_differences": differences}


def _saved(root: Path, run_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    evidence = root / "runs" / "task_2_5_prediction" / run_id
    manifest = RunManifest.read(evidence / "run_manifest.json")
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    if (manifest.run_id != run_id or manifest.status != "complete"
            or summary.get("run_id") != run_id or summary.get("status") != "complete"
            or metadata.get("summary_sha256") != _sha(evidence / "summary.json")
            or len(manifest.outputs) != 166 or len(summary.get("outputs", [])) != 166
            or any(not item["passed"] for item in manifest.validations)):
        raise ValueError(f"Saved prediction run is incomplete: {run_id}")
    outputs = {item["quarter"]: item for item in summary["outputs"]}
    if len(outputs) != 166:
        raise ValueError(f"Duplicate prediction quarter in {run_id}")
    return outputs, {"manifest_sha256": _sha(evidence / "run_manifest.json"),
                     "summary_sha256": _sha(evidence / "summary.json")}


def compare_runs(root: Path, previous_id: str, current_id: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    previous, old_evidence = _saved(root, previous_id)
    current, new_evidence = _saved(root, current_id)
    if set(previous) != set(current):
        raise ValueError("Prediction quarter inventories differ")
    counts = {name: 0 for name in COMPARISON_FIELDS}
    totals = {"previous_rows": 0, "current_rows": 0, "shared_keys": 0,
              "previous_only_keys": 0, "current_only_keys": 0}
    per_quarter = []
    columns = ["CERT", "Q", *COMPARISON_FIELDS]
    for quarter in sorted(previous):
        before, after = previous[quarter], current[quarter]
        if _sha(Path(before["path"])) != before["sha256"] or _sha(Path(after["path"])) != after["sha256"]:
            raise ValueError(f"Prediction file checksum differs: {quarter}")
        with pq.ParquetFile(before["path"]) as parquet:
            old = parquet.read(columns=columns)
        with pq.ParquetFile(after["path"]) as parquet:
            new = parquet.read(columns=columns)
        row = compare_tables(old, new)
        if row["previous_rows"] != before["rows"] or row["current_rows"] != after["rows"]:
            raise ValueError(f"Prediction file row count differs: {quarter}")
        per_quarter.append({"quarter": quarter, **row})
        for name in totals:
            totals[name] += row[name]
        for name in COMPARISON_FIELDS:
            counts[name] += row["field_differences"][name]
    status = ("identical" if not totals["previous_only_keys"]
              and not totals["current_only_keys"] and not any(counts.values())
              else "different")
    return {"status": status, "previous_run": previous_id, "current_run": current_id,
            "compared_utc": datetime.now(timezone.utc).isoformat(),
            "quarters": len(per_quarter), "keys": totals,
            "field_differences": counts,
            "predictor_difference_total": sum(counts[name] for name in PREDICTOR_FIELDS),
            "previous_evidence": old_evidence, "current_evidence": new_evidence,
            "by_quarter": per_quarter}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare two accepted P2.4 runs by CERT/q")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--previous-run", required=True)
    parser.add_argument("--current-run", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    ignored_root = (args.project_root.resolve() / "runs" / "task_2_4_revision_repair").resolve()
    if not output.is_relative_to(ignored_root) or output.exists():
        parser.error("Comparison output must be a fresh path in ignored revision-repair runs")
    result = compare_runs(args.project_root, args.previous_run, args.current_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "by_quarter"}, indent=2))
    if result["status"] != "identical":
        parser.exit(1, "Prediction runs differ; see comparison evidence\n")


if __name__ == "__main__":
    main()
