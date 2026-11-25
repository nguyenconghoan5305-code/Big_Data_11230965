"""Read-only, keyed comparison of two complete PIT candidate runs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pyarrow.parquet as pq

from bft65b.panel.prediction import RUN_NAMESPACE, verify_saved
from bft65b.quality.audit import _read_json, _sha, _write_json


def compare(left_root: Path, left_id: str, right_root: Path, right_id: str) -> dict:
    left_root, right_root = left_root.resolve(strict=True), right_root.resolve(strict=True)
    left_check = verify_saved(left_root, left_id)
    right_check = verify_saved(right_root, right_id)
    left = _read_json(left_root / "runs" / RUN_NAMESPACE / left_id / "summary.json")
    right = _read_json(right_root / "runs" / RUN_NAMESPACE / right_id / "summary.json")
    left_outputs = {row["quarter"]: row for row in left["outputs"]}
    right_outputs = {row["quarter"]: row for row in right["outputs"]}
    if left_outputs.keys() != right_outputs.keys():
        raise ValueError("PIT quarter coverage differs")
    mismatches = Counter()
    rows = 0
    physical_hash_matches = 0
    for quarter in sorted(left_outputs):
        a, b = left_outputs[quarter], right_outputs[quarter]
        with pq.ParquetFile(a["path"]) as source, pq.ParquetFile(b["path"]) as replay:
            if not source.schema_arrow.equals(replay.schema_arrow, check_metadata=True):
                mismatches["schema"] += 1
                continue
            left_table = source.read().sort_by([("CERT", "ascending"), ("Q", "ascending")])
            right_table = replay.read().sort_by([("CERT", "ascending"), ("Q", "ascending")])
        rows += left_table.num_rows
        if left_table.num_rows != right_table.num_rows:
            mismatches["rows"] += 1
            continue
        if not left_table.select(["CERT", "Q"]).equals(right_table.select(["CERT", "Q"])):
            mismatches["keys"] += 1
            continue
        for field in left_table.column_names:
            if not left_table[field].equals(right_table[field]):
                mismatches[field] += 1
        physical_hash_matches += a["sha256"] == b["sha256"]
    if left["overall"] != right["overall"]:
        mismatches["overall"] += 1
    if left["by_quarter"] != right["by_quarter"]:
        mismatches["by_quarter"] += 1
    if left["future_perturbation"] != right["future_perturbation"]:
        mismatches["future_perturbation"] += 1
    return {"status": "PASS" if not mismatches else "FAIL",
        "left_run": left_id, "right_run": right_id,
        "left_summary_sha256": _sha(left_root / "runs" / RUN_NAMESPACE / left_id / "summary.json"),
        "right_summary_sha256": _sha(right_root / "runs" / RUN_NAMESPACE / right_id / "summary.json"),
        "quarters": len(left_outputs), "keyed_rows_compared": rows,
        "physical_hash_matches": physical_hash_matches,
        "column_or_evidence_mismatches": dict(sorted(mismatches.items())),
        "left_validation": left_check, "right_validation": right_check}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-root", required=True, type=Path)
    parser.add_argument("--left-run", required=True)
    parser.add_argument("--right-root", required=True, type=Path)
    parser.add_argument("--right-run", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare(args.left_root, args.left_run, args.right_root, args.right_run)
    if args.output:
        _write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
