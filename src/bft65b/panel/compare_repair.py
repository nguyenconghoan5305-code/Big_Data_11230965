"""Read-only logical comparison of accepted and relocated P2.3 repairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from bft65b.manifest import RunManifest
from bft65b.quality.audit import _read_json, _sha


KEYS = {"failure_events": "ID", "exit_assessments": "CERT"}


def _load(root: Path, run_id: str) -> tuple[dict, dict]:
    evidence = root / "runs/task_2_3_transaction_repair" / run_id
    manifest = RunManifest.read(evidence / "run_manifest.json")
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    if (manifest.run_id != f"p2_3_repair_{run_id}" or manifest.status != "complete"
            or not manifest.validations or any(item.get("passed") is not True for item in manifest.validations)
            or summary.get("status") != "complete" or metadata.get("status") != "complete"
            or metadata.get("summary_sha256") != _sha(evidence / "summary.json")
            or len(manifest.outputs) != 2 or len(summary.get("outputs", [])) != 2):
        raise ValueError(f"Incomplete repair evidence: {run_id}")
    tables = {}
    output_by_dataset = {item["dataset"]: item for item in manifest.outputs}
    if set(output_by_dataset) != set(KEYS):
        raise ValueError(f"Repair output coverage differs: {run_id}")
    for dataset, key in KEYS.items():
        item = output_by_dataset[dataset]
        expected = (root / "data_processed/task_2_3_transaction_repair" / run_id /
                    dataset / "part-000.parquet").resolve()
        recorded = next((entry for entry in summary["outputs"] if entry.get("dataset") == dataset), None)
        if (Path(item["path"]).resolve() != expected or recorded is None
                or Path(recorded["path"]).resolve() != expected
                or item["sha256"] != recorded.get("sha256") or item["sha256"] != _sha(expected)):
            raise ValueError(f"Repair output path/checksum differs: {run_id}:{dataset}")
        table = pq.read_table(expected)
        values = table[key].to_pylist()
        if (table.num_rows != item["rows"] or table.num_rows != recorded.get("rows")
                or any(value is None for value in values) or len(set(values)) != len(values)):
            raise ValueError(f"Repair output row/key validation differs: {run_id}:{dataset}")
        tables[dataset] = table.sort_by([(key, "ascending")])
    return summary, tables


def compare(accepted_root: Path, accepted_run: str, replay_root: Path, replay_run: str) -> dict:
    accepted, left = _load(accepted_root, accepted_run)
    replay, right = _load(replay_root, replay_run)
    outputs = {}
    for dataset in KEYS:
        schema_equal = left[dataset].schema.equals(right[dataset].schema, check_metadata=True)
        values_equal = left[dataset].equals(right[dataset], check_metadata=True)
        outputs[dataset] = {"accepted_rows": left[dataset].num_rows,
                            "replay_rows": right[dataset].num_rows,
                            "schema_equal": schema_equal, "keyed_values_equal": values_equal}
    categories = {"accepted": accepted["new_categories"], "replay": replay["new_categories"]}
    dispositions = {"accepted": accepted["transaction_dispositions"],
                    "replay": replay["transaction_dispositions"]}
    result = {"status": "PASS" if all(x["schema_equal"] and x["keyed_values_equal"]
        for x in outputs.values()) and categories["accepted"] == categories["replay"]
        and dispositions["accepted"] == dispositions["replay"] else "FAIL",
        "accepted_run": accepted_run, "replay_run": replay_run,
        "outputs": outputs, "categories": categories, "transactions": dispositions}
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare two completed P2.3 repairs by keyed values")
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--accepted-run", default="repair20261115f")
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--replay-run", required=True)
    args = parser.parse_args(argv)
    result = compare(args.accepted_root.resolve(), args.accepted_run,
                     args.replay_root.resolve(), args.replay_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        parser.exit(2, "P2.3 repaired-result parity failed\n")


if __name__ == "__main__":
    main()
