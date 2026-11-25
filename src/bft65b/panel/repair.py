"""Retrospective P2.3 transaction repair without rewriting the accepted panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from bft65b.acquisition.client import FDICClient
from bft65b.config import load_config
from bft65b.manifest import RunManifest, dependency_versions
from bft65b.panel.build import EXIT_SCHEMA, _write_verified, classify_exits
from bft65b.quality.audit import _read_json, _sha, _write_json


SOURCE_FIELDS = "CERT,FAILDATE,COST,RESTYPE,RESTYPE1"
SOURCE_INDEX = "failures_1787667198788"
PREVIOUS_RUN = "panel20261115b"  # CLI default for historical compatibility only.
CATEGORIES = ("confirmed_failure", "documented_merger", "documented_other_exit",
              "ambiguous_exit", "boundary_observed")
FAILURE_METHODS = frozenset({"PA", "P&A", "PI", "IDT", "ABT", "PO", "DINB", "MGR"})
ASSISTANCE_METHODS = frozenset({"A/A", "REP"})
UNCERTAIN_FAILURE_METHODS = frozenset({"REP"})
TRANSACTION_FIELDS = [
    pa.field("RESTYPE", pa.string()),
    pa.field("RESTYPE1", pa.string()),
    pa.field("TRANSACTION_DISPOSITION", pa.string(), nullable=False),
    pa.field("TRANSACTION_RULE", pa.string(), nullable=False),
    pa.field("SUPPLEMENT_MATCH", pa.string(), nullable=False),
]


def transaction_disposition(restype: Any, method: Any) -> tuple[str, str]:
    """Use FDIC resolution status and charter-method evidence separately."""
    if not isinstance(restype, str) or not isinstance(method, str):
        return "unknown_transaction", "RESTYPE or RESTYPE1 is absent or null"
    status, code = restype.upper(), method.upper()
    if status == "FAILURE" and code in FAILURE_METHODS:
        return "verified_failure", "FDIC Failure with documented failure-resolution method"
    if status == "FAILURE" and code in UNCERTAIN_FAILURE_METHODS:
        return "failure_charter_uncertain", "FDIC Failure with reprivatization method; charter continuity needs review"
    if status == "ASSISTANCE" and code == "OBAM":
        return "open_bank_assistance_code_unverified", "FDIC Assistance; OBAM literal is outside current dictionary enum"
    if status == "ASSISTANCE" and code in ASSISTANCE_METHODS:
        return "assistance_transaction", "FDIC Assistance; not a verified failed-bank exit"
    if status in {"FAILURE", "ASSISTANCE"} and code in (FAILURE_METHODS | ASSISTANCE_METHODS | UNCERTAIN_FAILURE_METHODS | {"OBAM"}):
        return "transaction_conflict", "FDIC status conflicts with resolution-method category"
    return "unmapped_transaction", "FDIC status or resolution code is not in the verified policy"


def reconcile_events(accepted: Sequence[Mapping[str, Any]],
                     supplemental: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Match by source ID only; retain every accepted event on mismatch."""
    candidates: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in supplemental:
        if isinstance(row.get("ID"), str) and row["ID"].strip():
            candidates[row["ID"]].append(row)
    seen: set[str] = set()
    result = []
    for event in accepted:
        source_id = event["ID"]
        if source_id in seen:
            raise ValueError(f"Accepted event source ID duplicated: {source_id}")
        seen.add(source_id)
        matches = candidates.get(source_id, [])
        match = "matched"
        if not matches:
            match = "missing_source_id"
        elif len(matches) != 1:
            match = "duplicate_source_id"
        else:
            source = matches[0]
            try:
                cert = source.get("CERT")
                cert = FDICClient._certificate_number(cert) if cert is not None else None
                cost = source.get("COST")
                same = (cert == event["CERT"]
                        and FDICClient._failure_date(source["FAILDATE"]) == event["FAILDATE"].isoformat()
                        and not isinstance(cost, bool)
                        and (cost is None or isinstance(cost, (int, float)))
                        and cost == event.get("COST"))
            except (KeyError, TypeError, ValueError):
                same = False
            if not same:
                match = "corroboration_conflict"
        if match == "matched":
            source = matches[0]
            status, rule = transaction_disposition(source.get("RESTYPE"), source.get("RESTYPE1"))
            restype, method = source.get("RESTYPE"), source.get("RESTYPE1")
        else:
            status, rule = "unresolved_source_match", f"Supplemental ID reconciliation: {match}"
            restype = method = None
        result.append({**event, "RESTYPE": restype, "RESTYPE1": method,
                       "TRANSACTION_DISPOSITION": status, "TRANSACTION_RULE": rule,
                       "SUPPLEMENT_MATCH": match})
    return result, {"accepted_ids": len(seen), "supplement_rows": len(supplemental),
                    "supplement_unique_ids": len(candidates),
                    "supplement_extra_ids": len(set(candidates) - seen),
                    **dict(sorted(Counter(row["SUPPLEMENT_MATCH"] for row in result).items()))}


def _source_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _read_json(path)
    meta, items = payload.get("meta"), payload.get("data")
    if not isinstance(meta, dict) or not isinstance(items, list):
        raise ValueError("Supplemental FDIC response lacks meta/data")
    params = meta.get("parameters", {})
    expected = {"fields": SOURCE_FIELDS, "limit": "10000", "offset": "0", "filters": ""}
    if any(str(params.get(k, "")) != v for k, v in expected.items()):
        raise ValueError("Supplemental FDIC query differs from frozen contract")
    if meta.get("total") != len(items) or not 0 < len(items) <= 10000:
        raise ValueError("Supplemental FDIC source total differs")
    index = meta.get("index", {})
    if index.get("name") != SOURCE_INDEX or not index.get("createTimestamp"):
        raise ValueError("Supplemental FDIC source index changed")
    rows = []
    for item in items:
        row = item.get("data") if isinstance(item, dict) else None
        if not isinstance(row, dict):
            raise ValueError("Supplemental FDIC row lacks data")
        rows.append(row)
    return rows, meta


def _accepted_panel(root: Path, panel_run: str = PREVIOUS_RUN) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[int, list[date]], dict[int, dict[str, Any]], dict[str, str]]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}", panel_run):
        raise ValueError("Unsafe panel run ID")
    evidence = root / "runs" / "task_2_2_panel" / panel_run
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    manifest = RunManifest.read(evidence / "run_manifest.json")
    if (manifest.status != "complete" or summary.get("status") != "complete"
            or metadata.get("summary_sha256") != _sha(evidence / "summary.json")
            or len(summary.get("outputs", [])) != 173
            or len(manifest.outputs) != 173
            or summary.get("panel_overall", {}).get("output_rows") != 1682616):
        raise ValueError("Accepted panel evidence is incomplete")
    if (manifest.run_id != f"p2_2_panel_{panel_run}"
            or summary.get("source_treatment_run") != manifest.config.get("source_treatment_run")):
        raise ValueError("Selected panel lineage differs")
    data_root = root / "data_processed" / "task_2_2_panel" / panel_run
    if metadata.get("data_root") != str(data_root.resolve()):
        raise ValueError("Selected panel data root differs")
    hashes: dict[str, str] = {}
    reports: dict[int, list[date]] = defaultdict(list)
    output_by_path = {item["path"]: item for item in manifest.outputs}
    if len(output_by_path) != 173:
        raise ValueError("Accepted panel manifest contains duplicate output paths")
    for name in ("summary.json", "metadata.json", "run_manifest.json", "report.md"):
        path = evidence / name
        hashes[str(path)] = _sha(path)
    for item in summary["outputs"]:
        relative = Path(item["relative_path"])
        path = (data_root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(data_root.resolve()):
            raise ValueError("Unsafe accepted panel path")
        if (_sha(path) != item["sha256"] or output_by_path.get(str(path), {}).get("sha256") != item["sha256"]):
            raise ValueError(f"Accepted panel checksum/manifest differs: {relative}")
        hashes[str(path)] = item["sha256"]
        with pq.ParquetFile(path) as parquet:
            if parquet.metadata.num_rows != item["rows"]:
                raise ValueError(f"Accepted panel row count differs: {relative}")
            if item["dataset"] == "financials":
                keys = parquet.read(columns=["CERT", "REPDTE"])
                certs, dates = keys["CERT"].to_pylist(), keys["REPDTE"].to_pylist()
                expected = date(int(item["quarter"][:4]), int(item["quarter"][4:6]), int(item["quarter"][6:]))
                if len(set(certs)) != len(certs) or any(value is None or day != expected for value, day in zip(certs, dates)):
                    raise ValueError(f"Accepted panel bank-quarter keys differ: {relative}")
                for cert, report_date in zip(certs, dates):
                    reports[cert].append(report_date)
    if len(reports) != summary["exits"]["bank_denominator"] or sum(map(len, reports.values())) != 1682616:
        raise ValueError("Accepted panel bank/key count differs")
    events = pq.read_table(data_root / "failure_events" / "part-000.parquet").to_pylist()
    exits = pq.read_table(data_root / "exit_assessments" / "part-000.parquet").to_pylist()
    institutions = {row["CERT"]: {"ACTIVE": row["INSTITUTION_ACTIVE_SNAPSHOT"],
        "CHANGEC1": row["INSTITUTION_CHANGEC1_SNAPSHOT"],
        "SOURCE_MISSING_FIELDS": row["INSTITUTION_SOURCE_MISSING_FIELDS"]} for row in exits}
    return summary, events, exits, reports, institutions, hashes


def _select_panel_run(root: Path, panel_run: str | None) -> str:
    if panel_run is None:
        if (root / "runs/source_vintage_bundle.json").exists():
            raise ValueError("Portable replay requires explicit --panel-run; no accepted-output fallback")
        return PREVIOUS_RUN
    return panel_run


def run(root: Path, source_path: Path, run_id: str, panel_run: str | None = None) -> dict[str, Any]:
    root = root.resolve(strict=True)
    panel_run = _select_panel_run(root, panel_run)
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}", value)
           for value in (run_id, panel_run)):
        raise ValueError("Unsafe run ID")
    source_path = source_path.resolve(strict=True)
    if not source_path.is_relative_to((root / "runs" / "task_2_3_transaction_repair").resolve()):
        raise ValueError("Supplemental evidence must be in the ignored P2.3 source namespace")
    evidence_root = root / "runs" / "task_2_3_transaction_repair" / run_id
    data_root = root / "data_processed" / "task_2_3_transaction_repair" / run_id
    if evidence_root.exists() or data_root.exists():
        raise FileExistsError("P2.3 repair run already exists")
    capture = _read_json(source_path.parent / "source_capture.json")
    capture_path = source_path.parent / "source_capture.json"
    capture_sha = _sha(capture_path)
    dictionary_path = source_path.parent / "failure_properties_20261115.yaml"
    if (capture.get("response_sha256") != _sha(source_path)
            or capture.get("dictionary_sha256") != _sha(dictionary_path)
            or capture.get("source_url") !=
            "https://api.fdic.gov/banks/failures?fields=CERT%2CFAILDATE%2CCOST%2CRESTYPE%2CRESTYPE1&limit=10000&offset=0&sort_by=CERT&sort_order=ASC&format=json"):
        raise ValueError("Supplemental source capture or dictionary checksum differs")
    config = load_config(root / "config" / "financials_full_v2.json")
    old_summary, old_events, old_exits, reports, institutions, protected = _accepted_panel(root, panel_run)
    bundle_path = root / "runs/source_vintage_bundle.json"
    if bundle_path.exists():
        panel_manifest = RunManifest.read(root / "runs" / "task_2_2_panel" / panel_run / "run_manifest.json")
        if panel_manifest.config.get("source_vintage_bundle_sha256") != _sha(bundle_path):
            raise ValueError("Selected panel source-vintage lineage differs")
    source_rows, source_meta = _source_rows(source_path)
    if capture.get("api_total") != source_meta["total"] or capture.get("index") != source_meta["index"]:
        raise ValueError("Supplemental capture and API metadata disagree")
    linked, reconciliation = reconcile_events(old_events, source_rows)
    if reconciliation["accepted_ids"] != old_summary["events"]["source_events"] or len(old_exits) != len(reports):
        raise ValueError("Accepted event/bank denominator differs")
    source_manifest = RunManifest.read(root / "runs" / "task_1_5_full_real20261111b.json")
    prior_source = next(item for item in source_manifest.sources if item["endpoint"] == "failures")
    new_index = source_meta["index"]
    if (prior_source["api_total"] != source_meta["total"] or
            prior_source["index"] != [new_index["name"], new_index["createTimestamp"]]):
        raise ValueError("FDIC source total/index changed since accepted acquisition")
    boundary = date(2026, 6, 30)
    revised = classify_exits(reports, linked, institutions, boundary)
    old_by_cert = {row["CERT"]: row for row in old_exits}
    new_by_cert = {row["CERT"]: row for row in revised}
    if set(old_by_cert) != set(new_by_cert) or len(new_by_cert) != 24064:
        raise ValueError("Bank assessment keys changed")
    transitions = Counter((old_by_cert[cert]["EXIT_CATEGORY"], row["EXIT_CATEGORY"])
                          for cert, row in new_by_cert.items())
    events_by_cert: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in linked:
        if event["CERT"] is not None:
            events_by_cert[event["CERT"]].append(event)
    old_confirmed_dispositions = Counter(
        (new_by_cert[cert]["EXIT_CATEGORY"], event["TRANSACTION_DISPOSITION"])
        for cert, old in old_by_cert.items() if old["EXIT_CATEGORY"] == "confirmed_failure"
        for event in events_by_cert[cert])
    boundary_transitions = Counter((old_by_cert[cert]["EXIT_CATEGORY"], row["EXIT_CATEGORY"])
                          for cert, row in new_by_cert.items() if row["LAST_REPDTE"] == boundary)
    evidence_root.mkdir(parents=True)
    data_root.mkdir(parents=True)
    manifest = RunManifest.new(f"p2_3_repair_{run_id}", config)
    manifest.set_status("running")
    manifest.config.update({"source_panel_run": panel_run,
        "source_treatment_run": old_summary["source_treatment_run"],
        "source_panel_summary_sha256": protected[str(root / "runs" / "task_2_2_panel" / panel_run / "summary.json")],
        "supplemental_fields": SOURCE_FIELDS, "supplemental_capture_sha256": capture_sha})
    source_manifest_path = root / "runs" / "task_1_5_full_real20261111b.json"
    manifest.record_input(str(source_manifest_path), _sha(source_manifest_path))
    if bundle_path.exists():
        manifest.config["source_vintage_bundle_sha256"] = _sha(bundle_path)
        manifest.record_input(str(bundle_path), _sha(bundle_path))
    manifest.record_input(str(source_path), _sha(source_path), len(source_rows), source_meta["total"])
    manifest.record_input(str(dictionary_path), _sha(dictionary_path))
    manifest.record_input(str(capture_path), capture_sha)
    for path, digest in protected.items():
        manifest.record_input(path, digest)
    try:
        original = pq.read_table(root / "data_processed" / "task_2_2_panel" / panel_run /
                                 "failure_events" / "part-000.parquet")
        event_table = pa.Table.from_pylist(linked, schema=pa.schema(list(original.schema) + TRANSACTION_FIELDS))
        if not event_table.select(original.column_names).equals(original, check_metadata=False):
            raise ValueError("Accepted failure event values changed")
        exit_table = pa.Table.from_pylist(revised, schema=EXIT_SCHEMA)
        outputs = []
        for name, table in (("failure_events", event_table), ("exit_assessments", exit_table)):
            path = data_root / name / "part-000.parquet"
            digest = _write_verified(path, table)
            outputs.append({"dataset": name, "path": str(path), "rows": table.num_rows, "sha256": digest})
            manifest.record_output(path=str(path), dataset=name, rows=table.num_rows,
                sha256=digest, compression="ZSTD", schema_version="transaction_repair_v1",
                partition=None, source_total=table.num_rows, source_query=None)
        if any(_sha(Path(path)) != digest for path, digest in protected.items()):
            raise ValueError("Accepted panel or prior evidence changed during repair")
        if (_sha(source_path) != capture["response_sha256"] or
                _sha(dictionary_path) != capture["dictionary_sha256"] or
                _sha(capture_path) != capture_sha):
            raise ValueError("Supplemental FDIC evidence changed during repair")
        statuses = Counter(row["TRANSACTION_DISPOSITION"] for row in linked)
        failure_links = Counter(row["BANK_LINK_DISPOSITION"] for row in linked
            if row["TRANSACTION_DISPOSITION"] == "verified_failure")
        link_conflicts = sum(value for key, value in failure_links.items() if key != "chronological_candidate")
        transition_rows = [{"before": before, "after": after, "banks": count} for
            (before, after), count in sorted(transitions.items())]
        boundary_rows = [{"before": before, "after": after, "banks": count} for
            (before, after), count in sorted(boundary_transitions.items())]
        result = {"status": "complete", "run_id": run_id, "source_panel_run": panel_run,
            "source_sha256": _sha(source_path), "source_index": new_index,
            "source_total": source_meta["total"], "source_query_fields": SOURCE_FIELDS,
            "source_retrieved_utc": capture["retrieved_utc"],
            "dictionary_sha256": _sha(dictionary_path),
            "event_reconciliation": reconciliation,
            "transaction_dispositions": dict(sorted(statuses.items())),
            "verified_failure_bank_links": dict(sorted(failure_links.items())),
            "bank_link_conflicts_for_verified_failures": link_conflicts,
            "old_categories": {key: sum(row["EXIT_CATEGORY"] == key for row in old_exits) for key in CATEGORIES},
            "new_categories": {key: sum(row["EXIT_CATEGORY"] == key for row in revised) for key in CATEGORIES},
            "bank_transitions": transition_rows, "boundary_transitions": boundary_rows,
            "old_confirmed_by_new_category_and_transaction": [
                {"new_category": category, "transaction_disposition": disposition, "banks": count}
                for (category, disposition), count in sorted(old_confirmed_dispositions.items())],
            "new_confirmed_entrants": sum(row["EXIT_CATEGORY"] == "confirmed_failure" and
                old_by_cert[cert]["EXIT_CATEGORY"] != "confirmed_failure" for cert, row in new_by_cert.items()),
            "boundary_day_changed_banks": sum(row["banks"] for row in boundary_rows if row["before"] != row["after"]),
            "outputs": outputs, "accepted_panel_files_unchanged": 173,
            "accepted_panel_evidence_files_unchanged": 4,
            "panel_rows_unchanged": old_summary["panel_overall"]["output_rows"]}
        if (sum(statuses.values()) != 4117 or sum(row["banks"] for row in transition_rows) != 24064 or
                sum(old_confirmed_dispositions.values()) != result["old_categories"]["confirmed_failure"] or
                len(outputs) != 2 or any(value for key, value in reconciliation.items()
                  if key in {"missing_source_id", "duplicate_source_id", "corroboration_conflict", "supplement_extra_ids"})):
            raise ValueError("Transaction and bank reconciliation failed")
        _write_json(evidence_root / "summary.json", result)
        lines = ["# P2.3 Transaction and Exit Repair", "", "All counts are measured from the accepted panel and a separate FDIC response.", "",
                 "## Source-event transactions (denominator: 4,117 source IDs)", "",
                 "| Disposition | Events |", "| --- | ---: |"]
        lines.extend(f"| {key} | {count:,} |" for key, count in sorted(statuses.items()))
        lines += ["", "## Bank categories before and after (denominator: 24,064 distinct CERTs)", "",
                  "| Category | Previous banks | Repaired banks |", "| --- | ---: | ---: |"]
        lines.extend(f"| {key} | {result['old_categories'][key]:,} | {result['new_categories'][key]:,} |"
                     for key in CATEGORIES)
        lines += ["", "## Bank-category transitions (denominator: 24,064 distinct CERTs)", "",
                  "| Before | After | Banks |", "| --- | --- | ---: |"]
        lines.extend(f"| {row['before']} | {row['after']} | {row['banks']:,} |" for row in transition_rows)
        lines += ["", "## Previous confirmed-failure banks by transaction evidence", "",
                  "| New category | Transaction disposition | Banks |", "| --- | --- | ---: |"]
        lines.extend(f"| {row['new_category']} | {row['transaction_disposition']} | {row['banks']:,} |"
                     for row in result["old_confirmed_by_new_category_and_transaction"])
        lines += ["", "## Boundary-day transitions (last REPDTE = 2026-06-30)", "",
                  "| Before | After | Banks |", "| --- | --- | ---: |"]
        lines.extend(f"| {row['before']} | {row['after']} | {row['banks']:,} |" for row in boundary_rows)
        lines += ["", "FDIC RESTYPE is the failure/assistance authority. RESTYPE1 describes resolution method. A/A assistance does not establish a bank exit. OBAM appears in the live data but is outside the current dictionary enum; its method remains unverified. MGR is listed in FDIC Category 2 and qualifies with Failure status; REP remains uncertain for charter-exit assessment. Acquiring a failed bank through P&A is not an ordinary merger. All bank-link and timing decisions remain separate from transaction status.", ""]
        (evidence_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
        metadata = {"status": "complete", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "dependencies": dependency_versions(), "config": config.to_dict(),
            "summary_sha256": _sha(evidence_root / "summary.json"),
            "report_sha256": _sha(evidence_root / "report.md"),
            "protected_hashes": protected}
        _write_json(evidence_root / "metadata.json", metadata)
        manifest.record_validation("supplemental_source_total_index_and_id", True, "One-page FDIC response and accepted source fingerprint agree")
        manifest.record_validation("one_to_one_corrobation", True, "Every accepted source ID has one matching CERT/FAILDATE/COST")
        manifest.record_validation("event_bank_reconciliation", True, "4,117 events and 24,064 banks reconcile")
        manifest.record_validation("accepted_panel_immutable", True, "All 173 accepted Parquet and four evidence SHA-256 values unchanged")
        manifest.record_validation("output_read_back", True, "Two derived Parquet files validated")
        manifest.set_status("complete")
        manifest.write(evidence_root / "run_manifest.json")
        return result
    except Exception as exc:
        manifest.record_validation("p2_3_transaction_repair", False, f"{type(exc).__name__}: {exc}")
        manifest.set_status("failed")
        manifest.write(evidence_root / "run_manifest.json")
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Repair retrospective FDIC failure/assistance exit evidence")
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--project-root", type=Path, default=Path.cwd())
    runner.add_argument("--source-json", type=Path, required=True)
    runner.add_argument("--panel-run")
    runner.add_argument("--run-id", required=True)
    show = sub.add_parser("show")
    show.add_argument("--project-root", type=Path, default=Path.cwd())
    show.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        print(json.dumps(run(args.project_root, args.source_json, args.run_id, args.panel_run), indent=2))
    else:
        evidence = args.project_root / "runs" / "task_2_3_transaction_repair" / args.run_id
        manifest = RunManifest.read(evidence / "run_manifest.json")
        summary = _read_json(evidence / "summary.json")
        metadata = _read_json(evidence / "metadata.json")
        panel_run = summary.get("source_panel_run")
        if (manifest.status != "complete" or summary.get("status") != "complete" or
                manifest.run_id != f"p2_3_repair_{args.run_id}" or
                panel_run != manifest.config.get("source_panel_run") or
                not isinstance(panel_run, str) or
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}", panel_run) or
                metadata.get("status") != "complete" or
                metadata.get("summary_sha256") != _sha(evidence / "summary.json") or
                metadata.get("report_sha256") != _sha(evidence / "report.md") or
                not manifest.validations or
                any(not item["passed"] for item in manifest.validations)):
            parser.exit(2, f"P2.3 repair run {args.run_id}: evidence incomplete or inconsistent\n")
        try:
            _, _, _, _, _, input_hashes = _accepted_panel(args.project_root.resolve(), panel_run)
            if metadata.get("protected_hashes") != input_hashes:
                raise ValueError("Selected panel input hashes differ")
            for item in manifest.inputs:
                if item.get("sha256") != _sha(Path(item["path"])):
                    raise ValueError(f"Repair input checksum differs: {item['path']}")
            for item in manifest.outputs:
                path = Path(item["path"])
                if (not path.resolve().is_relative_to((args.project_root / "data_processed" / "task_2_3_transaction_repair" / args.run_id).resolve())
                        or item.get("sha256") != _sha(path)):
                    raise ValueError(f"Repair output checksum differs: {path}")
        except (OSError, ValueError, KeyError) as exc:
            parser.exit(2, f"P2.3 repair run {args.run_id}: {exc}\n")
        print(json.dumps({"status": "complete", "run_id": args.run_id,
            "transaction_dispositions": summary["transaction_dispositions"],
            "new_categories": summary["new_categories"],
            "report": str(evidence / "report.md")}, indent=2))


if __name__ == "__main__":
    main()
