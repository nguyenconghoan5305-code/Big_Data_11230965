"""Observed bank-quarter panel and evidence-limited exit reconciliation.

These outputs are retrospective audit artifacts, not point-in-time features or
four-quarter modelling labels. Only Financials records create panel rows.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from bft65b.acquisition.extended import full_plan
from bft65b.config import ProjectConfig, load_config
from bft65b.manifest import RunManifest, dependency_versions
from bft65b.quality.audit import _read_json, _sha, _write_json
from bft65b.quality.treatment import _completed_summary


ACCEPTED_TREATMENT = "treatment20261114a"
CONFIG = Path("config/financials_full_v2.json")


@dataclass(frozen=True)
class DocumentedExit:
    """A dated, independently documented exit event supplied to the classifier.

    The production runner supplies none: its accepted Institutions fields do
    not contain a verified change-code mapping plus an effective date.
    """

    cert: int
    kind: str  # merger or other_exit
    source: str
    source_id: str
    effective_date: date

    def __post_init__(self) -> None:
        if (type(self.cert) is not int or self.kind not in {"merger", "other_exit"}
                or not self.source or not self.source_id
                or not isinstance(self.effective_date, date)):
            raise ValueError("Documented exit requires CERT, kind, source ID, and effective date")


def _quarter_index(day: date) -> int:
    if (day.month, day.day) not in {(3, 31), (6, 30), (9, 30), (12, 31)}:
        raise ValueError(f"Not a report quarter-end: {day}")
    return day.year * 4 + day.month // 3 - 1


def reporting_gaps(reports: Mapping[int, Sequence[date]]) -> list[dict[str, Any]]:
    """Record resumed reporting gaps; never synthesize bank-quarter rows."""
    gaps = []
    for cert, unsorted in sorted(reports.items()):
        dates = sorted(unsorted)
        for previous, current in zip(dates, dates[1:]):
            missing = _quarter_index(current) - _quarter_index(previous) - 1
            if missing < 0:
                raise ValueError(f"Duplicate or reversed bank-quarter history: {cert}")
            if missing:
                gaps.append({"CERT": cert, "PREVIOUS_REPDTE": previous,
                             "NEXT_REPDTE": current,
                             "MISSING_INTERVENING_QUARTERS": missing})
    return gaps


def link_failure_events(events: Sequence[Mapping[str, Any]],
                        reports: Mapping[int, Sequence[date]]) -> list[dict[str, Any]]:
    """Preserve one source event per ID and describe chronological CERT links."""
    seen = set()
    result = []
    for source in events:
        source_id, cert, day = source.get("ID"), source.get("CERT"), source.get("FAILDATE")
        if not isinstance(source_id, str) or not source_id.strip() or source_id in seen:
            raise ValueError("Failures source ID must be nonempty and unique")
        seen.add(source_id)
        if not isinstance(day, date):
            raise ValueError(f"Invalid failure date for source ID {source_id}")
        dates = sorted(reports.get(cert, ())) if cert is not None else []
        prior = [value for value in dates if value <= day]
        later = [value for value in dates if value > day]
        if cert is None:
            disposition, rule = "missing_cert", "CERT is null; source ID remains the event key"
        elif not dates:
            disposition, rule = "unmatched_cert", "No Financials CERT in accepted scope"
        elif not prior:
            disposition, rule = "future_reports_only", "Every matching REPDTE is after FAILDATE"
        elif later:
            disposition, rule = "post_event_report_conflict", "Matching REPDTE occurs after FAILDATE"
        else:
            disposition, rule = "chronological_candidate", "At least one REPDTE <= FAILDATE; availability unverified"
        result.append({**source, "BANK_LINK_DISPOSITION": disposition,
                       "BANK_LINK_RULE": rule, "PRIOR_REPORT_COUNT": len(prior),
                       "POST_EVENT_REPORT_COUNT": len(later),
                       "LATEST_PRIOR_REPDTE": max(prior) if prior else None,
                       "EVENT_EVIDENCE_SOURCE": "FDIC Failures",
                       "EVENT_EVIDENCE_ID": source_id,
                       "EVENT_EVIDENCE_DATE": day})
    return result


def classify_exits(reports: Mapping[int, Sequence[date]],
                   linked_events: Sequence[Mapping[str, Any]],
                   institutions: Mapping[int, Mapping[str, Any]],
                   boundary: date,
                   documented: Sequence[DocumentedExit] = ()) -> list[dict[str, Any]]:
    """Assess each bank's last observed report without inferring an undated exit."""
    by_cert: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for event in linked_events:
        if event.get("CERT") is not None:
            by_cert[event["CERT"]].append(event)
    docs_by_cert: dict[int, list[DocumentedExit]] = defaultdict(list)
    for event in documented:
        docs_by_cert[event.cert].append(event)
    assessments = []
    for cert in sorted(reports):
        last = max(reports[cert])
        failures = by_cert.get(cert, [])
        docs = docs_by_cert.get(cert, [])
        refs = [{"source": "FDIC Financials", "identifier": str(cert),
                 "date": last.isoformat(), "rule": "last observed REPDTE; not an exit date"}]
        refs.extend({"source": "FDIC Failures", "identifier": event["ID"],
                     "date": event["FAILDATE"].isoformat(),
                     "rule": event["BANK_LINK_DISPOSITION"],
                     "transaction": event.get("TRANSACTION_DISPOSITION", "unverified")}
                    for event in sorted(failures, key=lambda value: value["ID"]))
        refs.extend({"source": item.source, "identifier": item.source_id,
                     "date": item.effective_date.isoformat(), "rule": item.kind}
                    for item in sorted(docs, key=lambda value: (value.effective_date, value.source_id)))
        eligible_failures = [event for event in failures
            if event.get("TRANSACTION_DISPOSITION") == "verified_failure"
            and event["BANK_LINK_DISPOSITION"] == "chronological_candidate"
            and last <= event["FAILDATE"] <= boundary]
        eligible_docs = [event for event in docs if last <= event.effective_date <= boundary]
        if len(failures) == len(eligible_failures) == 1 and not docs:
            category, rule = "confirmed_failure", "One verified FDIC failure transaction with valid chronological bank link"
        elif not failures and len(docs) == len(eligible_docs) == 1:
            category = "documented_merger" if docs[0].kind == "merger" else "documented_other_exit"
            rule = "One independently documented dated exit after last report"
        elif last == boundary and not docs and all(
                event.get("TRANSACTION_DISPOSITION", "unverified") in
                {"assistance_transaction", "open_bank_assistance_code_unverified"}
                for event in failures):
            category, rule = "boundary_observed", "No qualifying exit evidence at data boundary"
        else:
            category = "ambiguous_exit"
            rule = ("Conflicting or nonqualifying event evidence" if failures or docs else
                    "Last observed report has no dated exit evidence; absence is not proof of exit")
        snapshot = institutions.get(cert)
        assessments.append({"CERT": cert, "LAST_REPDTE": last,
            "EXIT_CATEGORY": category, "CLASSIFICATION_RULE": rule,
            "EVIDENCE_JSON": json.dumps(refs, sort_keys=True, separators=(",", ":")),
            "INSTITUTION_ACTIVE_SNAPSHOT": snapshot.get("ACTIVE") if snapshot else None,
            "INSTITUTION_CHANGEC1_SNAPSHOT": snapshot.get("CHANGEC1") if snapshot else None,
            "INSTITUTION_SOURCE_MISSING_FIELDS": snapshot.get("SOURCE_MISSING_FIELDS") if snapshot else None,
            "SNAPSHOT_AUDIT_ONLY": True})
    return assessments


GAP_SCHEMA = pa.schema([
    pa.field("CERT", pa.int64(), nullable=False),
    pa.field("PREVIOUS_REPDTE", pa.date32(), nullable=False),
    pa.field("NEXT_REPDTE", pa.date32(), nullable=False),
    pa.field("MISSING_INTERVENING_QUARTERS", pa.int32(), nullable=False),
])
EXIT_SCHEMA = pa.schema([
    pa.field("CERT", pa.int64(), nullable=False),
    pa.field("LAST_REPDTE", pa.date32(), nullable=False),
    pa.field("EXIT_CATEGORY", pa.string(), nullable=False),
    pa.field("CLASSIFICATION_RULE", pa.string(), nullable=False),
    pa.field("EVIDENCE_JSON", pa.string(), nullable=False),
    pa.field("INSTITUTION_ACTIVE_SNAPSHOT", pa.int8()),
    pa.field("INSTITUTION_CHANGEC1_SNAPSHOT", pa.string()),
    pa.field("INSTITUTION_SOURCE_MISSING_FIELDS", pa.list_(pa.field("element", pa.string()))),
    pa.field("SNAPSHOT_AUDIT_ONLY", pa.bool_(), nullable=False),
])
EVENT_EXTRA_FIELDS = [
    pa.field("BANK_LINK_DISPOSITION", pa.string(), nullable=False),
    pa.field("BANK_LINK_RULE", pa.string(), nullable=False),
    pa.field("PRIOR_REPORT_COUNT", pa.int32(), nullable=False),
    pa.field("POST_EVENT_REPORT_COUNT", pa.int32(), nullable=False),
    pa.field("LATEST_PRIOR_REPDTE", pa.date32()),
    pa.field("EVENT_EVIDENCE_SOURCE", pa.string(), nullable=False),
    pa.field("EVENT_EVIDENCE_ID", pa.string(), nullable=False),
    pa.field("EVENT_EVIDENCE_DATE", pa.date32(), nullable=False),
]


def _write_verified(path: Path, table: pa.Table) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Panel output already exists: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table, temporary, compression="zstd", compression_level=3)
        with pq.ParquetFile(temporary) as parquet:
            observed = parquet.read()
            if parquet.metadata.num_rows != table.num_rows:
                raise ValueError(f"Panel read-back row count differs: {path}")
            for group in range(parquet.metadata.num_row_groups):
                for column in range(parquet.metadata.row_group(group).num_columns):
                    if parquet.metadata.row_group(group).column(column).compression != "ZSTD":
                        raise ValueError(f"Panel codec differs: {path}")
        if not observed.equals(table, check_metadata=False):
            raise ValueError(f"Panel read-back differs: {path}")
        digest = _sha(Path(temporary))
        os.replace(temporary, path)
        if _sha(path) != digest:
            raise ValueError(f"Panel checksum differs after promotion: {path}")
        with pq.ParquetFile(path) as parquet:
            if not parquet.read().equals(table, check_metadata=False):
                raise ValueError(f"Promoted panel output differs on read-back: {path}")
        return digest
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _config_matches(recorded: dict[str, Any], config: ProjectConfig) -> bool:
    """Allow recorded lineage fields without confusing JSON lists with Python tuples."""
    return all(json.dumps(recorded.get(key), sort_keys=True) == json.dumps(value, sort_keys=True)
               for key, value in config.to_dict().items())


def _accepted_inputs(root: Path, config: ProjectConfig, treatment_run: str) -> tuple[dict, list[dict], RunManifest]:
    evidence = root / "runs" / "task_2_1_treatment" / treatment_run
    summary = _completed_summary(evidence, treatment_run)
    metadata = _read_json(evidence / "metadata.json")
    manifest = RunManifest.read(evidence / "run_manifest.json")
    data_root = (root / "data_processed" / "task_2_1_treatment" / treatment_run).resolve()
    quarters = set(full_plan(config).quarters)
    items = summary.get("outputs", [])
    if (metadata.get("data_root") != str(data_root)
            or summary.get("accepted_attempt") != "real20261111b"
            or not _config_matches(manifest.config, config)
            or len(items) != 172 or len(quarters) != 170):
        raise ValueError("Accepted treatment scope or configuration differs")
    output_by_path = {item["path"]: item for item in manifest.outputs}
    if len(output_by_path) != 172:
        raise ValueError("Treatment manifest output paths are not unique")
    seen = set()
    verified = []
    for item in items:
        dataset, quarter = item["dataset"], item["quarter"]
        identity = dataset, quarter
        if identity in seen or (dataset == "financials" and quarter not in quarters):
            raise ValueError(f"Repeated or unexpected treatment input: {identity}")
        seen.add(identity)
        relative = Path(item["relative_path"])
        path = (data_root / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts
                or not path.is_relative_to(data_root)
                or item.get("quality_schema_version") != "flag_only_v1"
                or item.get("codec") != "ZSTD"):
            raise ValueError(f"Unsafe or mismatched treatment input: {identity}")
        recorded = output_by_path.get(str(path))
        if not recorded or recorded.get("sha256") != item["derived_sha256"] or recorded.get("rows") != item["rows"]:
            raise ValueError(f"Treatment manifest and summary disagree: {identity}")
        if _sha(path) != item["derived_sha256"]:
            raise ValueError(f"Treatment input checksum differs: {identity}")
        with pq.ParquetFile(path) as parquet:
            if (parquet.metadata.num_rows != item["rows"]
                    or parquet.schema_arrow.metadata.get(b"bft65b.quality_treatment") != b"flag_only_v1"
                    or "DQ_FLAGS" not in parquet.schema_arrow.names):
                raise ValueError(f"Treatment input schema or rows differ: {identity}")
            for group in range(parquet.metadata.num_row_groups):
                for column in range(parquet.metadata.row_group(group).num_columns):
                    if parquet.metadata.row_group(group).column(column).compression != "ZSTD":
                        raise ValueError(f"Treatment input codec differs: {identity}")
        verified.append({**item, "path": str(path)})
    if seen != {("financials", q) for q in quarters} | {("failures", None), ("institutions", None)}:
        raise ValueError("Treatment input plan does not cover 170 quarters and two tables")
    return summary, verified, manifest


def _panel_reconciliation(quarter: str, source: pa.Table, result: pa.Table) -> dict[str, Any]:
    before = list(zip(source["CERT"].to_pylist(), source["REPDTE"].to_pylist()))
    after = list(zip(result["CERT"].to_pylist(), result["REPDTE"].to_pylist()))
    input_keys, output_keys = set(before), set(after)
    day = date(int(quarter[:4]), int(quarter[4:6]), int(quarter[6:]))
    if any(cert is None or report != day for cert, report in before + after):
        raise ValueError(f"Financials key or partition date differs: {quarter}")
    counts = {"quarter": quarter, "input_rows": len(before), "input_keys": len(input_keys),
        "output_rows": len(after), "output_keys": len(output_keys),
        "input_duplicate_keys": len(before) - len(input_keys),
        "output_duplicate_keys": len(after) - len(output_keys),
        "missing_keys": len(input_keys - output_keys), "added_keys": len(output_keys - input_keys)}
    if (any(counts[key] for key in ("input_duplicate_keys", "output_duplicate_keys",
                                     "missing_keys", "added_keys"))
            or not source.equals(result, check_metadata=True)):
        raise ValueError(f"Panel differs from treated Financials input: {quarter}: {counts}")
    return counts


def _report(summary: Mapping[str, Any]) -> str:
    panel = summary["panel_overall"]
    lines = ["# P2.2/P2.3 Observed Panel and Exit Evidence", "",
        "The panel contains observed Financials bank-quarters only. Retrospective event, gap, and snapshot assessments are separate audit outputs, not point-in-time features or labels.", "",
        "## Panel reconciliation (unit: bank-quarters and distinct bank-quarter keys)", "",
        "| Period | Input rows | Input keys | Output rows | Output keys | Input dupes | Output dupes | Missing keys | Added keys |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| All 170 quarters | {panel['input_rows']:,} | {panel['input_keys']:,} | {panel['output_rows']:,} | {panel['output_keys']:,} | {panel['input_duplicate_keys']:,} | {panel['output_duplicate_keys']:,} | {panel['missing_keys']:,} | {panel['added_keys']:,} |"]
    for quarter in summary["panel_by_quarter"]:
        lines.append("| {quarter} | {input_rows:,} | {input_keys:,} | {output_rows:,} | {output_keys:,} | {input_duplicate_keys:,} | {output_duplicate_keys:,} | {missing_keys:,} | {added_keys:,} |".format(**quarter))
    lines += ["", "## Source events and bank-link disposition (unit: distinct Failures source IDs)", "",
        "| Disposition | Events | Denominator |",
        "| --- | ---: | ---: |"]
    for name, count in sorted(summary["events"]["link_dispositions"].items()):
        lines.append(f"| {name} | {count:,} | {summary['events']['source_events']:,} |")
    lines += ["", "## Last-report exit assessments (unit: distinct Financials CERTs)", "",
        "| Category | Banks | Denominator |",
        "| --- | ---: | ---: |"]
    for name, count in sorted(summary["exits"]["categories"].items()):
        lines.append(f"| {name} | {count:,} | {summary['exits']['bank_denominator']:,} |")
    lines += ["", f"Resumed reporting gaps: {summary['gaps']['gap_events']:,} gaps across {summary['gaps']['distinct_banks']:,} banks; missing intervening quarters total {summary['gaps']['missing_quarters']:,}. These gaps do not create panel rows or prove exits.", "",
        "Institutions ACTIVE and CHANGEC1 are retained as snapshot audit fields only. The accepted Institutions selection lacks a verified structural-event mapping plus an applicable effective date; no merger or other exit is confirmed from it. A last report before the data boundary without dated corroboration is ambiguous, not a proven exit. Failure-source events with missing or uncertain CERT remain in the event table.", "",
        "No 120-day availability join, prediction-date feature, or four-quarter target is created in this milestone.", ""]
    return "\n".join(lines)


def build(root: Path, config_file: Path, treatment_run: str, run_id: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    for value in (treatment_run, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}", value):
            raise ValueError("Run IDs must be short safe identifiers")
    config_path = config_file if config_file.is_absolute() else root / config_file
    config = load_config(config_path)
    source_summary, inventory, treatment_manifest = _accepted_inputs(root, config, treatment_run)
    data_root = root / "data_processed" / "task_2_2_panel" / run_id
    evidence_root = root / "runs" / "task_2_2_panel" / run_id
    if data_root.exists() or evidence_root.exists():
        raise FileExistsError("Panel run already exists; use a fresh run ID")
    data_root.mkdir(parents=True)
    evidence_root.mkdir(parents=True)
    manifest = RunManifest.new(f"p2_2_panel_{run_id}", config)
    manifest.config["source_treatment_run"] = treatment_run
    manifest.config["source_treatment_summary_sha256"] = _sha(
        root / "runs" / "task_2_1_treatment" / treatment_run / "summary.json")
    bundle_path = root / "runs/source_vintage_bundle.json"
    if bundle_path.exists():
        if (source_summary.get("accepted_attempt") != "real20261111b"
                or treatment_manifest.config.get("source_vintage_bundle_sha256") != _sha(bundle_path)):
            raise ValueError("Source-vintage treatment lineage differs")
        manifest.config["source_vintage_bundle_sha256"] = _sha(bundle_path)
    manifest.set_status("running")
    for item in inventory:
        manifest.record_input(item["path"], item["derived_sha256"], item["rows"], item["rows"])
    indexed = {(item["dataset"], item["quarter"]): item for item in inventory}
    outputs = []
    reconciliations = []
    reports: dict[int, list[date]] = defaultdict(list)
    started = datetime.now(timezone.utc).isoformat()
    try:
        for quarter in full_plan(config).quarters:
            item = indexed[("financials", quarter)]
            with pq.ParquetFile(item["path"]) as parquet:
                treated = parquet.read()
            for cert, day in zip(treated["CERT"].to_pylist(), treated["REPDTE"].to_pylist()):
                reports[cert].append(day)
            target_rel = Path("financials") / f"REPDTE={quarter}" / "part-000.parquet"
            target = data_root / target_rel
            digest = _write_verified(target, treated)
            with pq.ParquetFile(target) as parquet:
                panel = parquet.read()
            reconciliations.append(_panel_reconciliation(quarter, treated, panel))
            outputs.append({"dataset": "financials", "quarter": quarter, "relative_path": target_rel.as_posix(),
                            "rows": panel.num_rows, "sha256": digest})
            manifest.record_output(path=str(target), dataset="financials", rows=panel.num_rows,
                sha256=digest, compression="ZSTD", schema_version="observed_panel_v1",
                partition=quarter, source_total=treated.num_rows, source_query=None)
        with pq.ParquetFile(indexed[("institutions", None)]["path"]) as parquet:
            inst_table = parquet.read()
        inst_rows = inst_table.to_pylist()
        institutions = {row["CERT"]: row for row in inst_rows}
        if len(institutions) != len(inst_rows):
            raise ValueError("Institutions CERT is not unique")
        with pq.ParquetFile(indexed[("failures", None)]["path"]) as parquet:
            failure_table = parquet.read()
        failure_rows = failure_table.to_pylist()
        linked = link_failure_events(failure_rows, reports)
        event_schema = pa.schema(list(failure_table.schema) + EVENT_EXTRA_FIELDS,
                                 metadata=failure_table.schema.metadata)
        event_table = pa.Table.from_pylist(linked, schema=event_schema)
        if (event_table.num_rows != len(failure_rows)
                or not event_table.select(failure_table.column_names).equals(failure_table, check_metadata=False)):
            raise ValueError("Failure source events or values changed")
        final_quarter = full_plan(config).quarters[-1]
        boundary = date(int(final_quarter[:4]), int(final_quarter[4:6]), int(final_quarter[6:]))
        exits = classify_exits(reports, linked, institutions, boundary)
        gaps = reporting_gaps(reports)
        extra = [
            ("failure_events", event_table),
            ("exit_assessments", pa.Table.from_pylist(exits, schema=EXIT_SCHEMA)),
            ("reporting_gaps", pa.Table.from_pylist(gaps, schema=GAP_SCHEMA)),
        ]
        for name, table in extra:
            target_rel = Path(name) / "part-000.parquet"
            target = data_root / target_rel
            digest = _write_verified(target, table)
            outputs.append({"dataset": name, "quarter": None, "relative_path": target_rel.as_posix(),
                            "rows": table.num_rows, "sha256": digest})
            manifest.record_output(path=str(target), dataset=name, rows=table.num_rows,
                sha256=digest, compression="ZSTD", schema_version="panel_evidence_v1",
                partition=None, source_total=table.num_rows, source_query=None)
        overall = {key: sum(period[key] for period in reconciliations) for key in (
            "input_rows", "input_keys", "output_rows", "output_keys",
            "input_duplicate_keys", "output_duplicate_keys", "missing_keys", "added_keys")}
        if (len(reconciliations) != 170 or overall["input_rows"] != source_summary["datasets"]["financials"]["after_rows"]
                or overall["input_rows"] != overall["output_rows"]
                or overall["input_keys"] != overall["output_keys"]
                or any(overall[key] for key in ("input_duplicate_keys", "output_duplicate_keys", "missing_keys", "added_keys"))
                or len(linked) != source_summary["datasets"]["failures"]["after_rows"]
                or len(exits) != len(reports) or len(outputs) != 173):
            raise ValueError("Panel, event, or source-count reconciliation failed")
        event_counts = dict(sorted(Counter(row["BANK_LINK_DISPOSITION"] for row in linked).items()))
        exit_counts = {name: 0 for name in ("confirmed_failure", "documented_merger",
                                            "documented_other_exit", "ambiguous_exit",
                                            "boundary_observed")}
        exit_counts.update(Counter(row["EXIT_CATEGORY"] for row in exits))
        exit_counts = dict(sorted(exit_counts.items()))
        if sum(event_counts.values()) != len(linked) or sum(exit_counts.values()) != len(exits):
            raise ValueError("Event/exit category counts do not reconcile")
        for item in inventory:
            if _sha(Path(item["path"])) != item["derived_sha256"]:
                raise ValueError(f"Treatment input changed during panel build: {item['dataset']}:{item['quarter']}")
        summary = {"status": "complete", "source_treatment_run": treatment_run,
            "source_treatment_summary_sha256": _sha(root / "runs" / "task_2_1_treatment" / treatment_run / "summary.json"),
            "panel_by_quarter": reconciliations, "panel_overall": overall,
            "events": {"source_events": len(linked), "distinct_source_ids": len({row["ID"] for row in linked}),
                       "nonnull_cert_events": sum(row["CERT"] is not None for row in linked),
                       "distinct_nonnull_certs": len({row["CERT"] for row in linked if row["CERT"] is not None}),
                       "link_dispositions": event_counts,
                       "multiple_prior_report_events": sum(row["PRIOR_REPORT_COUNT"] > 1 for row in linked)},
            "exits": {"bank_denominator": len(reports), "categories": exit_counts,
                      "documented_external_exit_events_used": 0,
                      "snapshot_change_code_used_for_classification": False},
            "gaps": {"gap_events": len(gaps), "distinct_banks": len({row["CERT"] for row in gaps}),
                     "missing_quarters": sum(row["MISSING_INTERVENING_QUARTERS"] for row in gaps)},
            "outputs": outputs, "source_files": len(inventory), "output_files": len(outputs),
            "source_hashes_unchanged": True,
            "data_boundary": final_quarter,
            "unit_note": "Panel counts are bank-quarters; failure counts are source IDs; exit assessments are distinct Financials CERTs."}
        _write_json(evidence_root / "summary.json", summary)
        report_path = evidence_root / "report.md"
        report_path.write_text(_report(summary), encoding="utf-8")
        metadata = {"status": "complete", "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": f"python -m bft65b.panel.build run --project-root {root} --config {config_file} --treatment-run {treatment_run} --run-id {run_id}",
            "os": platform.platform(), "python": platform.python_version(),
            "dependencies": dependency_versions(), "config": config.to_dict(),
            "source_treatment_run": treatment_run, "input_hashes": inventory,
            "summary_sha256": _sha(evidence_root / "summary.json"),
            "report_sha256": _sha(report_path), "data_root": str(data_root)}
        _write_json(evidence_root / "metadata.json", metadata)
        manifest.record_validation("accepted_treatment_inputs", True, "172 checksums and schemas verified")
        manifest.record_validation("panel_key_reconciliation", True, "170 quarter and overall key sets match")
        manifest.record_validation("events_and_exits", True, "Source IDs and bank assessments reconcile")
        manifest.record_validation("output_read_back", True, "173 outputs validated after atomic promotion")
        manifest.record_validation("source_hashes_unchanged", True, "172 treated inputs rehashed after build")
        manifest.set_status("complete")
        manifest.write(evidence_root / "run_manifest.json")
        return {"status": "PASS", "run_id": run_id, "source_treatment_run": treatment_run,
            "evidence_root": str(evidence_root), "data_root": str(data_root),
            "panel_rows": overall["output_rows"], "failure_events": len(linked),
            "exit_assessments": len(exits), "gap_events": len(gaps),
            "summary_sha256": metadata["summary_sha256"]}
    except Exception as exc:
        manifest.record_validation("p2_2_p2_3_panel", False, f"{type(exc).__name__}: {exc}")
        manifest.set_status("failed")
        manifest.write(evidence_root / "run_manifest.json")
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Observed Financials panel and evidence-limited exit reconciliation")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--project-root", type=Path, default=Path.cwd())
    run.add_argument("--config", type=Path, default=CONFIG)
    run.add_argument("--treatment-run", default=ACCEPTED_TREATMENT)
    run.add_argument("--run-id", required=True)
    show = sub.add_parser("show")
    show.add_argument("--project-root", type=Path, default=Path.cwd())
    show.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        print(json.dumps(build(args.project_root, args.config, args.treatment_run, args.run_id), indent=2))
    else:
        evidence = args.project_root.resolve() / "runs" / "task_2_2_panel" / args.run_id
        manifest = RunManifest.read(evidence / "run_manifest.json")
        metadata = _read_json(evidence / "metadata.json")
        summary = _read_json(evidence / "summary.json")
        if (manifest.run_id != f"p2_2_panel_{args.run_id}" or manifest.status != "complete"
                or not manifest.validations or any(not item.get("passed") for item in manifest.validations)
                or metadata.get("status") != "complete" or summary.get("status") != "complete"
                or metadata.get("summary_sha256") != _sha(evidence / "summary.json")
                or len(manifest.outputs) != summary.get("output_files")):
            parser.exit(2, f"Panel run {args.run_id!r}: saved evidence is incomplete or inconsistent\n")
        print(json.dumps({"status": "complete", "source_treatment_run": summary["source_treatment_run"],
            "panel_overall": summary["panel_overall"], "events": summary["events"],
            "exits": summary["exits"], "gaps": summary["gaps"],
            "report": str(evidence / "report.md")}, indent=2))


if __name__ == "__main__":
    main()
