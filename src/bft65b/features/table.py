"""Production P4.3 feature–label table from accepted, immutable evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq

from bft65b.config import load_config
from bft65b.features.contract import (
    AMOUNT, CONTRACT_VERSION, DERIVED_PERCENT, FEATURE_NAMES, FEATURES,
    FEATURE_INPUTS, MAX_PREDICTION_Q, _quarter_end_on_or_after,
    compute_features, validate_schema,
)
from bft65b.features.preprocessing import PREDICTOR_COLUMNS, PREPROCESSING_VERSION
from bft65b.features.production import verify_saved as verify_labels
from bft65b.manifest import RunManifest
from bft65b.panel.asof import (
    AVAILABILITY_LAG_DAYS, FinancialReport, PREDICTOR_FIELDS,
    availability_date, population_from_available_reports,
)
from bft65b.panel.build import _write_verified
from bft65b.panel.prediction import verify_saved as verify_pit
from bft65b.panel.repair import _accepted_panel
from bft65b.panel.retrospective import verify_saved as verify_membership
from bft65b.quality.audit import _read_json, _sha, _write_json


TABLE_VERSION = "feature_label_v1"
PANEL_RUN = "panel20261115b"
PIT_RUN = "pit20261115f"
CONTRACT_RUN = "featurecontract20261116b"
MEMBERSHIP_RUN = "membership20261120a"
LABEL_RUN = "labels20261120a"
REVIEW_RUN = "p42review20261120b"
NAMESPACE = "task_4_3_feature_table"
SOURCE_COLUMNS = ("CERT", "REPDTE", "ID", "SOURCE_MISSING_FIELDS", *sorted(FEATURE_INPUTS))
PIT_COLUMNS = ("CERT", "Q", "SELECTED_REPDTE", "AVAILABILITY_DATE",
               "REPORT_AGE_DAYS", "AVAILABILITY_BASIS", "SOURCE_ID_DIAGNOSTIC",
               "SOURCE_MISSING_FIELDS_DIAGNOSTIC", *sorted(FEATURE_INPUTS))
FEATURE_FIELDS = tuple(pa.field(spec.name,
    AMOUNT if spec.name == "asset_size" else
    pa.float64() if spec.output_type == "float64" else DERIVED_PERCENT)
    for spec in FEATURES)
MODEL_SCHEMA = pa.schema([
    pa.field("CERT", pa.int64(), nullable=False), pa.field("Q", pa.date32(), nullable=False),
    pa.field("Y", pa.int8(), nullable=False), *FEATURE_FIELDS,
], metadata={b"bft65b.table": TABLE_VERSION.encode(),
             b"bft65b.predictors": ",".join(FEATURE_NAMES).encode()})
CANDIDATE_SCHEMA = pa.schema([
    pa.field("CERT", pa.int64(), nullable=False), pa.field("Q", pa.date32(), nullable=False),
    *FEATURE_FIELDS,
    pa.field("SELECTED_REPDTE", pa.date32(), nullable=False),
    pa.field("AVAILABILITY_DATE", pa.date32(), nullable=False),
    pa.field("REPORT_AGE_DAYS", pa.int32(), nullable=False),
    pa.field("AVAILABILITY_BASIS", pa.string(), nullable=False),
    pa.field("SOURCE_ID_DIAGNOSTIC", pa.string()),
    pa.field("SOURCE_MISSING_FIELDS_DIAGNOSTIC", pa.list_(pa.string())),
    pa.field("FEATURE_MISSING_REASONS", pa.list_(pa.string()), nullable=False),
    pa.field("FEATURE_LINEAGE_JSON", pa.string(), nullable=False),
], metadata={b"bft65b.table": TABLE_VERSION.encode(),
             b"bft65b.predictors": ",".join(FEATURE_NAMES).encode(),
             b"bft65b.audit_columns_are_not_predictors": b"true"})
if FEATURE_NAMES != PREDICTOR_COLUMNS:
    raise RuntimeError("Feature dictionary and preprocessing predictor order differ")


@dataclass(frozen=True)
class Anchor:
    cert: int
    repdte: date
    available: date
    source_id: str | None
    source_missing: tuple[str, ...]
    source_values: dict[str, object]
    features: dict[str, object]
    missing_reasons: tuple[str, ...]
    feature_lineage_json: str


@dataclass
class ReportHistoryState:
    """Advance the production report cache only as source quarters become usable."""

    sources: list[dict]
    pointer: int = 0
    scanned_reports: int = 0
    histories: dict[int, deque[FinancialReport]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=4)))
    anchors: dict[int, Anchor] = field(default_factory=dict)
    last_q: date | None = None

    def __post_init__(self) -> None:
        dates = [_day(item["quarter"]) for item in self.sources]
        if dates != sorted(set(dates)):
            raise ValueError("Source quarters must be unique and in calendar order")

    def advance(self, q: date, load_source_rows) -> None:
        if _quarter_end_on_or_after(q) != q or (
                self.last_q is not None and q < self.last_q):
            raise ValueError("Prediction quarters must advance in calendar order")
        while self.pointer < len(self.sources) and (
                _day(self.sources[self.pointer]["quarter"]) +
                timedelta(days=AVAILABILITY_LAG_DAYS) <= q):
            source = self.sources[self.pointer]
            source_rows = load_source_rows(source)
            if len(source_rows) != source["rows"]:
                raise ValueError(f"Observed source row count differs: {source['quarter']}")
            seen_source = set()
            for row in source_rows:
                cert = row["CERT"]
                if cert in seen_source or row["REPDTE"] != _day(source["quarter"]):
                    raise ValueError(f"Duplicate or wrong-quarter observed report: {source['quarter']}")
                seen_source.add(cert)
                report = _report_from_panel(row)
                history = self.histories[cert]
                history.append(report)
                self.anchors[cert] = _anchor_from_history(history, q)
            self.scanned_reports += len(source_rows)
            self.pointer += 1
        self.last_q = q


def _day(value: str) -> date:
    return date(int(value[:4]), int(value[4:6]), int(value[6:8]))


def _run_id(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}", value):
        raise ValueError("Unsafe P4.3 run ID")


def _verified_inputs(root: Path, *, panel_run: str = PANEL_RUN,
                     pit_run: str = PIT_RUN, contract_run: str = CONTRACT_RUN,
                     membership_run: str = MEMBERSHIP_RUN, label_run: str = LABEL_RUN,
                     review_run: str = REVIEW_RUN) -> tuple[dict, dict, dict, dict[str, str]]:
    """Use the accepted validators; never infer acceptance from a status string."""
    portable = (root / "runs/source_vintage_bundle.json").is_file()
    if portable and any(selected == historical for selected, historical in (
            (panel_run, PANEL_RUN), (pit_run, PIT_RUN),
            (contract_run, CONTRACT_RUN), (membership_run, MEMBERSHIP_RUN),
            (label_run, LABEL_RUN), (review_run, REVIEW_RUN))):
        raise ValueError("Portable feature table requires explicit upstream run selection")
    panel, _, _, _, _, panel_hashes = _accepted_panel(root, panel_run)
    pit = verify_pit(root, pit_run, check_upstream=True)
    verify_membership(root, membership_run)
    verified_labels = verify_labels(root, label_run)
    if (pit["candidate_rows"] != 3697094 or verified_labels["quarters"] != 166):
        raise ValueError("Accepted PIT or label scope differs")
    if portable:
        pit_manifest = RunManifest.read(root / "runs/task_2_5_prediction" / pit_run /
                                        "run_manifest.json")
        member_manifest = RunManifest.read(root / "runs/task_4_2_membership" /
                                           membership_run / "run_manifest.json")
        label_manifest = RunManifest.read(root / "runs/task_4_2_production_labels" /
                                          label_run / "run_manifest.json")
        if (pit_manifest.config.get("source_panel_run") != panel_run or
                member_manifest.config.get("source_panel_run") != panel_run or
                member_manifest.config.get("source_candidate_run") != pit_run or
                label_manifest.config.get("membership_run") != membership_run):
            raise ValueError("Portable feature table upstream lineage differs")
    contract_root = root / "runs" / "task_4_1_feature_contract" / contract_run
    contract = _read_json(contract_root / "contract.json")
    contract_manifest = RunManifest.read(contract_root / "run_manifest.json")
    if (contract_manifest.status != "complete" or not contract_manifest.validations
            or any(not item["passed"] for item in contract_manifest.validations)
            or contract.get("status") != "complete" or contract.get("contract_version") != CONTRACT_VERSION
            or tuple(item["name"] for item in contract["features"]) != FEATURE_NAMES
            or contract_manifest.outputs[0]["sha256"] != _sha(contract_root / "contract.json")):
        raise ValueError("Accepted seven-feature contract is incomplete")
    if (contract.get("panel_run") != panel_run or contract.get("pit_run") != pit_run):
        raise ValueError("Feature contract panel/PIT lineage differs")
    if portable and (contract_manifest.config.get("source_panel_run") != panel_run or
                     contract_manifest.config.get("source_pit_run") != pit_run):
        raise ValueError("Portable feature contract manifest lineage differs")
    for item in contract_manifest.inputs:
        if portable and not Path(item["path"]).resolve().is_relative_to(root):
            raise ValueError("Portable feature contract input escaped root")
        if item["sha256"] and _sha(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Feature-contract upstream checksum differs: {item['path']}")
    review_path = root / "runs" / "task_4_2_review" / review_run / "summary.json"
    review = _read_json(review_path)
    if (review.get("membership_run") != membership_run
            or review.get("label_run") != label_run
            or review.get("repeat_comparison", {}).get("membership_output_hash_differences") != 0
            or review.get("repeat_comparison", {}).get("label_output_hash_differences") != 0):
        raise ValueError("Accepted P4.2 repeat reconciliation differs")
    if portable and review.get("candidate_run") != pit_run:
        raise ValueError("Portable P4.2 review/PIT lineage differs")
    if not portable and _sha(review_path) != "f55b7a39c8eb182c02ec77c976aacf2ab45baadbaed2662c57c5d4747e69fb55":
        raise ValueError("Accepted P4.2 review checksum differs")
    pit_summary = _read_json(root / "runs" / "task_2_5_prediction" / pit_run / "summary.json")
    label_summary = _read_json(root / "runs" / "task_4_2_production_labels" / label_run / "summary.json")
    for summary in (pit_summary, label_summary):
        if len(summary["outputs"]) != 166:
            raise ValueError("Upstream quarter inventory differs")
    source_hashes = dict(panel_hashes)
    for name, run_path in (("pit", root / "runs" / "task_2_5_prediction" / pit_run),
                           ("membership", root / "runs" / "task_4_2_membership" / membership_run),
                           ("labels", root / "runs" / "task_4_2_production_labels" / label_run)):
        for suffix in ("summary.json", "metadata.json", "run_manifest.json"):
            path = run_path / suffix
            source_hashes[str(path)] = _sha(path)
    for path in (contract_root / "contract.json", contract_root / "run_manifest.json", review_path):
        source_hashes[str(path)] = _sha(path)
    return panel, pit_summary, label_summary, source_hashes


def _report_from_panel(row: dict) -> FinancialReport:
    values = {field: None for field in PREDICTOR_FIELDS}
    for field in FEATURE_INPUTS:
        values[field] = row[field]
    values["CERT"] = row["CERT"]
    values["REPDTE"] = row["REPDTE"]
    values["ID"] = row["ID"]
    values["SOURCE_MISSING_FIELDS"] = row["SOURCE_MISSING_FIELDS"]
    return FinancialReport.from_panel_row(values)


def _anchor_from_history(history: deque[FinancialReport], q: date) -> Anchor:
    reports = tuple(history)
    current = reports[-1]
    roster = population_from_available_reports(q, reports)
    if len(roster) != 1:
        raise ValueError("Newly available source report did not create one anchor")
    result = compute_features(roster[0], reports)
    if result.status != "selected" or result.source_repdte != current.repdte:
        raise ValueError("Feature contract selected a different anchor")
    reasons = tuple(f"{name}:{result.diagnostics[name]['reason']}" for name in FEATURE_NAMES
                    if result.values[name] is None)
    lineage = json.dumps(result.diagnostics, sort_keys=True, separators=(",", ":"))
    return Anchor(current.cert, current.repdte, availability_date(current),
                  current.source_id, current.source_missing_fields,
                  {field: current.values[field] for field in FEATURE_INPUTS},
                  result.values, reasons, lineage)


def build_quarter_tables(q: date, candidates: list[dict], labels: list[dict],
                         anchors: dict[int, Anchor]) -> tuple[pa.Table, pa.Table, dict]:
    """Join keys one-to-one and keep all non-model dispositions in evidence."""
    if q > MAX_PREDICTION_Q:
        raise ValueError("Prediction quarter exceeds 2025Q2")
    label_by_cert: dict[int, dict] = {}
    for label in labels:
        cert = label["CERT"]
        if label["Q"] != q or cert in label_by_cert:
            raise ValueError(f"Duplicate or wrong-quarter label key: {cert}, {q}")
        membership = label["MEMBERSHIP"]
        disposition = label["LABEL_DISPOSITION"]
        expected_y = 1 if disposition == "positive" else (
            0 if disposition == "negative" else None)
        if (membership not in {"eligible", "ineligible", "uncertain"}
                or disposition not in {"positive", "negative", "unresolved", "excluded"}
                or (expected_y is not None and membership != "eligible")
                or label["Y"] != expected_y):
            raise ValueError(f"Contradictory label disposition or Y: {cert}, {q}")
        label_by_cert[cert] = label
    candidate_rows, model_rows = [], []
    seen = set()
    counts = Counter()
    candidate_nulls = Counter()
    model_nulls = Counter()
    cross = Counter()
    exclusions = Counter()
    for item in candidates:
        cert = item["CERT"]
        if item["Q"] != q or cert in seen:
            raise ValueError(f"Duplicate or wrong-quarter feature key: {cert}, {q}")
        seen.add(cert)
        label = label_by_cert.get(cert)
        if label is None:
            counts["feature_only"] += 1
            continue
        anchor = anchors.get(cert)
        if anchor is None or item["SELECTED_REPDTE"] != anchor.repdte:
            raise ValueError(f"PIT anchor differs from complete report history: {cert}, {q}")
        if (item["AVAILABILITY_DATE"] != anchor.available
                or anchor.available > q or item["REPORT_AGE_DAYS"] != (q - anchor.repdte).days
                or item["SOURCE_ID_DIAGNOSTIC"] != anchor.source_id
                or tuple(item["SOURCE_MISSING_FIELDS_DIAGNOSTIC"] or ()) != anchor.source_missing
                or any(item[field] != anchor.source_values[field] for field in FEATURE_INPUTS)):
            raise ValueError(f"PIT source values or availability differ: {cert}, {q}")
        if not label["FEATURE_HISTORY_ELIGIBLE"]:
            raise ValueError(f"PIT candidate lacks label-side feature-history flag: {cert}, {q}")
        values = anchor.features
        candidate_rows.append({"CERT": cert, "Q": q, **values,
            "SELECTED_REPDTE": anchor.repdte, "AVAILABILITY_DATE": anchor.available,
            "REPORT_AGE_DAYS": item["REPORT_AGE_DAYS"],
            "AVAILABILITY_BASIS": item["AVAILABILITY_BASIS"],
            "SOURCE_ID_DIAGNOSTIC": anchor.source_id,
            "SOURCE_MISSING_FIELDS_DIAGNOSTIC": list(anchor.source_missing),
            "FEATURE_MISSING_REASONS": list(anchor.missing_reasons),
            "FEATURE_LINEAGE_JSON": anchor.feature_lineage_json})
        counts["matched"] += 1
        cross[(label["MEMBERSHIP"], label["LABEL_DISPOSITION"])] += 1
        for name in FEATURE_NAMES:
            if values[name] is None:
                candidate_nulls[name] += 1
        if label["MEMBERSHIP"] == "eligible" and label["LABEL_DISPOSITION"] in {"positive", "negative"}:
            y = 1 if label["LABEL_DISPOSITION"] == "positive" else 0
            if label["Y"] != y:
                raise ValueError(f"Label value/disposition differs: {cert}, {q}")
            model_rows.append({"CERT": cert, "Q": q, "Y": y, **values})
            counts["model_positive" if y else "model_negative"] += 1
            for name in FEATURE_NAMES:
                if values[name] is None:
                    model_nulls[name] += 1
        elif label["MEMBERSHIP"] != "eligible":
            exclusions[f"membership_{label['MEMBERSHIP']}"] += 1
        else:
            exclusions[f"label_{label['LABEL_DISPOSITION']}"] += 1
    counts["candidate_keys"] = len(candidates)
    counts["label_keys"] = len(labels)
    label_only = set(label_by_cert) - seen
    if any(label_by_cert[cert]["FEATURE_HISTORY_ELIGIBLE"] for cert in label_only):
        raise ValueError(f"Label asserts feature-history without a PIT key: {q}")
    counts["label_only"] = len(label_only)
    exclusions["no_feature_history"] = counts["label_only"]
    counts["excluded_or_unresolved_with_history"] = counts["matched"] - len(model_rows)
    if counts["feature_only"] or counts["matched"] != len(candidate_rows) or (
            len(model_rows) != counts["model_positive"] + counts["model_negative"]):
        raise ValueError("Feature–label one-to-one reconciliation failed")
    candidates_table = pa.Table.from_pylist(candidate_rows, schema=CANDIDATE_SCHEMA)
    model_table = pa.Table.from_pylist(model_rows, schema=MODEL_SCHEMA)
    return candidates_table, model_table, {
        "quarter": q.strftime("%Y%m%d"), **dict(counts),
        "candidate_nulls": {name: candidate_nulls[name] for name in FEATURE_NAMES},
        "model_nulls": {name: model_nulls[name] for name in FEATURE_NAMES},
        "matched_membership_label": [
            {"membership": member, "label": outcome, "count": count}
            for (member, outcome), count in sorted(cross.items())],
        "label_only_membership_label": [
            {"membership": member, "label": outcome, "count": count}
            for (member, outcome), count in sorted(Counter(
                (label_by_cert[cert]["MEMBERSHIP"], label_by_cert[cert]["LABEL_DISPOSITION"])
                for cert in set(label_by_cert) - seen).items())],
        "exclusion_reasons": dict(sorted(exclusions.items())),
    }


def run(root: Path, run_id: str, *, panel_run: str = PANEL_RUN,
        pit_run: str = PIT_RUN, contract_run: str = CONTRACT_RUN,
        membership_run: str = MEMBERSHIP_RUN, label_run: str = LABEL_RUN,
        review_run: str = REVIEW_RUN) -> dict:
    root = root.resolve(strict=True)
    _run_id(run_id)
    evidence = root / "runs" / NAMESPACE / run_id
    data = root / "data_processed" / NAMESPACE / run_id
    if evidence.exists() or data.exists():
        raise FileExistsError("P4.3 run ID already exists")
    selected_runs = {"panel": panel_run, "pit": pit_run,
        "feature_contract": contract_run, "membership": membership_run,
        "labels": label_run, "review": review_run}
    panel, pit, labels, upstream_hashes = _verified_inputs(root,
        panel_run=panel_run, pit_run=pit_run, contract_run=contract_run,
        membership_run=membership_run, label_run=label_run, review_run=review_run)
    financial = sorted((item for item in panel["outputs"] if item["dataset"] == "financials"),
                       key=lambda item: item["quarter"])
    if len(financial) != 170:
        raise ValueError("Observed Financials quarter inventory differs")
    config = load_config(root / "config" / "financials_full_v2.json")
    manifest = RunManifest.new(run_id, config)
    manifest.config["feature_contract"] = CONTRACT_VERSION
    manifest.config["table_version"] = TABLE_VERSION
    manifest.config["preprocessing_factory_unfitted"] = PREPROCESSING_VERSION
    manifest.config["predictor_columns"] = list(PREDICTOR_COLUMNS)
    manifest.config["source_runs"] = selected_runs
    for path, digest in sorted(upstream_hashes.items()):
        manifest.record_input(path, digest)
    state = ReportHistoryState(financial)
    outputs = []
    by_quarter = []
    evidence.mkdir(parents=True)
    data.mkdir(parents=True)
    manifest.set_status("running")
    manifest.write(evidence / "run_manifest.json")
    try:
        for pit_item, label_item in zip(pit["outputs"], labels["outputs"]):
            if pit_item["quarter"] != label_item["quarter"]:
                raise ValueError("PIT and label quarter order differs")
            qtext = pit_item["quarter"]
            q = _day(qtext)
            if q > MAX_PREDICTION_Q:
                raise ValueError("Prediction quarter exceeds accepted modelling cutoff")
            def load_source_rows(source: dict) -> list[dict]:
                path = root / "data_processed" / "task_2_2_panel" / panel_run / source["relative_path"]
                if _sha(path) != source["sha256"]:
                    raise ValueError(f"Observed source checksum differs: {path}")
                manifest.record_input(str(path), source["sha256"], source["rows"])
                with pq.ParquetFile(path) as file:
                    validate_schema(file.schema_arrow)
                    return file.read(columns=list(SOURCE_COLUMNS)).to_pylist()

            state.advance(q, load_source_rows)
            for item in (pit_item, label_item):
                path = Path(item["path"])
                if _sha(path) != item["sha256"]:
                    raise ValueError(f"Accepted quarter checksum differs: {path}")
                manifest.record_input(str(path), item["sha256"], item["rows"])
            with pq.ParquetFile(pit_item["path"]) as file:
                candidate_rows = file.read(columns=list(PIT_COLUMNS)).to_pylist()
            with pq.ParquetFile(label_item["path"]) as file:
                label_rows = file.read().to_pylist()
            if len(candidate_rows) != pit_item["rows"] or len(label_rows) != label_item["rows"]:
                raise ValueError(f"Accepted quarter row count differs: {qtext}")
            candidate_table, model_table, stats = build_quarter_tables(
                q, candidate_rows, label_rows, state.anchors)
            for dataset, table in (("feature_candidates", candidate_table),
                                   ("model_table", model_table)):
                target = data / dataset / f"Q={qtext}" / "part-000.parquet"
                digest = _write_verified(target, table)
                manifest.record_output(path=str(target), dataset=dataset,
                    rows=table.num_rows, sha256=digest, compression="ZSTD",
                    schema_version=TABLE_VERSION, partition=qtext,
                    source_total=table.num_rows, source_query=None)
                outputs.append({"path": str(target), "sha256": digest,
                                "rows": table.num_rows, "dataset": dataset, "quarter": qtext})
            by_quarter.append(stats)
            if len(by_quarter) % 20 == 0:
                manifest.write(evidence / "run_manifest.json")
        overall = {key: sum(row.get(key, 0) for row in by_quarter)
                   for key in ("candidate_keys", "label_keys", "matched", "feature_only",
                               "label_only", "excluded_or_unresolved_with_history",
                               "model_positive", "model_negative")}
        candidate_nulls = {name: sum(row["candidate_nulls"][name] for row in by_quarter)
                           for name in FEATURE_NAMES}
        model_nulls = {name: sum(row["model_nulls"][name] for row in by_quarter)
                       for name in FEATURE_NAMES}
        exclusion_reasons = {key: sum(row["exclusion_reasons"].get(key, 0)
                                      for row in by_quarter)
                             for key in sorted(set().union(*(
                                 row["exclusion_reasons"] for row in by_quarter)))}
        if (len(by_quarter) != 166 or len(outputs) != 332
                or overall["candidate_keys"] != pit["overall"]["candidates"]
                or overall["label_keys"] != labels["overall"]["decisions"]
                or overall["model_positive"] != labels["overall"]["feature_history_and_positive"]
                or overall["model_negative"] != labels["overall"]["feature_history_and_negative"]
                or overall["feature_only"] != 0
                or overall["label_only"] != overall["label_keys"] - overall["candidate_keys"]):
            raise ValueError("Complete feature–label reconciliation differs")
        summary = {"status": "complete", "run_id": run_id,
                   "table_version": TABLE_VERSION, "contract_version": CONTRACT_VERSION,
                   "predictor_columns": list(PREDICTOR_COLUMNS),
                   "preprocessing_factory": PREPROCESSING_VERSION,
                   "preprocessing_fitted_on_real_data": False,
                   "source_runs": selected_runs,
                   "source_report_quarters_consumed": state.pointer,
                   "source_reports_consumed": state.scanned_reports,
                   "overall": overall, "candidate_nulls": candidate_nulls,
                   "model_nulls": model_nulls,
                   "exclusion_reasons": exclusion_reasons,
                   "by_quarter": by_quarter, "outputs": outputs,
                   "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "limitations": ["Financials publication timing uses the accepted 120-day assumption.",
                       "The accepted panel has one version per bank-quarter and no historical revision vintages.",
                       "The cohort is retrospective; its structural evidence is never a predictor."]}
        _write_json(evidence / "summary.json", summary)
        _write_json(evidence / "metadata.json", {"status": "complete",
            "summary_sha256": _sha(evidence / "summary.json"), "output_count": len(outputs)})
        manifest.record_validation("accepted_inputs_and_contract", True,
            "Accepted panel/PIT/contract/membership/label/review evidence verified")
        manifest.record_validation("one_to_one_join_and_modelling_eligibility", True,
            "All candidate keys matched labels; final rows have resolved active membership and binary outcomes")
        manifest.record_validation("atomic_readback_schema_key_codec_checksum", True,
            "332 quarterly Parquet outputs validated on atomic promotion")
        manifest.set_status("complete")
        manifest.write(evidence / "run_manifest.json")
        return {"status": "complete", "run_id": run_id, "overall": overall,
                "candidate_nulls": candidate_nulls, "model_nulls": model_nulls,
                "summary": str(evidence / "summary.json")}
    except Exception as exc:
        manifest.record_validation("feature_label_run", False, f"{type(exc).__name__}: {exc}")
        manifest.set_status("failed")
        manifest.write(evidence / "run_manifest.json")
        raise


def verify_saved(root: Path, run_id: str, *, check_upstream: bool = True) -> dict:
    root = root.resolve(strict=True)
    _run_id(run_id)
    evidence = root / "runs" / NAMESPACE / run_id
    manifest = RunManifest.read(evidence / "run_manifest.json")
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    if manifest.config.get("source_runs", summary.get("source_runs")) != summary.get("source_runs"):
        raise ValueError("Feature table manifest/source lineage differs")
    if (manifest.status != "complete" or not manifest.validations
            or any(not item["passed"] for item in manifest.validations)
            or summary.get("status") != "complete" or summary.get("run_id") != run_id
            or summary.get("table_version") != TABLE_VERSION
            or summary.get("predictor_columns") != list(PREDICTOR_COLUMNS)
            or metadata.get("status") != "complete"
            or metadata.get("summary_sha256") != _sha(evidence / "summary.json")
            or len(summary["outputs"]) != 332 or len(manifest.outputs) != 332
            or len(summary["by_quarter"]) != 166):
        raise ValueError("P4.3 manifest/summary/metadata incomplete")
    if check_upstream:
        for item in manifest.inputs:
            if ((root / "runs/source_vintage_bundle.json").is_file() and
                    not Path(item["path"]).resolve().is_relative_to(root)):
                raise ValueError("Portable feature table input escaped root")
            if item["sha256"] and _sha(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"P4.3 upstream checksum differs: {item['path']}")
    recorded = {item["path"]: item for item in manifest.outputs}
    seen: set[tuple[str, str]] = set()
    totals = Counter()
    keys_by_q: dict[str, dict[str, set[int]]] = defaultdict(dict)
    for item in summary["outputs"]:
        path = Path(item["path"])
        dataset, quarter = item["dataset"], item["quarter"]
        schema = CANDIDATE_SCHEMA if dataset == "feature_candidates" else MODEL_SCHEMA
        if (dataset not in {"feature_candidates", "model_table"}
                or not path.resolve().is_relative_to((root / "data_processed" / NAMESPACE / run_id).resolve())
                or recorded.get(str(path), {}).get("sha256") != item["sha256"]
                or _sha(path) != item["sha256"]
                or (dataset, quarter) in seen):
            raise ValueError(f"P4.3 output path/key/checksum differs: {path}")
        seen.add((dataset, quarter))
        with pq.ParquetFile(path) as file:
            if (file.schema_arrow != schema or file.metadata.num_rows != item["rows"]
                    or any(file.metadata.row_group(g).column(c).compression != "ZSTD"
                           for g in range(file.metadata.num_row_groups)
                           for c in range(file.metadata.row_group(g).num_columns))):
                raise ValueError(f"P4.3 schema/codec/row count differs: {path}")
            table = file.read(columns=["CERT", "Q"] + (["Y"] if dataset == "model_table" else
                ["SELECTED_REPDTE", "AVAILABILITY_DATE"]))
        rows = table.to_pylist()
        certs = {row["CERT"] for row in rows}
        if (len(certs) != len(rows) or any(row["Q"] != _day(quarter) for row in rows)
                or (dataset == "model_table" and any(row["Y"] not in {0, 1} for row in rows))
                or (dataset == "feature_candidates" and any(
                    row["AVAILABILITY_DATE"] > row["Q"] for row in rows))):
            raise ValueError(f"P4.3 keys, labels or availability differ: {path}")
        keys_by_q[quarter][dataset] = certs
        totals[dataset] += len(rows)
    if (len(seen) != 332 or len(keys_by_q) != 166
            or any(set(parts) != {"feature_candidates", "model_table"}
                   or not parts["model_table"].issubset(parts["feature_candidates"])
                   for parts in keys_by_q.values())
            or totals["feature_candidates"] != summary["overall"]["candidate_keys"]
            or totals["model_table"] != (summary["overall"]["model_positive"] +
                                         summary["overall"]["model_negative"])):
        raise ValueError("P4.3 complete output reconciliation differs")
    return {"status": "complete", "run_id": run_id,
            "quarters": len(keys_by_q), "outputs": len(seen),
            "overall": summary["overall"], "candidate_nulls": summary["candidate_nulls"],
            "model_nulls": summary["model_nulls"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Accepted P4.3 feature–label table")
    parser.add_argument("mode", choices=("run", "verify", "show"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--panel-run", default=PANEL_RUN)
    parser.add_argument("--pit-run", default=PIT_RUN)
    parser.add_argument("--contract-run", default=CONTRACT_RUN)
    parser.add_argument("--membership-run", default=MEMBERSHIP_RUN)
    parser.add_argument("--label-run", default=LABEL_RUN)
    parser.add_argument("--review-run", default=REVIEW_RUN)
    args = parser.parse_args()
    if args.mode == "run":
        result = run(args.project_root, args.run_id,
                     panel_run=args.panel_run, pit_run=args.pit_run,
                     contract_run=args.contract_run, membership_run=args.membership_run,
                     label_run=args.label_run, review_run=args.review_run)
    elif args.mode == "verify":
        result = verify_saved(args.project_root, args.run_id)
    else:
        path = args.project_root / "runs" / NAMESPACE / args.run_id
        manifest = RunManifest.read(path / "run_manifest.json")
        summary = _read_json(path / "summary.json")
        metadata = _read_json(path / "metadata.json")
        if (manifest.run_id != args.run_id or manifest.status != "complete"
                or not manifest.validations
                or any(not item["passed"] for item in manifest.validations)
                or summary.get("run_id") != args.run_id or summary.get("status") != "complete"
                or metadata.get("status") != "complete"
                or metadata.get("summary_sha256") != _sha(path / "summary.json")
                or metadata.get("output_count") != 332
                or len(manifest.outputs) != 332):
            raise ValueError(f"P4.3 saved evidence is incomplete: {args.run_id}")
        result = {"status": "complete", "run_id": args.run_id,
                  "overall": summary["overall"], "candidate_nulls": summary["candidate_nulls"],
                  "model_nulls": summary["model_nulls"],
                  "q2025q2": summary["by_quarter"][-1]}
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
