"""Bounded, point-in-time feature definitions; no labels or fitted statistics."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from bft65b.config import EXTENDED_FINANCIAL_FIELDS, load_config
from bft65b.manifest import RunManifest
from bft65b.panel.asof import (
    AVAILABILITY_LAG_DAYS, FinancialReport, PredictionRequest,
    _eligible_versions, availability_date, select_asof,
)
from bft65b.panel.repair import _accepted_panel
from bft65b.panel.prediction import verify_saved as verify_prediction
from bft65b.quality.audit import _read_json, _sha, _write_json


CONTRACT_VERSION = "risk_features_v1"
OFFICIAL_DICTIONARY_URL = "https://api.fdic.gov/banks/docs/risview_properties.yaml"
OFFICIAL_DICTIONARY_SHA256 = "b257457e4e821d6a0d9f88ffc4e6e0da80e4ccf67fa394cadb21587b50b23d59"
OFFICIAL_DICTIONARY_RETRIEVED = "2026-11-16"
PANEL_RUN = "panel20261115b"
PIT_RUN = "pit20261115f"
MAX_PREDICTION_Q = date(2025, 6, 30)
AMOUNT = pa.decimal128(20, 4)
DERIVED_PERCENT = pa.decimal128(30, 6)
RATIO_SCALE = Decimal("0.000001")
FEATURE_INPUTS = frozenset({"ASSET", "DEP", "EQ", "ROA", "RBC1AAJ"})
FORBIDDEN_MODEL_INPUTS = frozenset({
    "CERT", "REPDTE", "ID", "SOURCE_MISSING_FIELDS", "DQ_FLAGS",
    "FAILDATE", "COST", "ACTIVE", "CHANGEC1", "RESTYPE", "RESTYPE1",
    "TRANSACTION_DISPOSITION", "EXIT_CLASSIFICATION",
})


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source_fields: tuple[str, ...]
    output_type: str
    output_unit: str
    formula: str
    rationale: str
    lookback_calendar_quarters: int
    minimum_consecutive_reports: int
    missing_policy: str
    availability_rule: str
    temporal_anchor: str = "r = latest eligible Financials REPDTE selected for bank at prediction quarter-end q"
    earliest_prediction_q_rule: str = "first calendar quarter-end on or after the latest required input availability date"


FEATURES = (
    FeatureSpec("asset_size", ("ASSET",), "decimal128(20,4)", "thousands of USD (assignment convention)",
        "ASSET[r]", "Institution size and potential loss exposure", 0, 1,
        "Null or source absent stays null; nonpositive ASSET is invalid for size.",
        "availability(r) <= q"),
    FeatureSpec("equity_to_assets_pct", ("EQ", "ASSET"), "decimal128(30,6)", "percent",
        "100 * EQ[r] / ASSET[r]", "Equity buffer relative to assets", 0, 1,
        "Null/absent numerator or denominator stays null; ASSET <= 0 yields null; negative EQ is retained.",
        "availability(r) <= q"),
    FeatureSpec("deposits_to_assets_pct", ("DEP", "ASSET"), "decimal128(30,6)", "percent",
        "100 * DEP[r] / ASSET[r]", "Reliance on deposit funding", 0, 1,
        "Null/absent numerator or denominator stays null; ASSET <= 0 yields null; negative DEP is retained and flagged for review.",
        "availability(r) <= q"),
    FeatureSpec("roa_reported_pct", ("ROA",), "float64", "percent",
        "ROA[r]", "Reported annualized profitability relative to average assets", 0, 1,
        "Null/absent stays null; nonfinite source value is rejected; no rescaling.",
        "availability(r) <= q"),
    FeatureSpec("pca_leverage_reported_pct", ("RBC1AAJ",), "float64", "percent",
        "RBC1AAJ[r]", "Reported PCA leverage capital measure", 0, 1,
        "Null/absent stays null; nonfinite source value is rejected; no rescaling.",
        "availability(r) <= q"),
    FeatureSpec("asset_growth_qoq_pct", ("ASSET",), "decimal128(30,6)", "percent",
        "100 * (ASSET[r] / ASSET[r-1 calendar quarter] - 1)", "Quarterly balance-sheet expansion or contraction", 1, 2,
        "Exact adjacent quarter required; gap, null/absent input or prior ASSET <= 0 yields null; no forward fill.",
        "max(availability(r), availability(r-1)) <= q"),
    FeatureSpec("roa_four_quarter_mean_pct", ("ROA",), "float64", "percent",
        "mean(ROA[r], ROA[r-1], ROA[r-2], ROA[r-3])", "Persistent reported profitability across four observations", 3, 4,
        "All four consecutive calendar quarters and finite non-null ROA values required; any gap/null/absence yields null.",
        "max(availability(r), ..., availability(r-3)) <= q"),
)
FEATURE_NAMES = tuple(item.name for item in FEATURES)


@dataclass(frozen=True)
class FeatureResult:
    cert: int
    q: date
    status: str
    source_repdte: date | None
    values: dict[str, Decimal | float | None]
    diagnostics: dict[str, dict[str, Any]]


def _previous_quarter(day: date, steps: int) -> date:
    index = day.year * 4 + day.month // 3 - 1 - steps
    year, quarter = divmod(index, 4)
    return (date(year, 3, 31), date(year, 6, 30), date(year, 9, 30),
            date(year, 12, 31))[quarter]


def _quarter_end_on_or_after(day: date) -> date:
    for year in (day.year, day.year + 1):
        for month, last in ((3, 31), (6, 30), (9, 30), (12, 31)):
            result = date(year, month, last)
            if result >= day:
                return result
    raise ValueError("Cannot determine prediction quarter-end")


def _source_value(report: FinancialReport, field: str, expected: type) -> tuple[Any, str | None]:
    if field in report.source_missing_fields:
        return None, "source_absent"
    value = report.values[field]
    if value is None:
        return None, "explicit_null"
    if expected is Decimal and type(value) is not Decimal:
        raise TypeError(f"{field} must be an exact Decimal from the accepted panel")
    if expected is float and (type(value) is not float or not math.isfinite(value)):
        raise TypeError(f"{field} must be finite float64 from the accepted panel")
    return value, None


def _percent(numerator: Decimal, denominator: Decimal, *, growth: bool = False) -> Decimal:
    with localcontext() as context:
        context.prec = 38
        result = ((numerator / denominator - 1) if growth else
                  numerator / denominator) * 100
        rounded = result.quantize(RATIO_SCALE, rounding=ROUND_HALF_EVEN)
        pa.scalar(rounded, type=DERIVED_PERCENT)
        return rounded


def compute_features(request: PredictionRequest,
                     reports: Sequence[FinancialReport]) -> FeatureResult:
    """Use the accepted as-of selector; report history is scoped to this bank.

    Population membership is supplied independently. Source IDs and quality/exit
    audit fields are diagnostic only and cannot enter ``values``.
    """
    if request.q > MAX_PREDICTION_Q:
        raise ValueError("Modelling prediction q must be no later than 2025Q2")
    selection = select_asof([request], reports)[0]
    values: dict[str, Decimal | float | None] = {name: None for name in FEATURE_NAMES}
    diagnostics: dict[str, dict[str, Any]] = {}
    if selection.status != "selected":
        for name in FEATURE_NAMES:
            diagnostics[name] = {"reason": "no_eligible_history", "earliest_input_availability": None}
        return FeatureResult(request.cert, request.q, selection.status, None, values, diagnostics)

    assert selection.source_repdte is not None
    eligible = _eligible_versions([report for report in reports if report.cert == request.cert], request.q)
    by_date: dict[date, FinancialReport] = {}
    for report in eligible:
        previous = by_date.get(report.repdte)
        if previous is None or (report.revision_published_at or date.min) > (
                previous.revision_published_at or date.min):
            by_date[report.repdte] = report
    anchor = by_date[selection.source_repdte]

    def finish(name: str, required: Sequence[date], field: str | None = None) -> list[FinancialReport] | None:
        found = [by_date.get(day) for day in required]
        if any(item is None for item in found):
            diagnostics[name] = {"reason": "missing_or_unavailable_calendar_quarter",
                                 "required_repdte": [day.isoformat() for day in required],
                                 "earliest_input_availability": None}
            return None
        present = [item for item in found if item is not None]
        first_available = max(availability_date(item) for item in present)
        diagnostics[name] = {
            "reason": "available", "required_repdte": [day.isoformat() for day in required],
            "earliest_input_availability": first_available.isoformat(),
            "earliest_prediction_q": _quarter_end_on_or_after(first_available).isoformat(),
            "availability_basis": "provisional_120_day_with_verified_later_revision_where_known",
        }
        return present

    for name, field in (("asset_size", "ASSET"), ("roa_reported_pct", "ROA"),
                        ("pca_leverage_reported_pct", "RBC1AAJ")):
        finish(name, [anchor.repdte])
        value, reason = _source_value(anchor, field, Decimal if field == "ASSET" else float)
        if reason is None and field == "ASSET" and value <= 0:
            reason = "nonpositive_asset"
        values[name] = value if reason is None else None
        if reason:
            diagnostics[name]["reason"] = reason

    for name, numerator_field in (("equity_to_assets_pct", "EQ"),
                                  ("deposits_to_assets_pct", "DEP")):
        finish(name, [anchor.repdte])
        numerator, numerator_reason = _source_value(anchor, numerator_field, Decimal)
        denominator, denominator_reason = _source_value(anchor, "ASSET", Decimal)
        reason = numerator_reason or denominator_reason
        if reason is None and denominator <= 0:
            reason = "nonpositive_denominator"
        if reason is None:
            values[name] = _percent(numerator, denominator)
        else:
            diagnostics[name]["reason"] = reason

    prior_date = _previous_quarter(anchor.repdte, 1)
    pair = finish("asset_growth_qoq_pct", [anchor.repdte, prior_date])
    if pair is not None:
        current, current_reason = _source_value(pair[0], "ASSET", Decimal)
        prior, prior_reason = _source_value(pair[1], "ASSET", Decimal)
        reason = current_reason or prior_reason
        if reason is None and prior <= 0:
            reason = "nonpositive_prior_denominator"
        if reason is None:
            values["asset_growth_qoq_pct"] = _percent(current, prior, growth=True)
        else:
            diagnostics["asset_growth_qoq_pct"]["reason"] = reason

    quarter_dates = [_previous_quarter(anchor.repdte, step) for step in range(4)]
    window = finish("roa_four_quarter_mean_pct", quarter_dates)
    if window is not None:
        points = [_source_value(item, "ROA", float) for item in window]
        reason = next((reason for _, reason in points if reason is not None), None)
        if reason is None:
            values["roa_four_quarter_mean_pct"] = math.fsum(value for value, _ in points) / 4
        else:
            diagnostics["roa_four_quarter_mean_pct"]["reason"] = reason
    return FeatureResult(request.cert, request.q, selection.status,
                         selection.source_repdte, values, diagnostics)


def validate_schema(schema: pa.Schema) -> None:
    """Check the accepted panel physical schema and bounded source allowlist."""
    expected = {"CERT": pa.int64(), "REPDTE": pa.date32(), "ASSET": AMOUNT,
                "DEP": AMOUNT, "EQ": AMOUNT, "ROA": pa.float64(),
                "RBC1AAJ": pa.float64(), "SOURCE_MISSING_FIELDS": pa.list_(pa.string())}
    if not FEATURE_INPUTS.issubset(EXTENDED_FINANCIAL_FIELDS):
        raise ValueError("Feature source field is outside verified Financials v2 allowlist")
    if FEATURE_INPUTS & FORBIDDEN_MODEL_INPUTS:
        raise ValueError("Retrospective or identifier field entered the feature inputs")
    for field, data_type in expected.items():
        if schema.get_field_index(field) < 0 or schema.field(field).type != data_type:
            raise ValueError(f"Accepted panel schema differs for {field}")
    if schema.metadata is None or schema.metadata.get(b"bft65b.schema_version") != b"2":
        raise ValueError("Accepted panel is not Financials schema v2")


def validate_accepted(root: Path, run_id: str,
                      panel_run: str = PANEL_RUN, pit_run: str = PIT_RUN) -> dict[str, Any]:
    """Read and verify accepted evidence; write only a small ignored contract run."""
    root = root.resolve(strict=True)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}", run_id):
        raise ValueError("Unsafe feature-contract run ID")
    evidence = root / "runs" / "task_4_1_feature_contract" / run_id
    if evidence.exists():
        raise FileExistsError(f"Feature-contract evidence already exists: {evidence}")
    portable = (root / "runs/source_vintage_bundle.json").is_file()
    if portable and (panel_run == PANEL_RUN or pit_run == PIT_RUN):
        raise ValueError("Portable feature contract requires explicit panel and PIT runs")
    panel_summary, _, _, _, _, panel_hashes = _accepted_panel(root, panel_run)
    pit_check = verify_prediction(root, pit_run, check_upstream=True)
    if portable:
        pit_manifest = RunManifest.read(root / "runs/task_2_5_prediction" / pit_run /
                                        "run_manifest.json")
        if pit_manifest.config.get("source_panel_run") != panel_run:
            raise ValueError("Feature contract panel/PIT lineage differs")
    if pit_check["status"] != "verified":
        raise ValueError("Accepted PIT run did not verify")
    outputs = [row for row in panel_summary["outputs"] if row["dataset"] == "financials"]
    if len(outputs) != 170:
        raise ValueError("Accepted panel does not have exactly 170 Financials partitions")
    for item in outputs:
        path = root / "data_processed" / "task_2_2_panel" / panel_run / item["relative_path"]
        with pq.ParquetFile(path) as parquet:
            validate_schema(parquet.schema_arrow)
            if parquet.schema_arrow.metadata.get(b"bft65b.partition") != (
                    "REPDTE=" + item["quarter"]).encode():
                raise ValueError(f"Panel partition metadata differs: {item['quarter']}")
    pit_evidence = root / "runs" / "task_2_5_prediction" / pit_run
    pit_hashes = {str(pit_evidence / name): _sha(pit_evidence / name)
                  for name in ("run_manifest.json", "summary.json", "metadata.json")}
    if sum(item["rows"] for item in outputs) != 1682616:
        raise ValueError("Accepted panel row total differs")
    config = load_config(root / "config" / "financials_full_v2.json")
    manifest = RunManifest.new(run_id, config)
    manifest.config.update({"source_panel_run": panel_run, "source_pit_run": pit_run})
    for path, digest in sorted({**panel_hashes, **pit_hashes}.items()):
        manifest.record_input(path, digest)
    manifest.record_validation("accepted_panel_and_pit_hashes", True,
                               f"{len(panel_hashes)} panel and {len(pit_hashes)} PIT evidence files")
    manifest.record_validation("financials_v2_schemas", True, "170 partitions; 26 source fields; 7 bounded features")
    contract = {
        "status": "complete", "contract_version": CONTRACT_VERSION, "run_id": run_id,
        "panel_run": panel_run, "pit_run": pit_run,
        "availability_lag_calendar_days": AVAILABILITY_LAG_DAYS,
        "availability_basis": "provisional; not historical vintage proof",
        "field_dictionary": {"url": OFFICIAL_DICTIONARY_URL,
            "sha256": OFFICIAL_DICTIONARY_SHA256,
            "retrieved_date": OFFICIAL_DICTIONARY_RETRIEVED},
        "maximum_prediction_q": MAX_PREDICTION_Q.isoformat(),
        "financials_partitions_checked": len(outputs),
        "financials_rows": sum(item["rows"] for item in outputs),
        "source_field_count": len(EXTENDED_FINANCIAL_FIELDS),
        "feature_input_fields": sorted(FEATURE_INPUTS),
        "features": [asdict(spec) for spec in FEATURES],
        "panel_input_hashes": dict(sorted(panel_hashes.items())),
        "pit_evidence_hashes": pit_hashes,
        "pit_validation": pit_check,
    }
    evidence.mkdir(parents=True)
    _write_json(evidence / "contract.json", contract)
    manifest.record_output(path=str(evidence / "contract.json"), dataset="feature_contract",
        rows=len(FEATURES), sha256=_sha(evidence / "contract.json"), compression="none",
        schema_version=CONTRACT_VERSION, partition=None, source_total=len(FEATURES), source_query=None)
    manifest.set_status("complete")
    manifest.write(evidence / "run_manifest.json")
    return {"status": "complete", "run_id": run_id, "contract": str(evidence / "contract.json"),
            "feature_count": len(FEATURES), "panel_quarters": len(outputs),
            "panel_rows": contract["financials_rows"], "pit": pit_check}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify bounded Part 4 point-in-time feature contract")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--project-root", type=Path, default=Path.cwd())
    validate.add_argument("--run-id", required=True)
    validate.add_argument("--panel-run", default=PANEL_RUN)
    validate.add_argument("--pit-run", default=PIT_RUN)
    show = sub.add_parser("show")
    show.add_argument("--project-root", type=Path, default=Path.cwd())
    show.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        result = validate_accepted(args.project_root, args.run_id,
                                   args.panel_run, args.pit_run)
    else:
        path = args.project_root / "runs" / "task_4_1_feature_contract" / args.run_id
        manifest = RunManifest.read(path / "run_manifest.json")
        result = _read_json(path / "contract.json")
        if (manifest.status != "complete" or result.get("status") != "complete"
                or result.get("run_id") != args.run_id
                or len(manifest.outputs) != 1
                or manifest.outputs[0]["sha256"] != _sha(path / "contract.json")):
            parser.exit(2, f"Feature contract {args.run_id}: incomplete or inconsistent evidence\n")
        result = {"status": result["status"], "run_id": args.run_id,
                  "contract": str(path / "contract.json"),
                  "feature_count": len(result["features"]),
                  "panel_quarters": result["financials_partitions_checked"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
