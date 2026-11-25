"""Pure, non-mutating raw-data checks for one FDIC endpoint period.

Counts describe observed source records. This module never cleans, imputes,
joins rows into a panel, or assigns a failure label.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import math
from typing import Any, Mapping, Sequence

import pyarrow as pa

from bft65b.acquisition.storage import arrow_schema
from bft65b.config import EXTENDED_FINANCIAL_FIELDS


RULES: dict[str, tuple[str, str]] = {
    "missing_primary_key": ("violation", "The endpoint-specific source key is null or empty."),
    "duplicate_primary_key": ("violation", "Additional row repeats an endpoint-specific primary key."),
    "invalid_report_date": ("violation", "REPDTE is not a valid calendar quarter-end."),
    "partition_date_mismatch": ("violation", "REPDTE differs from the Financials partition date."),
    "invalid_failure_date": ("violation", "FAILDATE is missing or not a date."),
    "invalid_provenance": ("violation", "SOURCE_MISSING_FIELDS is malformed or names a field outside the request."),
    "absent_field_has_value": ("violation", "A field marked absent in the response has a non-null stored value."),
    "invalid_value_type": ("violation", "A stored non-null value differs from its verified Arrow logical type."),
    "nonfinite_numeric": ("violation", "A numeric value is NaN or infinite."),
    "negative_total_assets": ("violation", "Total assets are negative; the official total-assets concept is nonnegative."),
    "domestic_deposits_above_total": ("review", "Domestic deposits exceed total deposits; confirm reporting basis and comparability before declaring an accounting error."),
    "active_outside_binary": ("violation", "ACTIVE is outside the dictionary's 0/1 status codes."),
    "financial_cert_absent_from_institutions": ("review", "Financials CERT is absent from the current Institutions endpoint snapshot; historical coverage is not implied."),
    "failure_cert_null": ("review", "Failure source record has no CERT bank-linkage identifier; source ID remains the record key."),
    "failure_cert_absent_from_financials": ("review", "Non-null failure CERT has no Financials CERT in the accepted 1984Q1-2026Q2 scope."),
    "failure_only_future_reports": ("review", "Failure CERT exists in Financials, but every report date is later than FAILDATE."),
    "failure_multiple_prior_reports": ("review", "Several earlier reports share the failure CERT; an unrestricted join would multiply the failure event."),
    "failure_ambiguous_latest_report": ("violation", "More than one report has the latest REPDTE on or before FAILDATE for this CERT."),
}


def _expected_fields(dataset: str) -> dict[str, pa.Field]:
    schema = arrow_schema(dataset, "20230331" if dataset == "financials" else None,
                          financial_fields=EXTENDED_FINANCIAL_FIELDS)
    return {field.name: field for field in schema}


def _type_ok(value: Any, arrow_type: pa.DataType) -> bool:
    if pa.types.is_decimal(arrow_type):
        return isinstance(value, Decimal)
    if pa.types.is_floating(arrow_type):
        return type(value) is float
    if pa.types.is_integer(arrow_type):
        return type(value) is int
    if pa.types.is_date(arrow_type):
        return isinstance(value, date) and not isinstance(value, datetime)
    if pa.types.is_string(arrow_type):
        return isinstance(value, str)
    return True


def _quarter_end(day: Any) -> bool:
    return isinstance(day, date) and not isinstance(day, datetime) and (
        day.month, day.day) in {(3, 31), (6, 30), (9, 30), (12, 31)}


def _finding(dataset: str, period: str, field: str, rule: str,
             count: int, denominator: int) -> dict[str, Any]:
    severity, interpretation = RULES[rule]
    return {"dataset": dataset, "period": period, "field": field, "rule": rule,
            "count": count, "denominator": denominator,
            "prevalence": count / denominator if denominator else None,
            "severity": severity, "interpretation": interpretation}


@dataclass
class PeriodAudit:
    summary: dict[str, Any]
    cert_dates: dict[int, list[date]]
    certs: set[int]
    unmatched_certs: set[int]
    linkage_certs: dict[str, set[int]]
    linkage_source_ids: dict[str, set[str]]
    row_flags: list[tuple[str, ...]]


def inspect_period(
    dataset: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str],
    period: str, *, partition: str | None = None,
    institution_certs: set[int] | None = None,
    financial_reports: Mapping[int, Sequence[date]] | None = None,
) -> PeriodAudit:
    """Count defects and source missingness; each affected row is counted once.

    A failure's latest report is only a chronological candidate. Publication
    availability and bank identity are not established by this diagnostic.
    """
    if dataset not in {"financials", "failures", "institutions"}:
        raise ValueError("Unknown FDIC endpoint")
    if dataset == "financials" and partition is None:
        raise ValueError("Financials require a quarter partition")
    expected = _expected_fields(dataset)
    selected = tuple(fields)
    if len(selected) != len(set(selected)) or not set(selected).issubset(expected):
        raise ValueError("Fields must be unique and belong to the verified endpoint schema")
    n = len(rows)
    absent = Counter({field: 0 for field in selected})
    explicit_null = Counter({field: 0 for field in selected})
    rules: Counter[tuple[str, str]] = Counter()
    affected = bytearray(n)
    missing_affected = bytearray(n)
    row_flags: list[set[str]] = [set() for _ in rows]
    seen: set[Any] = set()
    certs: set[int] = set()
    unmatched_certs: set[int] = set()
    cert_dates: dict[int, list[date]] = defaultdict(list)
    linkage_counts: Counter[str] = Counter()
    linkage_certs: dict[str, set[int]] = defaultdict(set)
    linkage_source_ids: dict[str, set[str]] = defaultdict(set)
    matched_institutions = 0

    def mark(index: int, rule: str, field: str) -> None:
        rules[(rule, field)] += 1
        affected[index] = 1
        row_flags[index].add(f"{rule}:{field}")

    for i, row in enumerate(rows):
        missing_value = row.get("SOURCE_MISSING_FIELDS")
        if (not isinstance(missing_value, list)
                or any(not isinstance(name, str) for name in missing_value)
                or len(missing_value) != len(set(missing_value))
                or not set(missing_value).issubset(selected)):
            mark(i, "invalid_provenance", "SOURCE_MISSING_FIELDS")
            source_absent: set[str] = set()
        else:
            source_absent = set(missing_value)
        for field in selected:
            value = row.get(field)
            if field in source_absent:
                absent[field] += 1
                missing_affected[i] = affected[i] = 1
                row_flags[i].add(f"source_field_absent:{field}")
                if value is not None:
                    mark(i, "absent_field_has_value", field)
            elif value is None:
                explicit_null[field] += 1
                missing_affected[i] = affected[i] = 1
                row_flags[i].add(f"source_explicit_null:{field}")
            elif not _type_ok(value, expected[field].type):
                mark(i, "invalid_value_type", field)
            elif isinstance(value, float) and not math.isfinite(value):
                mark(i, "nonfinite_numeric", field)

        if dataset == "financials":
            cert, day = row.get("CERT"), row.get("REPDTE")
            key = (cert, day)
            if cert is None or day is None:
                mark(i, "missing_primary_key", "CERT,REPDTE")
            elif key in seen:
                mark(i, "duplicate_primary_key", "CERT,REPDTE")
            else:
                seen.add(key)
            if not _quarter_end(day):
                mark(i, "invalid_report_date", "REPDTE")
            if not isinstance(day, date) or day.strftime("%Y%m%d") != partition:
                mark(i, "partition_date_mismatch", "REPDTE")
            if type(cert) is int:
                certs.add(cert)
                if isinstance(day, date):
                    cert_dates[cert].append(day)
                if institution_certs is not None:
                    if cert in institution_certs:
                        matched_institutions += 1
                    else:
                        unmatched_certs.add(cert)
                        mark(i, "financial_cert_absent_from_institutions", "CERT")
            asset = row.get("ASSET")
            if isinstance(asset, Decimal) and asset < 0:
                mark(i, "negative_total_assets", "ASSET")
            domestic, deposits = row.get("DEPDOM"), row.get("DEP")
            if isinstance(domestic, Decimal) and isinstance(deposits, Decimal) and domestic > deposits:
                mark(i, "domestic_deposits_above_total", "DEPDOM,DEP")
        elif dataset == "failures":
            source_id, cert, day = row.get("ID"), row.get("CERT"), row.get("FAILDATE")
            if not isinstance(source_id, str) or not source_id.strip():
                mark(i, "missing_primary_key", "ID")
            elif source_id in seen:
                mark(i, "duplicate_primary_key", "ID")
            else:
                seen.add(source_id)
            if not isinstance(day, date):
                mark(i, "invalid_failure_date", "FAILDATE")
            if cert is None:
                mark(i, "failure_cert_null", "CERT")
                category = "missing_cert"
            elif type(cert) is int:
                certs.add(cert)
                reports = list(financial_reports.get(cert, ())) if financial_reports is not None else []
                if not reports:
                    mark(i, "failure_cert_absent_from_financials", "CERT")
                    category = "no_financial_cert"
                elif not isinstance(day, date):
                    category = "invalid_date"
                else:
                    prior = [report for report in reports if report <= day]
                    if not prior:
                        mark(i, "failure_only_future_reports", "CERT,FAILDATE")
                        category = "future_reports_only"
                    else:
                        category = "chronological_candidate"
                        if len(prior) > 1:
                            mark(i, "failure_multiple_prior_reports", "CERT,FAILDATE")
                        if prior.count(max(prior)) > 1:
                            mark(i, "failure_ambiguous_latest_report", "CERT,REPDTE")
                            category = "ambiguous_latest_report"
                linkage_certs[category].add(cert)
            else:
                category = "invalid_cert_type"
            linkage_counts[category] += 1
            if isinstance(source_id, str) and source_id.strip():
                linkage_source_ids[category].add(source_id)
        else:
            cert = row.get("CERT")
            if cert is None:
                mark(i, "missing_primary_key", "CERT")
            elif cert in seen:
                mark(i, "duplicate_primary_key", "CERT")
            else:
                seen.add(cert)
            if type(cert) is int:
                certs.add(cert)
            active = row.get("ACTIVE")
            if active is not None and type(active) is int and active not in (0, 1):
                mark(i, "active_outside_binary", "ACTIVE")

    field_counts = {field: {"absent": absent[field], "explicit_null": explicit_null[field],
                            "non_null": n - absent[field] - explicit_null[field], "denominator": n}
                    for field in selected}
    findings = [_finding(dataset, period, field, rule, count, n)
                for (rule, field), count in sorted(rules.items()) if count]
    summary = {"dataset": dataset, "period": period, "rows": n,
               "field_counts": field_counts, "rule_findings": findings,
               "affected_records_unique": sum(affected),
               "missing_or_null_records_unique": sum(missing_affected),
               "distinct_cert_count": len(certs)}
    if dataset == "financials" and institution_certs is not None:
        summary["institution_snapshot_join"] = {
            "matched_records": matched_institutions,
            "unmatched_records": rules[("financial_cert_absent_from_institutions", "CERT")],
            "matched_distinct_certs": len(certs - unmatched_certs),
            "unmatched_distinct_certs": len(unmatched_certs),
            "record_denominator": n, "distinct_cert_denominator": len(certs),
        }
    if dataset == "failures":
        summary["failure_linkage"] = {
            "record_categories": dict(sorted(linkage_counts.items())),
            "distinct_nonnull_cert_by_category": {name: len(ids) for name, ids in sorted(linkage_certs.items())},
            "distinct_source_id_by_category": {name: len(ids) for name, ids in sorted(linkage_source_ids.items())},
            "record_denominator": n, "distinct_nonnull_cert_denominator": len(certs),
            "time_rule": "REPDTE <= FAILDATE only; report publication and bank identity are not established",
        }
    return PeriodAudit(summary, dict(cert_dates), certs, unmatched_certs,
                       dict(linkage_certs), dict(linkage_source_ids),
                       [tuple(sorted(flags)) for flags in row_flags])
