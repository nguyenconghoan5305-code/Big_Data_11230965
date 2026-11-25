"""Provisional point-in-time Financials selection for an explicit prediction roster.

This module does not build a full panel, determine exits, create labels, or fit models.
The 120-day delay is an assumption about availability, not a verified release date.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Mapping, Sequence

from bft65b.config import EXTENDED_FINANCIAL_FIELDS


AVAILABILITY_LAG_DAYS = 120
PREDICTOR_FIELDS = (
    "ASSET", "DEP", "EQ", "NETINC", "ROA", "NCLNLS", "LNATRES",
    "RBC1AAJ", "CHBAL", "SC", "BRO", "DEPDOM", "EINTEXP",
    "LNLSGR", "LNRE", "LNCI", "P3ASSET", "P9ASSET", "NAASSET",
    "NTLNLS", "ELNATR", "NIMY", "RBCRWAJ",
)
if set(PREDICTOR_FIELDS) != set(EXTENDED_FINANCIAL_FIELDS) - {"CERT", "REPDTE", "STNAME"}:
    raise RuntimeError("PIT predictor allowlist differs from verified Financials fields")

POPULATION_SOURCES = frozenset({"available_report_roster", "verified_registry"})


class AmbiguousRevisionError(ValueError):
    """A bank-quarter has versions that cannot be ordered by verified release date."""


def _quarter_end(value: date) -> bool:
    return type(value) is date and (value.month, value.day) in {
        (3, 31), (6, 30), (9, 30), (12, 31)}


@dataclass(frozen=True)
class PredictionRequest:
    """Population membership established separately from the selected feature report."""

    cert: int
    q: date
    population_source: str
    population_known_at: date
    population_evidence_id: str

    def __post_init__(self) -> None:
        if type(self.cert) is not int or self.cert <= 0 or not _quarter_end(self.q):
            raise ValueError("Prediction requires a positive CERT and a quarter-end q")
        if (self.population_source not in POPULATION_SOURCES
                or type(self.population_known_at) is not date
                or self.population_known_at > self.q
                or not self.population_evidence_id):
            raise ValueError("Population membership must be evidenced no later than q")


@dataclass(frozen=True)
class FinancialReport:
    """One physical Financials version; values may include audit-only columns."""

    cert: int
    repdte: date
    values: Mapping[str, Any]
    source_id: str | None = None
    source_missing_fields: tuple[str, ...] = ()
    revision_published_at: date | None = None
    revision_time_verified: bool = False

    def __post_init__(self) -> None:
        if type(self.cert) is not int or self.cert <= 0 or not _quarter_end(self.repdte):
            raise ValueError("Financials report requires a positive CERT and quarter-end REPDTE")
        missing = set(PREDICTOR_FIELDS) - set(self.values)
        if missing:
            raise ValueError(f"Financials report lacks predictor columns: {sorted(missing)}")
        if type(self.revision_time_verified) is not bool or (
                (self.revision_published_at is None) != (not self.revision_time_verified)):
            raise ValueError("A revision date must be present and explicitly verified together")
        if self.revision_published_at is not None and (
                type(self.revision_published_at) is not date
                or self.revision_published_at < self.repdte):
            raise ValueError("Verified revision date must be a date on or after REPDTE")
        if not isinstance(self.source_missing_fields, tuple) or any(
                field not in EXTENDED_FINANCIAL_FIELDS for field in self.source_missing_fields):
            raise ValueError("Source-absence provenance contains unverified field names")

    @classmethod
    def from_panel_row(cls, row: Mapping[str, Any]) -> "FinancialReport":
        """Read an accepted P2.2 row without importing DQ or exit fields as predictors."""
        return cls(cert=row["CERT"], repdte=row["REPDTE"], values=row,
                   source_id=row.get("ID"),
                   source_missing_fields=tuple(row.get("SOURCE_MISSING_FIELDS") or ()))


def availability_date(report: FinancialReport) -> date:
    """Apply the provisional 120-calendar-day delay, then a verified later revision date."""
    provisional = report.repdte + timedelta(days=AVAILABILITY_LAG_DAYS)
    return max(provisional, report.revision_published_at or provisional)


@dataclass(frozen=True)
class AsOfSelection:
    cert: int
    q: date
    population_source: str
    population_known_at: date
    status: str
    source_repdte: date | None
    source_availability_date: date | None
    report_age_days: int | None
    availability_basis: str | None
    predictors: dict[str, Any] | None
    diagnostics: dict[str, Any]


def _eligible_versions(reports: Sequence[FinancialReport], q: date) -> list[FinancialReport]:
    """Reject ambiguity only among versions provisionally available by q."""
    by_quarter: dict[date, list[FinancialReport]] = defaultdict(list)
    for report in reports:
        if availability_date(report) <= q:
            by_quarter[report.repdte].append(report)
    eligible = []
    for repdte, versions in by_quarter.items():
        if len(versions) > 1:
            times = [item.revision_published_at for item in versions]
            if any(value is None for value in times) or len(set(times)) != len(times):
                raise AmbiguousRevisionError(
                    f"Bank-quarter {versions[0].cert}:{repdte} has undated or tied versions")
        eligible.extend(versions)
    return eligible


def population_from_available_reports(q: date,
                                      reports: Sequence[FinancialReport]) -> list[PredictionRequest]:
    """Build a conservative as-of roster from reports presumed available by q.

    This is a candidate roster, not proof that a bank remained open at q.
    A future q report cannot create membership retroactively.
    """
    if not _quarter_end(q):
        raise ValueError("Prediction date q must be a quarter-end")
    by_bank: dict[int, list[FinancialReport]] = defaultdict(list)
    for report in reports:
        by_bank[report.cert].append(report)
    population = []
    for cert in sorted(by_bank):
        eligible = _eligible_versions(by_bank[cert], q)
        if eligible:
            first = min(eligible, key=lambda item: (availability_date(item), item.repdte))
            population.append(PredictionRequest(
                cert, q, "available_report_roster", availability_date(first),
                f"Financials:{cert}:{first.repdte.isoformat()}"))
    return population


def select_asof(population: Sequence[PredictionRequest],
                reports: Sequence[FinancialReport]) -> list[AsOfSelection]:
    """Choose the latest eligible REPDTE per explicit (CERT, q) population row.

    The function never reads Failures, exit assessments, Institutions, or DQ_FLAGS.
    It preserves input population order and does not manufacture report values.
    """
    by_bank: dict[int, list[FinancialReport]] = defaultdict(list)
    for report in reports:
        by_bank[report.cert].append(report)
    seen: set[tuple[int, date]] = set()
    selections = []
    for request in population:
        key = request.cert, request.q
        if key in seen:
            raise ValueError(f"Duplicate prediction population key: {key}")
        seen.add(key)
        eligible = _eligible_versions(by_bank.get(request.cert, ()), request.q)
        if request.population_source == "available_report_roster":
            if not eligible or request.population_known_at != min(
                    availability_date(item) for item in eligible):
                raise ValueError(f"Available-report population evidence does not reconcile: {key}")
        if not eligible:
            selections.append(AsOfSelection(request.cert, request.q,
                request.population_source, request.population_known_at,
                "no_eligible_history", None, None, None, None, None,
                {"population_evidence_id": request.population_evidence_id}))
            continue
        chosen = max(eligible, key=lambda item: (
            item.repdte, item.revision_published_at or date.min))
        basis = ("provisional_120_day_plus_verified_revision"
                 if chosen.revision_time_verified else "provisional_120_day")
        selections.append(AsOfSelection(request.cert, request.q,
            request.population_source, request.population_known_at,
            "selected", chosen.repdte, availability_date(chosen),
            (request.q - chosen.repdte).days, basis,
            {field: chosen.values[field] for field in PREDICTOR_FIELDS},
            {"population_evidence_id": request.population_evidence_id,
             "source_id": chosen.source_id,
             "source_missing_fields": chosen.source_missing_fields,
             "revision_published_at": chosen.revision_published_at}))
    return selections
