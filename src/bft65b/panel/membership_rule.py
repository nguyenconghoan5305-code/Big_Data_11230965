"""Historical activity cohort and separate hypothetical q-known rule.

The production study uses retrospectively reconstructed *effective* activity
intervals. This does not assert that the entire register was public at q.
The earlier ``membership_at_q`` fixture remains a separate deployment
thought experiment and is never used to manufacture production eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

RETROSPECTIVE_POLICY_VERSION = "retrospective_operational_v1"

# FDIC CHANGECODE_DESC evidence: codeprobe20261119a, scoped FI samples.
# Partial purchase/assumption (225), participation (810-812), insurance
# changes (410), assistance (830), and identifier changes (150) are not exits.
ORDINARY_MERGER_CODES = frozenset({221, 222, 223, 224})
OTHER_EXIT_CODES = frozenset({240})
FAILURE_RESOLUTION_CODES = frozenset({211, 213, 215, 216, 217, 230, 235, 260})
TERMINAL_CODES = ORDINARY_MERGER_CODES | OTHER_EXIT_CODES | FAILURE_RESOLUTION_CODES


@dataclass(frozen=True)
class ActiveInterval:
    start: date
    end: date | None  # exclusive; an exit effective on q excludes q
    source: str
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.end is not None and self.end < self.start:
            raise ValueError("Activity interval cannot end before entry")


@dataclass(frozen=True)
class BankEvidence:
    cert: int
    disposition: str  # supported, unresolved_missing_identity, unresolved_dates
    intervals: tuple[ActiveInterval, ...]
    source: str
    insured_date: date | None
    insurance_date_status: str
    report_dates: tuple[date, ...]
    history_ids: tuple[str, ...]
    cert_change_dates: tuple[date, ...] = ()
    conflict_start: date | None = None
    history_end_conflict: tuple[date, date] | None = None
    reason: str = ""


@dataclass(frozen=True)
class MembershipDecision:
    cert: int
    q: date
    status: str  # eligible, ineligible, uncertain
    reason: str
    evidence_source: str
    start: date | None
    end: date | None
    basis: str = RETROSPECTIVE_POLICY_VERSION


def source_date(value: str | None) -> date | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("FDIC source date is not text")
    if "T" in value:
        return date.fromisoformat(value[:10])
    return datetime.strptime(value, "%m/%d/%Y").date()


def derive_bank_evidence(cert: int, institution: dict | None,
                         history: Sequence[dict], reports: Sequence[date]) -> BankEvidence:
    """Derive one auditable CERT disposition from current structure sources.

    ``INSDATE`` is insurance entry, not operational establishment. A 9999
    value is treated as unresolved/open-ended, never a literal historical
    activity date. History-only intervals require more than a CERT overlap.
    """
    if type(cert) is not int or cert <= 0 or not reports:
        raise ValueError("Bank evidence requires a CERT and observed reports")
    observed = tuple(sorted(reports))
    if len(observed) != len(set(observed)):
        raise ValueError("Duplicate observed bank-quarter")
    relevant = tuple(sorted((row for row in history if row.get("CERT") == cert),
                            key=lambda row: (row["EFFDATE"], row["ID"])))
    ids = tuple(row["ID"] for row in relevant)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate FI History source IDs for CERT")
    changes = tuple(sorted(source_date(row["EFFDATE"]) for row in relevant
                           if row["CHANGECODE"] == 150))
    starts = sorted({source_date(row["EFFDATE"]) for row in relevant
                     if row["CHANGECODE"] == 110})
    terminals = [(source_date(row["EFFDATE"]), row["CHANGECODE"], row["ID"])
                 for row in relevant if row["CHANGECODE"] in TERMINAL_CODES]
    terminal_days = sorted({item[0] for item in terminals})
    insurance_raw = institution.get("INSDATE") if institution else None
    insurance = source_date(insurance_raw)
    insurance_status = ("unavailable_9999" if insurance is not None and insurance.year == 9999
                        else "available" if insurance is not None else "missing")
    if insurance_status != "available":
        insurance = None

    if institution is None:
        safe_terminal = [item for item in terminals
                         if item[1] in ORDINARY_MERGER_CODES | OTHER_EXIT_CODES]
        unsafe_codes = FAILURE_RESOLUTION_CODES | {150}
        supported = (len(starts) == 1 and len(terminal_days) == 1
                     and len(safe_terminal) == 1
                     and not any(row["CHANGECODE"] in unsafe_codes for row in relevant)
                     and starts[0] <= observed[0] <= observed[-1] <= terminal_days[0]
                     and starts[0] < terminal_days[0])
        if supported:
            interval = ActiveInterval(starts[0], terminal_days[0],
                                      "fdic_history_110_and_terminal_fi",
                                      tuple(row["ID"] for row in relevant
                                            if row["CHANGECODE"] == 110 or
                                            row["CHANGECODE"] in ORDINARY_MERGER_CODES | OTHER_EXIT_CODES))
            return BankEvidence(cert, "supported", (interval,), "history_interval",
                                None, "missing_institution", observed, ids,
                                reason="unique_history_entry_and_exit_without_identity_conflict")
        return BankEvidence(cert, "unresolved_missing_identity", (), "history_cert_overlap_only"
                            if relevant else "no_institution_or_history", None,
                            "missing_institution", observed, ids, changes,
                            reason="institution_absent_and_history_interval_unproven")

    established = source_date(institution.get("ESTYMD"))
    end = source_date(institution.get("ENDEFYMD"))
    if end is not None and end.year == 9999:
        end = None
    if established is None or (end is not None and end < established):
        return BankEvidence(cert, "unresolved_dates", (), "institutions", insurance,
                            insurance_status, observed, ids, changes,
                            reason="establishment_or_end_date_invalid")
    if (institution.get("ACTIVE") == 0 and end is None) or (
            institution.get("ACTIVE") == 1 and end is not None):
        return BankEvidence(cert, "unresolved_dates", (), "institutions", insurance,
                            insurance_status, observed, ids, changes,
                            reason="current_status_and_end_date_disagree")
    if len(terminal_days) > 1:
        return BankEvidence(cert, "unresolved_dates", (), "institutions", insurance,
                            insurance_status, observed, ids, changes,
                            reason="multiple_distinct_history_terminal_dates")
    conflict_reports = [day for day in observed
                        if day < established or (end is not None and day > end)]
    conflict_start = min(conflict_reports) if conflict_reports else None
    history_conflict = None
    if terminal_days:
        terminal = terminal_days[0]
        if end is None:
            conflict_start = min(filter(None, (conflict_start, terminal)), default=terminal)
        elif terminal != end:
            history_conflict = (min(end, terminal), max(end, terminal))
    interval = ActiveInterval(established, end, "fdic_institutions_estymd_endefymd")
    return BankEvidence(cert, "supported", (interval,), "institution_interval",
                        insurance, insurance_status, observed, ids, changes,
                        conflict_start, history_conflict,
                        reason="operational_interval_from_institution_snapshot")


def retrospective_membership(bank: BankEvidence, q: date) -> MembershipDecision:
    """Evaluate dated activity evidence without using report age or predictors.

    Future reports and events cannot change an earlier q. A later correction
    to an earlier *effective* source date may revise the historical cohort;
    that is a documented retrospective revision, not feature availability.
    """
    def result(status: str, reason: str, interval: ActiveInterval | None = None) -> MembershipDecision:
        return MembershipDecision(bank.cert, q, status, reason, bank.source,
                                  interval.start if interval else None,
                                  interval.end if interval else None)

    if bank.disposition != "supported":
        return result("uncertain", bank.reason or bank.disposition)
    if bank.conflict_start is not None and bank.conflict_start <= q:
        return result("uncertain", "report_or_identity_conflict_effective_by_q")
    if any(day <= q for day in bank.cert_change_dates):
        return result("uncertain", "certificate_change_mapping_unresolved")
    if bank.history_end_conflict is not None:
        earlier, later = bank.history_end_conflict
        if earlier <= q < later:
            return result("uncertain", "institution_history_exit_dates_disagree")
    intervals = tuple(sorted(bank.intervals, key=lambda item: item.start))
    if not intervals:
        return result("uncertain", "no_supported_activity_interval")
    for interval in intervals:
        if interval.start <= q and (interval.end is None or q < interval.end):
            return result("eligible", "within_retrospective_activity_interval", interval)
    if q < intervals[0].start:
        return result("ineligible", "before_establishment", intervals[0])
    return result("ineligible", "exit_effective_on_or_before_q", intervals[-1])


@dataclass(frozen=True)
class RegistryEvent:
    cert: int
    kind: str  # entry, exit, assistance, identity_conflict
    effective_date: date
    known_at: date | None
    source_id: str


def membership_at_q(cert: int, q: date, events: list[RegistryEvent],
                    *, complete_register_known_at_q: bool) -> tuple[str, str]:
    """Return status and reason using only certified q-known facts.

    An exit effective on q takes precedence over same-day entry. Gaps in
    Financials reporting are irrelevant. A later documented entry permits
    re-entry only when it is independently known by q.
    """
    if not complete_register_known_at_q:
        return "uncertain", "no_complete_historical_register_known_at_q"
    relevant = [event for event in events if event.cert == cert
                and event.known_at is not None and event.known_at <= q
                and event.effective_date <= q]
    if any(event.kind == "identity_conflict" for event in relevant):
        return "uncertain", "identity_conflict_known_at_q"
    dated = [event for event in relevant if event.kind in {"entry", "exit"}]
    if not dated:
        return "uncertain", "no_known_entry"
    latest = max(event.effective_date for event in dated)
    same_day = [event for event in dated if event.effective_date == latest]
    if any(event.kind == "exit" for event in same_day):
        return "ineligible", "known_exit_effective_on_or_before_q"
    return "eligible", "known_entry_without_later_known_exit"
