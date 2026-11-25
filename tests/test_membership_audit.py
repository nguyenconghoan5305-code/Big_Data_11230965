"""Synthetic membership-source and conditional as-of contract tests."""

from datetime import date, timedelta
from pathlib import Path

import pytest

from bft65b.panel.active_audit import _fetch_history, _fetch_institutions
from bft65b.panel.membership_rule import RegistryEvent, membership_at_q


Q = date(2023, 6, 30)


def fact(kind, day, *, known=None, cert=11, source="event"):
    return RegistryEvent(cert, kind, day, known or day, source)


def decide(events, *, complete=True):
    return membership_at_q(11, Q, events, complete_register_known_at_q=complete)


def test_entry_exit_and_same_day_precedence():
    entry = fact("entry", Q)
    assert decide([entry])[0] == "eligible"
    assert decide([fact("entry", Q + timedelta(days=1))])[0] == "uncertain"
    assert decide([entry, fact("exit", Q)])[0] == "ineligible"
    assert decide([fact("entry", date(2022, 1, 1)),
                   fact("exit", Q - timedelta(days=1))])[0] == "ineligible"
    assert decide([fact("entry", date(2022, 1, 1)),
                   fact("exit", Q + timedelta(days=1))])[0] == "eligible"
    assert decide([entry], complete=False)[0] == "uncertain"


def test_future_knowledge_assistance_gaps_and_identity():
    entry = fact("entry", date(2022, 1, 1))
    baseline = decide([entry])
    assert decide([entry, fact("assistance", Q - timedelta(days=30))]) == baseline
    # Reporting gaps never enter this register-based decision.
    assert decide([entry, fact("exit", Q - timedelta(days=1),
                               known=Q + timedelta(days=1))]) == baseline
    assert decide([entry, fact("exit", Q + timedelta(days=1))]) == baseline
    assert decide([entry, fact("entry", Q, cert=99)]) == baseline
    assert decide([entry, fact("identity_conflict", Q)])[0] == "uncertain"
    assert decide([entry, fact("exit", Q - timedelta(days=1)),
                   fact("entry", Q)])[0] == "eligible"


def test_history_preserves_source_id_and_filters_institution_role(monkeypatch, tmp_path):
    monkeypatch.setattr("bft65b.panel.active_audit.SCOPE_YEARS", range(2023, 2024))
    def response(_session, endpoint, params):
        assert endpoint == "history" and "ORG_ROLE_CDE:FI" in params["filters"]
        return {"meta": {"total": 1, "index": {"name": "h", "createTimestamp": "t"}},
                "data": [{"data": {"ID": "source-1", "CERT": 11,
                          "ORG_ROLE_CDE": "FI", "EFFDATE": "2023-01-01T00:00:00",
                          "PROCDATE": "2023-03-01T00:00:00"}}]}
    monkeypatch.setattr("bft65b.panel.active_audit._request", response)
    rows, records, index = _fetch_history(None, tmp_path)
    assert rows[0]["ID"] == "source-1"
    assert records[0]["rows"] == 1 and index == ("h", "t")


def test_history_rejects_branch_leakage_and_incomplete_year(monkeypatch, tmp_path):
    monkeypatch.setattr("bft65b.panel.active_audit.SCOPE_YEARS", range(2023, 2024))
    def response(_session, _endpoint, _params):
        return {"meta": {"total": 2, "index": {"name": "h", "createTimestamp": "t"}},
                "data": [{"data": {"ID": "branch", "ORG_ROLE_CDE": "BR",
                          "EFFDATE": "2023-01-01T00:00:00"}}]}
    monkeypatch.setattr("bft65b.panel.active_audit._request", response)
    with pytest.raises(ValueError, match="requires verified subpartition"):
        _fetch_history(None, tmp_path)


def test_institutions_duplicate_cert_fails(monkeypatch, tmp_path):
    def response(_session, _endpoint, _params):
        return {"meta": {"total": 2, "index": {"name": "i", "createTimestamp": "t"}},
                "data": [{"data": {"CERT": 11}}, {"data": {"CERT": 11}}]}
    monkeypatch.setattr("bft65b.panel.active_audit._request", response)
    with pytest.raises(ValueError, match="total/key/order"):
        _fetch_institutions(None, tmp_path)
