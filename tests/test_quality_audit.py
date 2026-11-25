"""Synthetic P2.1 defect fixtures with exact counts and denominators."""

from datetime import date
from decimal import Decimal

import pytest

from bft65b.quality import audit as workflow
from bft65b.quality.rules import inspect_period


Q1 = date(2023, 3, 31)
Q2 = date(2023, 6, 30)


def financial(cert=1, day=Q1, **values):
    row = {"CERT": cert, "REPDTE": day, "ASSET": Decimal("100.0000"),
           "DEP": Decimal("50.0000"), "DEPDOM": Decimal("40.0000"),
           "ROA": 1.0, "SOURCE_MISSING_FIELDS": []}
    row.update(values)
    return row


def failure(source_id="f1", cert=1, day=date(2023, 5, 1), **values):
    row = {"ID": source_id, "CERT": cert, "FAILDATE": day,
           "COST": 1.0, "SOURCE_MISSING_FIELDS": []}
    row.update(values)
    return row


def institution(cert=1, **values):
    row = {"CERT": cert, "STNAME": "State", "ACTIVE": 1,
           "CHANGEC1": None, "SOURCE_MISSING_FIELDS": ["CHANGEC1"]}
    row.update(values)
    return row


def rules(result):
    return {(x["rule"], x["field"]): x for x in result.summary["rule_findings"]}


def test_valid_financial_record_has_no_rule_and_exact_denominators():
    result = inspect_period("financials", [financial()],
                            ("CERT", "REPDTE", "ASSET", "DEP", "DEPDOM", "ROA"),
                            "20230331", partition="20230331", institution_certs={1})
    assert result.summary["rows"] == 1
    assert result.summary["affected_records_unique"] == 0
    assert result.summary["field_counts"]["ASSET"] == {
        "absent": 0, "explicit_null": 0, "non_null": 1, "denominator": 1}
    assert result.summary["institution_snapshot_join"]["matched_records"] == 1
    assert not result.summary["rule_findings"]


def test_source_absent_vs_explicit_null_and_union_count():
    rows = [financial(ASSET=None, DEP=None, SOURCE_MISSING_FIELDS=["DEP"]), financial(cert=2)]
    result = inspect_period("financials", rows, ("CERT", "REPDTE", "ASSET", "DEP"),
                            "20230331", partition="20230331")
    fields = result.summary["field_counts"]
    assert fields["ASSET"] == {"absent": 0, "explicit_null": 1, "non_null": 1, "denominator": 2}
    assert fields["DEP"] == {"absent": 1, "explicit_null": 0, "non_null": 1, "denominator": 2}
    assert result.summary["affected_records_unique"] == 1
    assert result.summary["missing_or_null_records_unique"] == 1


def test_duplicate_financial_key_counts_extra_row_once():
    result = inspect_period("financials", [financial(), financial()],
                            ("CERT", "REPDTE", "ASSET"), "20230331", partition="20230331")
    assert rules(result)[("duplicate_primary_key", "CERT,REPDTE")]["count"] == 1
    assert rules(result)[("duplicate_primary_key", "CERT,REPDTE")]["denominator"] == 2
    assert result.summary["affected_records_unique"] == 1


def test_missing_financial_key_is_distinct_from_source_null():
    result = inspect_period("financials", [financial(cert=None)],
                            ("CERT", "REPDTE", "ASSET"), "20230331", partition="20230331")
    assert result.summary["field_counts"]["CERT"]["explicit_null"] == 1
    assert rules(result)[("missing_primary_key", "CERT,REPDTE")]["count"] == 1
    assert result.summary["affected_records_unique"] == 1


@pytest.mark.parametrize("day,invalid,partition", [
    (date(2023, 4, 1), 1, 1), (Q2, 0, 1),
])
def test_dates_and_partition_are_separate_rules(day, invalid, partition):
    result = inspect_period("financials", [financial(day=day)],
                            ("CERT", "REPDTE", "ASSET"), "20230331", partition="20230331")
    findings = rules(result)
    assert findings.get(("invalid_report_date", "REPDTE"), {}).get("count", 0) == invalid
    assert findings.get(("partition_date_mismatch", "REPDTE"), {}).get("count", 0) == partition
    assert result.summary["affected_records_unique"] == 1


def test_negative_total_assets_is_definite_violation():
    result = inspect_period("financials", [financial(ASSET=Decimal("-1.0000"))],
                            ("CERT", "REPDTE", "ASSET"), "20230331", partition="20230331")
    item = rules(result)[("negative_total_assets", "ASSET")]
    assert (item["count"], item["denominator"], item["severity"]) == (1, 1, "violation")


def test_domestic_deposits_above_total_is_review_not_definite_error():
    result = inspect_period("financials", [financial(DEPDOM=Decimal("51.0000"))],
                            ("CERT", "REPDTE", "DEP", "DEPDOM"), "20230331", partition="20230331")
    item = rules(result)[("domestic_deposits_above_total", "DEPDOM,DEP")]
    assert (item["count"], item["severity"]) == (1, "review")


def test_invalid_numeric_type_and_nonfinite_are_counted():
    result = inspect_period("financials", [financial(ASSET="bad", ROA=float("nan"))],
                            ("CERT", "REPDTE", "ASSET", "ROA"), "20230331", partition="20230331")
    assert rules(result)[("invalid_value_type", "ASSET")]["count"] == 1
    assert rules(result)[("nonfinite_numeric", "ROA")]["count"] == 1
    assert result.summary["affected_records_unique"] == 1


def test_invalid_source_provenance_and_absent_value_conflict():
    result = inspect_period("financials", [financial(SOURCE_MISSING_FIELDS=["ASSET"])],
                            ("CERT", "REPDTE", "ASSET"), "20230331", partition="20230331")
    assert result.summary["field_counts"]["ASSET"]["absent"] == 1
    assert rules(result)[("absent_field_has_value", "ASSET")]["count"] == 1
    bad = inspect_period("financials", [financial(SOURCE_MISSING_FIELDS=["NOT_REQUESTED"])],
                         ("CERT", "REPDTE", "ASSET"), "20230331", partition="20230331")
    assert rules(bad)[("invalid_provenance", "SOURCE_MISSING_FIELDS")]["count"] == 1


def test_institutions_absent_change_code_not_explicit_null():
    result = inspect_period("institutions", [institution(), institution(2, ACTIVE=2)],
                            ("CERT", "STNAME", "ACTIVE", "CHANGEC1"), "snapshot")
    assert result.summary["field_counts"]["CHANGEC1"] == {
        "absent": 2, "explicit_null": 0, "non_null": 0, "denominator": 2}
    assert rules(result)[("active_outside_binary", "ACTIVE")]["count"] == 1
    assert result.summary["affected_records_unique"] == 2


def test_duplicate_institution_cert_and_null_key():
    result = inspect_period("institutions", [institution(SOURCE_MISSING_FIELDS=[]),
                            institution(SOURCE_MISSING_FIELDS=[]),
                            institution(None, SOURCE_MISSING_FIELDS=[])],
                            ("CERT", "ACTIVE"), "snapshot")
    assert rules(result)[("duplicate_primary_key", "CERT")]["count"] == 1
    assert rules(result)[("missing_primary_key", "CERT")]["count"] == 1
    assert result.summary["affected_records_unique"] == 2


def test_financial_to_institution_join_record_and_distinct_denominators():
    rows = [financial(1, Q1), financial(1, Q2), financial(2, Q1)]
    result = inspect_period("financials", rows, ("CERT", "REPDTE", "ASSET"),
                            "20230331", partition="20230331", institution_certs={1})
    join = result.summary["institution_snapshot_join"]
    assert (join["matched_records"], join["unmatched_records"], join["record_denominator"]) == (2, 1, 3)
    assert (join["matched_distinct_certs"], join["unmatched_distinct_certs"],
            join["distinct_cert_denominator"]) == (1, 1, 2)
    assert rules(result)[("financial_cert_absent_from_institutions", "CERT")]["count"] == 1


def test_failure_null_cert_keeps_source_id_as_primary_key():
    result = inspect_period("failures", [failure(cert=None, COST=None)],
                            ("CERT", "FAILDATE", "COST"), "2023", financial_reports={})
    assert result.summary["field_counts"]["CERT"]["explicit_null"] == 1
    assert result.summary["failure_linkage"]["record_categories"] == {"missing_cert": 1}
    assert result.summary["failure_linkage"]["distinct_source_id_by_category"] == {"missing_cert": 1}
    assert ("missing_primary_key", "ID") not in rules(result)
    assert result.summary["affected_records_unique"] == 1


def test_duplicate_failure_source_id_and_missing_id():
    result = inspect_period("failures", [failure(), failure(), failure(source_id="")],
                            ("CERT", "FAILDATE", "COST"), "2023", financial_reports={1: [Q1]})
    assert rules(result)[("duplicate_primary_key", "ID")]["count"] == 1
    assert rules(result)[("missing_primary_key", "ID")]["count"] == 1
    assert result.summary["failure_linkage"]["record_denominator"] == 3


def test_invalid_failure_date_is_reported_without_dropping_source_id():
    result = inspect_period("failures", [failure(day="not-a-date")],
                            ("CERT", "FAILDATE", "COST"), "undated",
                            financial_reports={1: [Q1]})
    assert rules(result)[("invalid_failure_date", "FAILDATE")]["count"] == 1
    assert rules(result)[("invalid_value_type", "FAILDATE")]["count"] == 1
    assert result.summary["failure_linkage"]["record_categories"] == {"invalid_date": 1}
    assert result.summary["affected_records_unique"] == 1


def test_failure_unmatched_vs_future_only_chronology():
    rows = [failure("a", cert=3), failure("b", cert=4)]
    result = inspect_period("failures", rows, ("CERT", "FAILDATE", "COST"),
                            "2023", financial_reports={4: [Q2]})
    assert result.summary["failure_linkage"]["record_categories"] == {
        "future_reports_only": 1, "no_financial_cert": 1}
    assert rules(result)[("failure_cert_absent_from_financials", "CERT")]["count"] == 1
    assert rules(result)[("failure_only_future_reports", "CERT,FAILDATE")]["count"] == 1


def test_multiple_prior_reports_and_duplicate_latest_are_distinct():
    rows = [failure("a", cert=1), failure("b", cert=2)]
    result = inspect_period("failures", rows, ("CERT", "FAILDATE", "COST"), "2023",
                            financial_reports={1: [date(2022, 12, 31), Q1], 2: [Q1, Q1]})
    assert result.summary["failure_linkage"]["record_categories"] == {
        "ambiguous_latest_report": 1, "chronological_candidate": 1}
    assert rules(result)[("failure_multiple_prior_reports", "CERT,FAILDATE")]["count"] == 2
    assert rules(result)[("failure_ambiguous_latest_report", "CERT,REPDTE")]["count"] == 1
    assert result.summary["failure_linkage"]["distinct_nonnull_cert_by_category"] == {
        "ambiguous_latest_report": 1, "chronological_candidate": 1}


def test_provenance_reconciliation_counts_and_denominator():
    scan = inspect_period("institutions", [institution(), institution(2, CHANGEC1="X", SOURCE_MISSING_FIELDS=[])],
                          ("CERT", "CHANGEC1"), "snapshot")
    source = {"missing_fields": {"CERT": 0, "CHANGEC1": 1},
              "null_fields": {"CERT": 0, "CHANGEC1": 0}}
    workflow._reconcile_fields(scan.summary, source, ("CERT", "CHANGEC1"), "institutions", "snapshot")
    source["null_fields"]["CHANGEC1"] = 1
    with pytest.raises(ValueError, match="null-versus-absent"):
        workflow._reconcile_fields(scan.summary, source, ("CERT", "CHANGEC1"), "institutions", "snapshot")


def test_aggregate_affected_rows_sums_disjoint_period_unions():
    one = inspect_period("financials", [financial(ASSET=None, DEP=None)],
                         ("CERT", "REPDTE", "ASSET", "DEP"), "20230331", partition="20230331")
    two = inspect_period("financials", [financial(2, Q2, ASSET=None)],
                         ("CERT", "REPDTE", "ASSET", "DEP"), "20230630", partition="20230630")
    combined = workflow._aggregate("financials", [one, two], ("CERT", "REPDTE", "ASSET", "DEP"))
    assert combined["rows"] == 2
    assert combined["affected_records_unique"] == 2
    assert combined["field_counts"]["ASSET"]["explicit_null"] == 2
    assert combined["field_counts"]["DEP"]["explicit_null"] == 1


def test_wrong_attempt_is_rejected_before_reading_real_files(tmp_path):
    from bft65b.config import ProjectConfig, EXTENDED_FINANCIAL_FIELDS
    config = ProjectConfig(financial_fields=EXTENDED_FINANCIAL_FIELDS)
    with pytest.raises(ValueError, match="pinned"):
        workflow._load_snapshot(tmp_path, config, "unaccepted")
