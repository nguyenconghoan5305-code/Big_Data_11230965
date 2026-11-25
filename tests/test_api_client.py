"""FDIC API contract tests use synthetic HTTP responses, never FDIC datasets."""

import pytest
import requests

from bft65b.acquisition.client import (
    AmbiguousPaginationError,
    FDICClient,
    FDICContractError,
    FDICRequestError,
)
from bft65b.config import ProjectConfig


class Response:
    def __init__(self, payload=None, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class Session:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, *, params, timeout):
        self.calls.append((url, dict(params), timeout))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def payload(rows, total, *, fields, offset=0, limit=2, filters="", index="stable", include_id=True):
    if "FAILDATE" in fields and include_id:
        for position, row in enumerate(rows):
            row.setdefault("ID", str(offset + position + 1))
    return {
        "meta": {
            "total": total,
            "parameters": {
                "fields": ",".join(fields),
                "offset": str(offset),
                "limit": str(limit),
                "filters": filters,
            },
            "index": {"name": index, "createTimestamp": "2026-01-01T00:00:00Z"},
        },
        "data": [{"data": row, "score": 0} for row in rows],
    }


FIN = ProjectConfig().financial_fields
FAIL = ProjectConfig().failure_fields
INST = ProjectConfig().institution_fields
QUARTER = "REPDTE:20230331"


def financial(cert):
    return {"CERT": cert, "REPDTE": "20230331", "ASSET": 100}


def client(*responses, **kwargs):
    session = Session(*responses)
    return FDICClient(session=session, sleep=lambda _: None, **kwargs), session


def test_request_parameters_and_response_shape():
    fdic, session = client(Response(payload([financial(14)], 1, fields=FIN, filters=QUARTER)))
    page = fdic.fetch_page("financials", filters=QUARTER, limit=2)
    assert (page.total, page.offset, len(page.rows)) == (1, 0, 1)
    url, params, timeout = session.calls[0]
    assert url.endswith("/financials")
    assert params["sort_by"] == "CERT"
    assert params["sort_order"] == "ASC"
    assert params["format"] == "json"
    assert params["filters"] == QUARTER
    assert timeout == (10.0, 60.0)


@pytest.mark.parametrize(
    "endpoint,fields,rows",
    [
        ("failures", FAIL, [{"CERT": 40, "FAILDATE": "12/31/1980", "COST": 1}]),
        ("institutions", INST, [{"CERT": 4, "ACTIVE": 0, "CHANGEC1": 223}]),
    ],
)
def test_other_endpoint_contracts(endpoint, fields, rows):
    fdic, _ = client(Response(payload(rows, 1, fields=fields)))
    assert fdic.fetch_page(endpoint, limit=2).rows[0]["CERT"] == rows[0]["CERT"]


def test_null_failure_certificate_is_preserved_with_source_id():
    fdic, _ = client(Response(payload(
        [{"CERT": None, "FAILDATE": "5/28/1934", "COST": None}],
        1, fields=FAIL, limit=10000,
    )))
    page = list(fdic.iter_pages("failures", limit=10000, require_nonempty=True))[0]
    assert page.rows[0]["ID"] == "1" and page.rows[0]["CERT"] is None
    assert page.null_fields[0] == ("CERT", "COST")


def test_failure_sort_places_null_cert_after_numeric_cert():
    fdic, _ = client(Response(payload([
        {"CERT": 40, "FAILDATE": "1/2/1984"},
        {"CERT": None, "FAILDATE": "5/28/1934"},
    ], 2, fields=FAIL)))
    assert sum(len(p.rows) for p in fdic.iter_pages("failures", limit=2)) == 2


def test_failure_sort_rejects_numeric_cert_after_null():
    fdic, _ = client(Response(payload([
        {"CERT": None, "FAILDATE": "5/28/1934"},
        {"CERT": 40, "FAILDATE": "1/2/1984"},
    ], 2, fields=FAIL)))
    with pytest.raises(FDICContractError, match="out of order"):
        list(fdic.iter_pages("failures", limit=2))


def test_complete_pagination_and_total():
    fdic, session = client(
        Response(payload([financial(14), financial(35)], 3, fields=FIN, filters=QUARTER)),
        Response(payload([financial(42)], 3, fields=FIN, offset=2, filters=QUARTER)),
    )
    pages = list(fdic.iter_pages("financials", filters=QUARTER, limit=2, require_nonempty=True))
    assert [len(page.rows) for page in pages] == [2, 1]
    assert [call[1]["offset"] for call in session.calls] == [0, 2]


@pytest.mark.parametrize(
    "second,match",
    [
        (payload([financial(35)], 3, fields=FIN, offset=2, filters=QUARTER), "Duplicate"),
        (payload([financial(42)], 4, fields=FIN, offset=2, filters=QUARTER), "total changed"),
        (payload([], 3, fields=FIN, offset=2, filters=QUARTER), "empty intermediate"),
        (payload([financial(42)], 3, fields=FIN, offset=2, filters=QUARTER, index="new"), "index changed"),
        (payload([financial(10)], 3, fields=FIN, offset=2, filters=QUARTER), "sort key"),
    ],
)
def test_cross_page_anomalies(second, match):
    fdic, _ = client(
        Response(payload([financial(14), financial(35)], 3, fields=FIN, filters=QUARTER)),
        Response(second),
    )
    with pytest.raises(FDICContractError, match=match):
        list(fdic.iter_pages("financials", filters=QUARTER, limit=2))


def test_duplicate_inside_page():
    fdic, _ = client(Response(payload([financial(14), financial(14)], 2, fields=FIN, filters=QUARTER)))
    with pytest.raises(FDICContractError, match="Duplicate"):
        list(fdic.iter_pages("financials", filters=QUARTER, limit=2))


def test_short_intermediate_page():
    fdic, _ = client(Response(payload([financial(14)], 3, fields=FIN, filters=QUARTER)))
    with pytest.raises(FDICContractError, match="Short intermediate"):
        list(fdic.iter_pages("financials", filters=QUARTER, limit=2))


@pytest.mark.parametrize(
    "bad,match",
    [
        ({}, "meta"),
        ({"meta": {"total": "3", "parameters": {}}, "data": []}, "total"),
        ({"meta": {"total": 1, "parameters": {}}, "data": []}, "echo"),
        (payload([{"REPDTE": "20230331"}], 1, fields=FIN, filters=QUARTER), "key"),
        (payload([financial(14), financial(35)], 1, fields=FIN, filters=QUARTER), "Page count"),
    ],
)
def test_invalid_response_is_rejected(bad, match):
    fdic, _ = client(Response(bad))
    with pytest.raises(FDICContractError, match=match):
        fdic.fetch_page("financials", filters=QUARTER, limit=2)


def test_retries_timeout_and_503_with_exponential_backoff():
    session = Session(
        requests.Timeout("slow"),
        Response(status_code=503),
        Response(payload([financial(14)], 1, fields=FIN, filters=QUARTER)),
    )
    waits = []
    fdic = FDICClient(session=session, sleep=waits.append, backoff_seconds=0.5)
    assert fdic.fetch_page("financials", filters=QUARTER, limit=2).total == 1
    assert waits == [0.5, 1.0]
    assert len(session.calls) == 3


def test_429_503_and_connection_timeout_recover_with_bounded_backoff():
    session = Session(
        Response(status_code=429), Response(status_code=503),
        requests.ConnectTimeout("connection timed out"),
        Response(payload([financial(14)], 1, fields=FIN, filters=QUARTER)),
    )
    waits = []
    fdic = FDICClient(session=session, sleep=waits.append,
                      max_attempts=4, backoff_seconds=0.25)
    assert fdic.fetch_page("financials", filters=QUARTER, limit=2).total == 1
    assert len(session.calls) == 4 and waits == [0.25, 0.5, 1.0]
    assert all(call[1]["filters"] == QUARTER for call in session.calls)


def test_exhausted_retries_and_nonretryable_error():
    fdic, session = client(
        requests.Timeout("one"), requests.Timeout("two"), max_attempts=2
    )
    with pytest.raises(FDICRequestError, match="2 attempts"):
        fdic.fetch_page("financials", filters=QUARTER, limit=2)
    assert len(session.calls) == 2
    fdic, session = client(Response(status_code=400))
    with pytest.raises(FDICRequestError, match="Non-retryable HTTP 400"):
        fdic.fetch_page("financials", filters=QUARTER, limit=2)
    assert len(session.calls) == 1


def test_retryable_http_exhausted():
    fdic, _ = client(Response(status_code=429), Response(status_code=503), max_attempts=2)
    with pytest.raises(FDICRequestError, match="HTTP 503 after 2 attempts"):
        fdic.fetch_page("financials", filters=QUARTER, limit=2)


@pytest.mark.parametrize(
    "endpoint,kwargs",
    [
        ("unknown", {}),
        ("financials", {"filters": "REPDTE:2023-03-31"}),
        ("financials", {"filters": QUARTER, "fields": ("CERT", "REPDTE", "UNVERIFIED")}),
        ("failures", {"fields": ("CERT", "COST")}),
        ("institutions", {"limit": 10001}),
        ("institutions", {"offset": -1}),
    ],
)
def test_invalid_request_never_reaches_network(endpoint, kwargs):
    fdic, session = client()
    with pytest.raises(ValueError):
        fdic.fetch_page(endpoint, **kwargs)
    assert session.calls == []


def test_unverified_configuration_blocks_requests():
    with pytest.raises(ValueError, match="not been verified"):
        FDICClient(config=ProjectConfig(fields_verified=False), session=Session())


def test_empty_query_can_be_declared_invalid():
    fdic, _ = client(Response(payload([], 0, fields=FIN, filters=QUARTER)))
    with pytest.raises(FDICContractError, match="zero records"):
        list(fdic.iter_pages("financials", filters=QUARTER, limit=2, require_nonempty=True))


def test_invalid_json_is_contract_failure():
    fdic, _ = client(Response(ValueError("not JSON")))
    with pytest.raises(FDICContractError, match="invalid JSON"):
        fdic.fetch_page("financials", filters=QUARTER, limit=2)


def test_certificate_order_is_numeric_across_page_boundary():
    fdic, _ = client(
        Response(payload([{"CERT": 9}, {"CERT": 10}], 3, fields=INST)),
        Response(payload([{"CERT": 14}], 3, fields=INST, offset=2)),
    )
    assert sum(len(page.rows) for page in fdic.iter_pages("institutions", limit=2)) == 3


def test_distinct_failures_with_same_cert_on_one_page_are_preserved():
    rows = [
        {"CERT": 40, "FAILDATE": "12/31/1980"},
        {"CERT": "40", "FAILDATE": "1/1/1981"},
    ]
    fdic, session = client(Response(payload(rows, 2, fields=FAIL)))
    pages = list(fdic.iter_pages("failures", limit=2))
    assert [row for page in pages for row in page.rows] == rows
    assert len(session.calls) == 1


def test_failures_cert_tie_across_page_boundary_is_explicit_blocker():
    fdic, _ = client(
        Response(payload([
            {"CERT": 10, "FAILDATE": "1/1/1980"},
            {"CERT": 40, "FAILDATE": "12/31/1980"},
        ], 3, fields=FAIL)),
        Response(payload([
            {"CERT": 40, "FAILDATE": "1/1/1981"},
        ], 3, fields=FAIL, offset=2)),
    )
    with pytest.raises(AmbiguousPaginationError, match="page boundary"):
        list(fdic.iter_pages("failures", limit=2))


def test_failures_tie_within_page_can_continue_when_next_page_starts_new_cert():
    fdic, _ = client(
        Response(payload([
            {"CERT": 40, "FAILDATE": "12/31/1980"},
            {"CERT": 40, "FAILDATE": "1/1/1981"},
        ], 3, fields=FAIL)),
        Response(payload([
            {"CERT": 41, "FAILDATE": "1/2/1981"},
        ], 3, fields=FAIL, offset=2)),
    )
    assert sum(len(p.rows) for p in fdic.iter_pages("failures", limit=2)) == 3


def test_distinct_source_ids_preserve_same_cert_and_date():
    fdic, _ = client(Response(payload([
        {"CERT": "0040", "FAILDATE": "1/2/1984"},
        {"CERT": 40, "FAILDATE": "1984-01-02"},
    ], 2, fields=FAIL)))
    assert sum(len(p.rows) for p in fdic.iter_pages("failures", limit=2)) == 2


def test_duplicate_failure_source_id_is_rejected():
    fdic, _ = client(Response(payload([
        {"ID": "same", "CERT": 40, "FAILDATE": "1/2/1984"},
        {"ID": "same", "CERT": 40, "FAILDATE": "1/3/1984"},
    ], 2, fields=FAIL)))
    with pytest.raises(FDICContractError, match="Duplicate"):
        list(fdic.iter_pages("failures", limit=2))


@pytest.mark.parametrize("source_id", [None, "", "   ", 17])
def test_missing_or_invalid_failure_source_id_blocks_page(source_id):
    row = {"CERT": None, "FAILDATE": "5/28/1934"}
    if source_id is not None:
        row["ID"] = source_id
    fdic, _ = client(Response(payload([row], 1, fields=FAIL, include_id=False)))
    with pytest.raises(FDICContractError, match="endpoint key|source ID"):
        fdic.fetch_page("failures", limit=2)


@pytest.mark.parametrize("date_value", ["1/2/1984", "1984-01-02", "19840102", "1984-01-02T00:00:00Z"])
def test_known_historical_failure_date_representations_preserve_raw_value(date_value):
    fdic, _ = client(Response(payload([{"CERT": 40, "FAILDATE": date_value}], 1, fields=FAIL)))
    assert list(fdic.iter_pages("failures", limit=2))[0].rows[0]["FAILDATE"] == date_value


def test_unknown_failure_date_format_blocks_unverified_key_normalization():
    fdic, _ = client(Response(payload([{"CERT": 40, "FAILDATE": "Jan 2 1984"}], 1, fields=FAIL)))
    with pytest.raises(FDICContractError, match="FAILDATE format"):
        list(fdic.iter_pages("failures", limit=2))


def test_full_field_response_distinguishes_absent_from_explicit_null():
    row = {
        "CERT": 14, "REPDTE": "20230331", "STNAME": "MASSACHUSETTS",
        "ASSET": 287069000, "DEP": 230488000, "EQ": "26373000",
        "NETINC": 616000.0, "ROA": None, "NCLNLS": 45000,
        "RBC1AAJ": 6.76, "ID": "14_20230331",
    }
    fdic, _ = client(Response(payload([row], 1, fields=FIN, filters=QUARTER)))
    page = fdic.fetch_page("financials", filters=QUARTER, limit=2)
    assert page.rows[0] == row  # Preserve source types and unrequested ID.
    assert page.missing_fields == (("LNATRES",),)
    assert page.null_fields == (("ROA",),)


def test_nonkey_financial_fields_may_be_absent_or_null_but_keys_are_required():
    rows = [
        {"CERT": 14, "REPDTE": "20230331", "ASSET": None},
        {"CERT": 15, "REPDTE": "20230331"},
    ]
    fdic, _ = client(Response(payload(rows, 2, fields=FIN, filters=QUARTER)))
    page = list(fdic.iter_pages("financials", filters=QUARTER, limit=2))[0]
    assert "ASSET" in page.null_fields[0]
    assert "ASSET" in page.missing_fields[1]
    assert page.rows == tuple(rows)


@pytest.mark.parametrize("bad_row", [
    {"CERT": 14, "REPDTE": None},
    {"CERT": None, "REPDTE": "20230331"},
    {"CERT": 14, "REPDTE": "20230630"},
])
def test_missing_null_or_wrong_quarter_key_is_rejected(bad_row):
    fdic, _ = client(Response(payload([bad_row], 1, fields=FIN, filters=QUARTER)))
    with pytest.raises(FDICContractError):
        fdic.fetch_page("financials", filters=QUARTER, limit=2)


@pytest.mark.parametrize("quarter", ["20230515", "20240229", "20230231", "19831231", "20260930"])
def test_nonquarter_or_out_of_scope_financial_date_is_rejected_before_network(quarter):
    fdic, session = client()
    with pytest.raises(ValueError):
        fdic.fetch_page("financials", filters="REPDTE:" + quarter, limit=2)
    assert session.calls == []


@pytest.mark.parametrize("quarter", ["19840331", "20260630"])
def test_assignment_boundary_quarters_are_allowed(quarter):
    fields = FIN
    row = {"CERT": 1, "REPDTE": quarter}
    fdic, _ = client(Response(payload([row], 1, fields=fields, filters="REPDTE:"+quarter)))
    assert fdic.fetch_page("financials", filters="REPDTE:"+quarter, limit=2).total == 1

@pytest.mark.parametrize("bad_cert", [40.5, True, "40.5"])
def test_noninteger_cert_never_truncates_or_passes_as_boolean(bad_cert):
    fdic, _ = client(Response(payload([
        {"CERT": bad_cert, "REPDTE": "20230331"},
    ], 1, fields=FIN, filters=QUARTER)))
    with pytest.raises(FDICContractError, match="integer identifier"):
        list(fdic.iter_pages("financials", filters=QUARTER, limit=2))


def test_absent_optional_failure_cost_is_recorded_without_losing_event():
    row = {"CERT": "40", "FAILDATE": "12/31/1980"}
    fdic, _ = client(Response(payload([row], 1, fields=FAIL)))
    page = list(fdic.iter_pages("failures", limit=2))[0]
    assert page.rows == (row,)
    assert page.missing_fields == (("COST",),)
    assert page.null_fields == ((),)
