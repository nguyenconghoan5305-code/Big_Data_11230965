"""Validated, paginated access to the three FDIC BankFind endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import requests

from bft65b.config import EXTENDED_FINANCIAL_FIELDS, ProjectConfig

BASE_URL = "https://api.fdic.gov/banks"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_PAGE_SIZE = 10_000

# Only codes checked against the FDIC endpoint dictionaries are allowed here.
VERIFIED_FIELDS = {
    "financials": frozenset(EXTENDED_FINANCIAL_FIELDS),
    "failures": frozenset("CERT FAILDATE COST".split()),
    "institutions": frozenset("CERT STNAME ACTIVE CHANGEC1".split()),
}
KEY_FIELDS = {
    "financials": ("CERT", "REPDTE"),
    "failures": ("CERT", "FAILDATE"),
    "institutions": ("CERT",),
}
NONNULL_RESPONSE_FIELDS = {
    "financials": ("CERT", "REPDTE"),
    "failures": ("ID", "FAILDATE"),
    "institutions": ("CERT",),
}
SORT_BY = {
    "financials": "CERT",
    "failures": "CERT",
    "institutions": "CERT",
}


class FDICRequestError(RuntimeError):
    """HTTP or transport failure after the permitted attempts."""


class FDICContractError(ValueError):
    """A response or page sequence violates the expected API contract."""

class AmbiguousPaginationError(FDICContractError):
    """A repeated sort value crosses pages, so offset paging is unsafe."""


@dataclass(frozen=True)
class Page:
    endpoint: str
    offset: int
    limit: int
    total: int
    rows: tuple[dict[str, Any], ...]
    index: tuple[str | None, str | None] | None
    missing_fields: tuple[tuple[str, ...], ...]
    null_fields: tuple[tuple[str, ...], ...]


class FDICClient:
    """Read API pages without writing datasets; callers own storage and manifests."""

    def __init__(
        self,
        config: ProjectConfig | None = None,
        *,
        session: requests.Session | None = None,
        timeout: tuple[float, float] = (10.0, 60.0),
        max_attempts: int = 4,
        backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or ProjectConfig()
        if not self.config.fields_verified:
            raise ValueError("Configured field codes have not been verified")
        if len(timeout) != 2 or any(v <= 0 for v in timeout):
            raise ValueError("Both connection and read timeouts must be positive")
        if max_attempts < 1 or backoff_seconds < 0 or max_backoff_seconds < 0:
            raise ValueError("Invalid retry settings")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.sleep = sleep
        self.session = session if session is not None else requests.Session()
        if hasattr(self.session, "headers"):
            self.session.headers.update({"User-Agent": "BFT65B-research/1.1"})

    def _fields(self, endpoint: str, fields: Sequence[str] | None) -> tuple[str, ...]:
        if endpoint not in VERIFIED_FIELDS:
            raise ValueError(f"Unsupported FDIC endpoint: {endpoint}")
        defaults = {
            "financials": self.config.financial_fields,
            "failures": self.config.failure_fields,
            "institutions": self.config.institution_fields,
        }
        selected = tuple(defaults[endpoint] if fields is None else fields)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Fields must be nonempty and unique")
        if not set(selected).issubset(VERIFIED_FIELDS[endpoint]):
            raise ValueError(f"Unverified field requested for {endpoint}")
        if not set(KEY_FIELDS[endpoint]).issubset(selected):
            raise ValueError(f"Required key fields missing for {endpoint}")
        return selected

    def fetch_page(
        self,
        endpoint: str,
        *,
        fields: Sequence[str] | None = None,
        filters: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> Page:
        selected = self._fields(endpoint, fields)
        size = self.config.api_page_size if limit is None else limit
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_PAGE_SIZE:
            raise ValueError(f"limit must be an integer from 1 to {MAX_PAGE_SIZE}")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if endpoint == "financials":
            if not isinstance(filters, str) or not re.fullmatch(r"REPDTE:\d{8}", filters):
                raise ValueError("Financials require REPDTE:YYYYMMDD; ISO date filters return zero rows")
            try:
                report_date = datetime.strptime(filters[7:], "%Y%m%d").date()
            except ValueError as exc:
                raise ValueError("Invalid REPDTE quarter date") from exc
            if (report_date.month, report_date.day) not in {
                (3, 31), (6, 30), (9, 30), (12, 31)
            }:
                raise ValueError("REPDTE must be a calendar quarter-end")
            if not self.config.financial_start <= report_date.isoformat() <= self.config.financial_end:
                raise ValueError("REPDTE is outside the assignment Financials range")
        elif filters is not None and (
            not isinstance(filters, str)
            or not filters.strip()
            or any(ord(char) < 32 for char in filters)
        ):
            raise ValueError("Invalid filter expression")

        params: dict[str, str | int] = {
            "fields": ",".join(selected),
            "limit": size,
            "offset": offset,
            "sort_by": SORT_BY[endpoint],
            "sort_order": "ASC",
            "format": "json",
        }
        if filters is not None:
            params["filters"] = filters
        url = f"{BASE_URL}/{endpoint}"
        for attempt in range(self.max_attempts):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt + 1 == self.max_attempts:
                    raise FDICRequestError(f"Transport failure after {self.max_attempts} attempts: {url}") from exc
                self._backoff(attempt)
                continue
            if response.status_code in RETRYABLE_STATUS:
                if attempt + 1 == self.max_attempts:
                    raise FDICRequestError(
                        f"HTTP {response.status_code} after {self.max_attempts} attempts: {url}"
                    )
                self._backoff(attempt)
                continue
            if response.status_code != 200:
                raise FDICRequestError(f"Non-retryable HTTP {response.status_code}: {url}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise FDICContractError("HTTP 200 contained invalid JSON") from exc
            return self._parse_page(endpoint, payload, selected, filters, offset, size)
        raise AssertionError("Retry loop exhausted without a result")

    def _backoff(self, attempt: int) -> None:
        self.sleep(min(self.max_backoff_seconds, self.backoff_seconds * 2**attempt))

    @staticmethod
    def _parse_page(
        endpoint: str,
        payload: Any,
        fields: tuple[str, ...],
        filters: str | None,
        offset: int,
        limit: int,
    ) -> Page:
        if not isinstance(payload, dict) or not isinstance(payload.get("meta"), dict):
            raise FDICContractError("Missing meta object")
        meta = payload["meta"]
        total = meta.get("total")
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise FDICContractError("meta.total must be a nonnegative integer")
        parameters = meta.get("parameters")
        if not isinstance(parameters, dict):
            raise FDICContractError("Missing meta.parameters")
        expected = {
            "fields": ",".join(fields),
            "limit": str(limit),
            "offset": str(offset),
            "filters": filters or "",
        }
        for name, value in expected.items():
            if str(parameters.get(name, "")) != value:
                raise FDICContractError(f"API did not echo requested {name}")
        records = payload.get("data")
        if not isinstance(records, list):
            raise FDICContractError("Missing data list")
        if len(records) > limit or offset + len(records) > total:
            raise FDICContractError("Page count exceeds limit or API total")
        if offset >= total and records:
            raise FDICContractError("Records returned beyond API total")
        rows = []
        missing_fields = []
        null_fields = []
        for item in records:
            if not isinstance(item, dict) or not isinstance(item.get("data"), dict):
                raise FDICContractError("Record lacks a data object")
            row = dict(item["data"])
            if any(key not in row or row[key] is None for key in NONNULL_RESPONSE_FIELDS[endpoint]):
                raise FDICContractError("Record lacks a non-null endpoint key")
            if endpoint == "failures" and (
                not isinstance(row["ID"], str) or not row["ID"].strip()
            ):
                raise FDICContractError("Failures source ID must be a nonempty string")
            if endpoint == "financials" and str(row["REPDTE"]) != filters[7:]:
                raise FDICContractError("Financials REPDTE does not match the requested quarter")
            missing_fields.append(tuple(field for field in fields if field not in row))
            null_fields.append(tuple(field for field in fields if field in row and row[field] is None))
            rows.append(row)
        index = meta.get("index")
        fingerprint = None
        if index is not None:
            if not isinstance(index, dict):
                raise FDICContractError("meta.index must be an object")
            fingerprint = (index.get("name"), index.get("createTimestamp"))
        return Page(endpoint, offset, limit, total, tuple(rows), fingerprint, tuple(missing_fields), tuple(null_fields))

    @staticmethod
    def _certificate_number(value: Any) -> int:
        if isinstance(value, bool):
            raise FDICContractError("CERT must be an integer identifier")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"\d+", value):
            return int(value)
        raise FDICContractError("CERT must be an integer identifier")

    @staticmethod
    def _failure_date(value: Any) -> str:
        if not isinstance(value, str):
            raise FDICContractError("FAILDATE must be a date string")
        for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except ValueError:
                pass
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
        except ValueError as exc:
            raise FDICContractError(f"Unrecognized FAILDATE format: {value}") from exc

    @classmethod
    def _row_key(cls, endpoint: str, row: Mapping[str, Any]) -> tuple[str, ...]:
        if endpoint == "failures":
            source_id = row.get("ID")
            if not isinstance(source_id, str) or not source_id.strip():
                raise FDICContractError("Failures source ID must be a nonempty string")
            cls._failure_date(row["FAILDATE"])
            return (source_id,)
        cert = str(cls._certificate_number(row["CERT"]))
        if endpoint == "financials":
            return (cert, str(row["REPDTE"]))
        return (cert,)

    @classmethod
    def _sort_key(cls, endpoint: str, row: Mapping[str, Any]) -> tuple[int, ...]:
        if endpoint == "failures" and row.get("CERT") is None:
            # The complete live Failures response places null CERT after numeric CERT.
            return (1, 0)
        if endpoint == "failures":
            return (0, cls._certificate_number(row["CERT"]))
        return (cls._certificate_number(row["CERT"]),)

    def iter_pages(
        self,
        endpoint: str,
        *,
        fields: Sequence[str] | None = None,
        filters: str | None = None,
        limit: int | None = None,
        require_nonempty: bool = False,
    ) -> Iterator[Page]:
        """Yield every page, stopping only after count and cross-page checks pass."""
        offset = 0
        total = None
        index = None
        seen: set[tuple[str, ...]] = set()
        previous_sort: tuple[Any, ...] | None = None
        while True:
            page = self.fetch_page(
                endpoint, fields=fields, filters=filters, offset=offset, limit=limit
            )
            if total is None:
                total, index = page.total, page.index
                if require_nonempty and total == 0:
                    raise FDICContractError("Query unexpectedly returned zero records")
            elif page.total != total:
                raise FDICContractError("API total changed between pages")
            elif page.index != index:
                raise FDICContractError("API index changed between pages")
            if not page.rows and offset < total:
                raise FDICContractError("Unexpectedly empty intermediate page")
            if offset + len(page.rows) < total and len(page.rows) < page.limit:
                raise FDICContractError("Short intermediate page")
            if (
                endpoint == "failures"
                and offset > 0
                and page.rows
                and self._sort_key(endpoint, page.rows[0]) == previous_sort
            ):
                raise AmbiguousPaginationError(
                    "Repeated Failures CERT crosses a page boundary; use a single "
                    "API page if total <= 10000, otherwise stop and use an official "
                    "bulk source or a separately verified partition strategy"
                )
            for row in page.rows:
                key = self._row_key(endpoint, row)
                if key in seen:
                    raise FDICContractError(f"Duplicate or overlapping record: {key}")
                seen.add(key)
                sort_key = self._sort_key(endpoint, row)
                if previous_sort is not None and sort_key < previous_sort:
                    raise FDICContractError("API sort key is out of order")
                if previous_sort == sort_key and endpoint != "failures":
                    raise FDICContractError("API sort key is repeated")
                previous_sort = sort_key
            yield page
            offset += len(page.rows)
            if offset == total:
                return
