"""Small, manifest-backed live API probes; never download complete tables."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import islice
import json
from pathlib import Path

from bft65b.acquisition.client import BASE_URL, FDICClient
from bft65b.config import load_config
from bft65b.manifest import RunManifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-size", type=int, default=2, choices=range(1, 11))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("runs/task_1_1_api_probe.json"),
        help="Ignored JSON run-manifest path",
    )
    args = parser.parse_args(argv)
    config = load_config()
    client = FDICClient(config)
    manifest = RunManifest.new(
        "task-1.1-probe-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        config,
    )
    manifest.set_status("running")
    summary: dict[str, dict] = {}
    for endpoint in ("financials", "failures", "institutions"):
        filters = "REPDTE:20230331" if endpoint == "financials" else None
        try:
            # islice stops after two pages; it never materializes an endpoint.
            pages = list(
                islice(
                    client.iter_pages(
                        endpoint,
                        filters=filters,
                        limit=args.page_size,
                        require_nonempty=True,
                    ),
                    2,
                )
            )
            first = pages[0]
            missing = Counter(
                field for page in pages for row_fields in page.missing_fields for field in row_fields
            )
            nulls = Counter(
                field for page in pages for row_fields in page.null_fields for field in row_fields
            )
            summary[endpoint] = {
                "api_total": first.total,
                "sample_pages": len(pages),
                "sample_rows": sum(len(page.rows) for page in pages),
                "missing_requested_fields": dict(sorted(missing.items())),
                "null_requested_fields": dict(sorted(nulls.items())),
                "index": first.index,
                "filter": filters,
                "selected_fields": list(
                    {
                        "financials": config.financial_fields,
                        "failures": config.failure_fields,
                        "institutions": config.institution_fields,
                    }[endpoint]
                ),
            }
            manifest.record_input(
                f"{BASE_URL}/{endpoint}",
                rows=summary[endpoint]["sample_rows"],
                source_total=first.total,
            )
            manifest.record_validation(
                f"{endpoint}_two_page_contract",
                True,
                f"filter={filters!r}; pages={len(pages)}; sample_rows={summary[endpoint]['sample_rows']}; "
                f"reported_total={first.total}; index={first.index!r}; "
                f"missing={dict(missing)!r}; null={dict(nulls)!r}",
            )
        except Exception as exc:
            manifest.record_validation(f"{endpoint}_two_page_contract", False, repr(exc))
            manifest.set_status("failed")
            manifest.write(args.manifest)
            raise
    manifest.set_status("complete")
    manifest.write(args.manifest)
    print(json.dumps(summary, indent=2))
    print(f"Manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
