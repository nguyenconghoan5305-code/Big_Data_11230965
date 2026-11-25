"""Atomic JSON run manifests with explicit stage status and validation results."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from .config import ProjectConfig

DEPENDENCIES = ("numpy", "pandas", "polars", "pyarrow", "pyspark", "scikit-learn", "requests", "pytest")
STATUSES = frozenset({"pending", "running", "complete", "failed"})


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@dataclass
class RunManifest:
    run_id: str
    config: dict[str, Any]
    created_at_utc: str = field(default_factory=_timestamp)
    updated_at_utc: str = field(default_factory=_timestamp)
    status: str = "pending"
    python_version: str = field(default_factory=platform.python_version)
    dependencies: dict[str, str | None] = field(default_factory=dependency_versions)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    validations: list[dict[str, Any]] = field(default_factory=list)
    stages: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id or self.status not in STATUSES or any(v not in STATUSES for v in self.stages.values()):
            raise ValueError("Invalid run ID or status")

    @classmethod
    def new(cls, run_id: str, config: ProjectConfig) -> "RunManifest":
        return cls(run_id=run_id, config=config.to_dict())

    def record_input(self, path: str, sha256: str | None = None, rows: int | None = None, source_total: int | None = None) -> None:
        self.inputs.append({"path": path, "sha256": sha256, "rows": rows, "source_total": source_total})
        self.updated_at_utc = _timestamp()

    def record_source(self, *, endpoint: str, query: str, total: int,
                      downloaded: int, pages: int, index: Any,
                      started_at_utc: str, finished_at_utc: str,
                      missing_fields: dict[str, int], null_fields: dict[str, int]) -> None:
        """Record API provenance and field presence without retaining source rows."""
        self.sources.append({
            "endpoint": endpoint, "query": query, "api_total": total,
            "downloaded_rows": downloaded, "pages": pages, "index": index,
            "started_at_utc": started_at_utc, "finished_at_utc": finished_at_utc,
            "missing_fields": missing_fields, "null_fields": null_fields,
        })
        self.updated_at_utc = _timestamp()

    def record_output(
        self, *, path: str, dataset: str, rows: int, sha256: str,
        compression: str, schema_version: str, partition: str | None,
        source_total: int, source_query: str | None,
    ) -> None:
        """Append a complete verified artifact; retain prior records for audit."""
        self.outputs.append({
            "path": path, "dataset": dataset, "rows": rows, "sha256": sha256,
            "compression": compression, "schema_version": schema_version,
            "partition": partition, "source_total": source_total,
            "source_query": source_query, "status": "complete", "at_utc": _timestamp(),
        })
        self.stages[f"storage:{dataset}:{partition or 'all'}"] = "complete"
        if self.status == "pending":
            self.status = "running"
        self.updated_at_utc = _timestamp()
    def record_validation(self, name: str, passed: bool, detail: str = "") -> None:
        self.validations.append({"name": name, "passed": passed, "detail": detail, "at_utc": _timestamp()})
        self.updated_at_utc = _timestamp()

    def set_status(self, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"Unknown status: {status}")
        if status == "complete" and (not self.validations or not all(item["passed"] for item in self.validations)):
            raise ValueError("Cannot complete a run without passing validations")
        self.status = status
        self.updated_at_utc = _timestamp()

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(asdict(self), handle, indent=2, sort_keys=True)
                handle.write("\n")
            for attempt in range(5):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError as exc:
                    # Some Windows file filters briefly deny an otherwise valid
                    # same-directory replacement. Persistent denial still fails.
                    if os.name != "nt" or exc.winerror != 5 or attempt == 4:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def read(cls, path: Path) -> "RunManifest":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
