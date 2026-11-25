"""Provision and validate a relocated, immutable FDIC source vintage."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from bft65b.acquisition.extended import _entry, _verify_complete, full_plan
from bft65b.acquisition.full import full_paths
from bft65b.acquisition.storage import ParquetStore
from bft65b.config import load_config
from bft65b.manifest import RunManifest


BUNDLE_FILE = Path("runs/source_vintage_bundle.json")
ATTEMPT = "real20261111b"
SUPPLEMENTARY = (
    Path("runs/task_2_3_transaction_repair/source_failures_20261115.json"),
    Path("runs/task_2_3_transaction_repair/source_capture.json"),
    Path("runs/task_2_3_transaction_repair/failure_properties_20261115.yaml"),
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(root: Path, relative: str) -> Path:
    part = Path(relative)
    target = (root / part).resolve()
    if part.is_absolute() or ".." in part.parts or not target.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe source-vintage path: {relative}")
    return target


def inventory(root: Path) -> dict:
    """Validate every supplied file; never trust the bundle's own status alone."""
    path = root / BUNDLE_FILE
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("version") != "source_vintage_v1" or bundle.get("attempt") != ATTEMPT:
        raise ValueError("Source-vintage version or accepted attempt differs")
    files = bundle.get("files")
    if not isinstance(files, list) or len(files) != 180:
        raise ValueError("Source-vintage inventory must cover 172 raw and eight evidence files")
    seen = set()
    for item in files:
        relative = item.get("relative_path")
        if not isinstance(relative, str) or relative in seen:
            raise ValueError("Duplicate or missing source-vintage relative path")
        seen.add(relative)
        file = _safe_relative(root, relative)
        if not file.is_file() or file.stat().st_size != item.get("bytes") or _sha(file) != item.get("sha256"):
            raise ValueError(f"Source-vintage checksum/size differs: {relative}")
    expected_evidence = {
        "config/financials_full_v2.json",
        f"runs/task_1_5_full_{ATTEMPT}.json",
        f"runs/task_1_5_full_{ATTEMPT}_summary.json",
        f"runs/task_1_5_full_{ATTEMPT}_acceptance.json",
        f"runs/task_1_5_full_{ATTEMPT}_summary_unlinked_ids.json",
        *(path.as_posix() for path in SUPPLEMENTARY),
    }
    if not expected_evidence.issubset(seen):
        raise ValueError("Source-vintage evidence is incomplete")
    if sum(p.startswith("data_raw/") for p in seen) != 172 or len(seen) != 180:
        raise ValueError("Source-vintage raw-file coverage differs")
    return bundle


def provision(source: Path, destination: Path) -> dict:
    """Copy only verified source files into a new root, retaining original bytes."""
    source, destination = source.resolve(strict=True), destination.resolve(strict=True)
    if source == destination or destination.is_relative_to(source / "data_raw"):
        raise ValueError("Destination must be a separate workspace root")
    if (destination / BUNDLE_FILE).exists():
        raise FileExistsError("Source-vintage bundle already exists")
    config = load_config(source / "config/financials_full_v2.json")
    paths = full_paths(source, config, ATTEMPT)
    manifest = RunManifest.read(paths.manifest)
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    plan = full_plan(config)
    if manifest.status != "complete" or summary.get("status") != "complete":
        raise ValueError("Original source acquisition is incomplete")
    _verify_complete(plan, config, ParquetStore(paths.raw_root, config), manifest, summary)
    source_files = [source / relative for relative in (
        Path("config/financials_full_v2.json"),
        paths.manifest.relative_to(source), paths.summary.relative_to(source),
        paths.linkage_audit.relative_to(source),
        Path(f"runs/task_1_5_full_{ATTEMPT}_acceptance.json"),
        *SUPPLEMENTARY,
    )]
    source_files += [Path(item["path"]) for item in manifest.outputs]
    records = []
    for original in source_files:
        if not original.resolve().is_relative_to(source):
            raise ValueError("Original manifest references a file outside source root")
        relative = original.relative_to(source).as_posix()
        target = _safe_relative(destination, relative)
        if target.exists():
            raise FileExistsError(f"Destination source file already exists: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".source-vintage-", suffix=".tmp")
        os.close(fd)
        try:
            shutil.copyfile(original, temporary)
            digest = _sha(original)
            if _sha(Path(temporary)) != digest:
                raise ValueError(f"Copied source checksum differs: {relative}")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        records.append({"relative_path": relative, "original_path": str(original),
                        "sha256": digest, "bytes": target.stat().st_size})
    bundle = {"version": "source_vintage_v1", "attempt": ATTEMPT,
              "source_root_provenance": str(source), "files": sorted(records, key=lambda row: row["relative_path"]),
              "source_manifest_sha256": _sha(paths.manifest),
              "source_summary_sha256": _sha(paths.summary),
              "mode": "captured_snapshot_no_network"}
    target = destination / BUNDLE_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(bundle, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    inventory(destination)
    return {"bundle": str(target), "files": len(records), "raw_files": 172,
            "bundle_sha256": _sha(target)}


def relocated_view(root: Path, manifest: RunManifest, summary: dict, store: ParquetStore) -> tuple[RunManifest, dict]:
    """Rebase paths in memory after verifying the byte-identical original evidence."""
    bundle = inventory(root)
    if (_sha(root / f"runs/task_1_5_full_{ATTEMPT}.json") != bundle["source_manifest_sha256"]
            or _sha(root / f"runs/task_1_5_full_{ATTEMPT}_summary.json") != bundle["source_summary_sha256"]):
        raise ValueError("Original source manifest or summary checksum differs")
    mapped = {item["original_path"]: _safe_relative(root, item["relative_path"])
              for item in bundle["files"]}
    result_manifest, result_summary = copy.deepcopy(manifest), copy.deepcopy(summary)
    jobs = full_plan(store.config).jobs(store.config)
    for endpoint, quarter, _, _ in jobs:
        target = store.target_path(endpoint, quarter)
        outputs = [item for item in result_manifest.outputs
                   if item.get("dataset") == endpoint and
                   item.get("partition") == (f"REPDTE={quarter}" if quarter else None)]
        if len(outputs) != 1 or mapped.get(outputs[0]["path"]) != target:
            raise ValueError(f"Source-vintage output mapping differs: {endpoint}:{quarter}")
        outputs[0]["path"] = str(target)
    failure = _entry(result_summary, "failures", None)
    linkage = failure.get("unresolved_linkage", {}) if failure else {}
    old = linkage.get("source_ids_path")
    if not isinstance(old, str) or old not in mapped:
        raise ValueError("Source-vintage linkage-audit mapping differs")
    linkage["source_ids_path"] = str(mapped[old])
    return result_manifest, result_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Provision an immutable captured FDIC source vintage")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("provision")
    prep.add_argument("--source-root", type=Path, required=True)
    prep.add_argument("--destination-root", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = provision(args.source_root, args.destination_root) if args.command == "provision" else inventory(args.project_root)
    print(json.dumps(result if args.command == "provision" else
                     {"status": "complete", "files": len(result["files"]), "raw_files": 172}, indent=2))


if __name__ == "__main__":
    main()
