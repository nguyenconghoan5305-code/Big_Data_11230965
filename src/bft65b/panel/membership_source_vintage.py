"""Provision captured FDIC membership responses without rewriting prior bundles."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

from bft65b.manifest import RunManifest
from bft65b.quality.audit import _sha, _read_json
from bft65b.quality.source_vintage import _safe_relative


BUNDLE = Path("runs/membership_source_bundle_v1.json")
SOURCE_RUN = "activeaudit20261119a"
SOURCE_DIR = Path("runs/task_4_2_active_membership") / SOURCE_RUN
PROBE = Path("runs/task_4_2_membership_code_probe/codeprobe20261119a.json")
PROBE_SHA256 = "cec7f7d7792bd81b509fcd8a60c5c08d1103c42e94bf89395e3aff265b846268"


def inventory(root: Path) -> dict:
    """Validate the independent 49-file membership source bundle in place."""
    root = root.resolve(strict=True)
    bundle = _read_json(root / BUNDLE)
    files = bundle.get("files")
    if (bundle.get("version") != "membership_source_v1"
            or bundle.get("source_run") != SOURCE_RUN
            or not isinstance(files, list) or len(files) != 49):
        raise ValueError("Membership source bundle scope differs")
    paths = [item["relative_path"] for item in files]
    if (len(paths) != len(set(paths)) or sum("/institutions/" in p for p in paths) != 3
            or sum("/history_fi/" in p for p in paths) != 42
            or PROBE.as_posix() not in paths
            or "final_report_bft65b.pdf" not in paths):
        raise ValueError("Membership response, code probe or assignment inventory differs")
    for item in files:
        path = _safe_relative(root, item["relative_path"])
        if (not path.is_file() or path.stat().st_size != item["bytes"]
                or _sha(path) != item["sha256"]):
            raise ValueError(f"Membership source checksum differs: {item['relative_path']}")
    if _sha(root / PROBE) != PROBE_SHA256:
        raise ValueError("Captured official History code probe changed")
    return bundle


def provision(source: Path, destination: Path) -> dict:
    """Copy source responses and their original evidence into a separate root."""
    source, destination = source.resolve(strict=True), destination.resolve(strict=True)
    if source == destination or (destination / BUNDLE).exists():
        raise FileExistsError("Membership source bundle already exists")
    summary_path = source / SOURCE_DIR / "summary.json"
    manifest_path = source / SOURCE_DIR / "run_manifest.json"
    summary = _read_json(summary_path)
    manifest = RunManifest.read(manifest_path)
    pages = summary.get("institutions_pages", []) + summary.get("history_years", [])
    if (summary.get("status") != "audit_complete" or summary.get("run_id") != SOURCE_RUN
            or manifest.status != "complete" or len(pages) != 45
            or len(manifest.inputs) != 45
            or {item["path"]: item["sha256"] for item in manifest.inputs}
               != {item["path"]: item["sha256"] for item in pages}):
        raise ValueError("Captured membership source manifest or page inventory differs")
    originals = [summary_path, manifest_path, source / PROBE,
                 source / "final_report_bft65b.pdf"] + [Path(item["path"]) for item in pages]
    records = []
    for original in originals:
        original = original.resolve(strict=True)
        if not original.is_relative_to(source):
            raise ValueError("Captured source path escapes project root")
        relative = original.relative_to(source).as_posix()
        target = _safe_relative(destination, relative)
        if target.exists():
            raise FileExistsError(f"Portable membership file already exists: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=".membership-source-", suffix=".tmp")
        os.close(descriptor)
        try:
            shutil.copyfile(original, temporary)
            digest = _sha(original)
            if _sha(Path(temporary)) != digest:
                raise ValueError(f"Copied membership source differs: {relative}")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        records.append({"relative_path": relative, "original_path": str(original),
                        "sha256": digest, "bytes": target.stat().st_size})
    bundle = {"version": "membership_source_v1", "source_run": SOURCE_RUN,
              "mode": "captured_snapshot_no_network",
              "source_root_provenance": str(source),
              "source_summary_sha256": _sha(summary_path),
              "source_manifest_sha256": _sha(manifest_path),
              "files": sorted(records, key=lambda item: item["relative_path"])}
    path = destination / BUNDLE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    inventory(destination)
    return {"status": "complete", "bundle": str(path), "files": len(records),
            "source_pages": len(pages), "bundle_sha256": _sha(path)}


def relocated_capture(root: Path) -> tuple[dict, RunManifest, str]:
    """Map captured page paths in memory after validating original bytes."""
    root = root.resolve(strict=True)
    bundle = inventory(root)
    summary_path, manifest_path = root / SOURCE_DIR / "summary.json", root / SOURCE_DIR / "run_manifest.json"
    if (_sha(summary_path) != bundle["source_summary_sha256"]
            or _sha(manifest_path) != bundle["source_manifest_sha256"]):
        raise ValueError("Original membership capture evidence changed")
    summary, manifest = copy.deepcopy(_read_json(summary_path)), copy.deepcopy(RunManifest.read(manifest_path))
    mapped = {item["original_path"]: _safe_relative(root, item["relative_path"])
              for item in bundle["files"]}
    for group in ("institutions_pages", "history_years"):
        for item in summary[group]:
            if item["path"] not in mapped:
                raise ValueError("Captured membership page lacks a portable mapping")
            item["path"] = str(mapped[item["path"]])
    for item in manifest.inputs:
        if item["path"] not in mapped:
            raise ValueError("Captured membership manifest input lacks a portable mapping")
        item["path"] = str(mapped[item["path"]])
    return summary, manifest, _sha(root / BUNDLE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("provision")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--destination-root", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    result = provision(args.source_root, args.destination_root) if args.mode == "provision" else inventory(args.project_root)
    print(json.dumps(result if args.mode == "provision" else
                     {"status": "complete", "files": len(result["files"]),
                      "source_pages": 45, "bundle_sha256": _sha(args.project_root / BUNDLE)}, indent=2))


if __name__ == "__main__":
    main()
