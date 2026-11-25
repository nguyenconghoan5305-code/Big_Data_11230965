import pytest
from bft65b import manifest as manifest_module

from bft65b.config import ProjectConfig
from bft65b.manifest import RunManifest


def test_manifest_round_trip(tmp_path):
    manifest = RunManifest.new("synthetic-run", ProjectConfig())
    manifest.record_input("synthetic.parquet", sha256="abc", rows=2, source_total=2)
    manifest.record_validation("synthetic-check", True)
    manifest.set_status("complete")
    target = tmp_path / "run.json"
    manifest.write(target)
    loaded = RunManifest.read(target)
    assert loaded.status == "complete"
    assert loaded.inputs[0]["rows"] == 2
    assert loaded.config["seed"] == 65
    assert loaded.dependencies["pandas"] is None or isinstance(loaded.dependencies["pandas"], str)


def test_manifest_cannot_complete_failed_or_missing_gate():
    manifest = RunManifest.new("synthetic-run", ProjectConfig())
    with pytest.raises(ValueError, match="passing validations"):
        manifest.set_status("complete")
    manifest.record_validation("failed-gate", False)
    with pytest.raises(ValueError, match="passing validations"):
        manifest.set_status("complete")


def test_windows_access_denied_replace_retries_then_promotes(tmp_path, monkeypatch):
    if manifest_module.os.name != "nt":
        pytest.skip("Windows WinError 5 behavior")
    manifest = RunManifest.new("synthetic-run", ProjectConfig())
    target = tmp_path / "run.json"
    original_replace = manifest_module.os.replace
    calls = []
    def transient(src, dst):
        calls.append((src, dst))
        if len(calls) == 1:
            raise PermissionError(5, "Access is denied", str(src), 5)
        return original_replace(src, dst)
    monkeypatch.setattr(manifest_module.os, "replace", transient)
    monkeypatch.setattr(manifest_module.time, "sleep", lambda _: None)
    manifest.write(target)
    assert len(calls) == 2
    assert RunManifest.read(target).run_id == "synthetic-run"
    assert not list(tmp_path.glob(".*.tmp"))


def test_persistent_windows_access_denied_preserves_existing_manifest(tmp_path, monkeypatch):
    if manifest_module.os.name != "nt":
        pytest.skip("Windows WinError 5 behavior")
    manifest = RunManifest.new("synthetic-run", ProjectConfig())
    target = tmp_path / "run.json"
    manifest.write(target)
    original = target.read_bytes()
    calls = []
    def denied(src, dst):
        calls.append((src, dst))
        raise PermissionError(5, "Access is denied", str(src), 5)
    monkeypatch.setattr(manifest_module.os, "replace", denied)
    monkeypatch.setattr(manifest_module.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError, match="Access is denied"):
        manifest.write(target)
    assert len(calls) == 5
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))
