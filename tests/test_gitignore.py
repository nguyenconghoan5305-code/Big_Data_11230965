from pathlib import Path
import shutil
import subprocess


def test_data_and_environment_ignored(tmp_path):
    source = Path(__file__).resolve().parents[1] / ".gitignore"
    shutil.copyfile(source, tmp_path / ".gitignore")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    paths = ["data_raw/fdic/a.parquet", "data_processed/panel.parquet", "runs/run.json", ".venv/Scripts/python.exe"]
    result = subprocess.run(["git", "-C", str(tmp_path), "check-ignore", *paths], capture_output=True, text=True, check=True)
    assert set(result.stdout.splitlines()) == set(paths)
    visible = subprocess.run(["git", "-C", str(tmp_path), "check-ignore", "EVIDENCE_REGISTER.csv"], capture_output=True, text=True)
    assert visible.returncode == 1
