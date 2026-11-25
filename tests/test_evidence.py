import csv
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_evidence_register_covers_plan():
    plan = (ROOT / "PROJECT_PLAN.md").read_text(encoding="utf-8")
    ids = set(re.findall(r"^\| ((?:P\d|R|S)\.\d+) \|", plan, re.M))
    with (ROOT / "EVIDENCE_REGISTER.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert ids == {row["rubric_id"] for row in rows}
    assert all(row["code_or_artifact"] and row["test_or_gate"] and row["report_output"] for row in rows)
