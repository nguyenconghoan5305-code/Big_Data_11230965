"""Synthetic contract tests for the first Part 5 chronological split gate."""
from datetime import date
import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bft65b.features.label import OutcomeEvent
from bft65b.features.production import LABEL_SCHEMA
from bft65b.features.table import MODEL_SCHEMA
from bft65b.modelling.split import (
    _quarter_records, choose_periods, period_for, quarter_index, validate_positive,
)
from bft65b.modelling import split


def quarters(start=2020, end=2025):
    return [date(y, m, {3: 31, 6: 30, 9: 30, 12: 31}[m])
            for y in range(start, end + 1) for m in (3, 6, 9, 12)
            if date(y, m, {3: 31, 6: 30, 9: 30, 12: 31}[m]) <= date(2025, 6, 30)]


def event(source_id="fdic-1", cert=1, when=date(2024, 3, 31)):
    return OutcomeEvent(source_id, cert, when, "verified_failure",
                        "chronological_candidate", "matched")


def test_exactly_four_intervening_quarters_and_inclusive_boundaries():
    qs = quarters()
    events = {qs[-1]: {f"id-{i}" for i in range(50)}}
    bounds = choose_periods(qs, events)
    assert bounds["test_start"] == date(2025, 6, 30)
    assert bounds["gap_start"] == date(2024, 6, 30)
    assert bounds["gap_end"] == date(2025, 3, 31)
    assert bounds["development_end"] == date(2024, 3, 31)
    assert bounds["development_target_right_end"] == date(2025, 3, 31)
    assert quarter_index(bounds["test_start"]) - quarter_index(bounds["development_end"]) - 1 == 4
    assert [period_for(q, bounds) for q in qs].count("gap") == 4
    assert period_for(bounds["development_end"], bounds) == "development"
    assert period_for(bounds["gap_start"], bounds) == "gap"
    assert period_for(bounds["test_start"], bounds) == "test"


def test_three_intervening_quarters_rejected():
    qs = quarters()
    with pytest.raises(ValueError, match="invalid"):
        choose_periods(qs, {qs[-1]: {f"id-{i}" for i in range(50)}}, gap_quarters=3)


def test_many_positive_rows_do_not_count_as_distinct_failures():
    qs = quarters()
    repeated = {f"id-{i}" for i in range(49)}
    # Hundreds of labelled rows can map to fewer than 50 source events.
    with pytest.raises(ValueError, match="required distinct test failures"):
        choose_periods(qs, {q: set(repeated) for q in qs})


def test_exactly_fifty_and_deterministic_recent_block():
    qs = quarters()
    events = {qs[-2]: {f"id-{i}" for i in range(49)}, qs[-1]: {"id-49"}}
    first = choose_periods(qs, events)
    assert first == choose_periods(list(qs), dict(events))
    assert first["test_start"] == qs[-2]
    assert first["test_distinct_events"] == 50


def test_every_synthetic_key_receives_one_period_without_loss():
    qs = quarters()
    bounds = choose_periods(qs, {qs[-1]: {f"id-{i}" for i in range(50)}})
    keys = [(cert, q) for q in qs for cert in (101, 202)]
    assignments = {(cert, q): period_for(q, bounds) for cert, q in keys}
    assert len(assignments) == len(keys)
    assert sum(value == "development" for value in assignments.values()) == 2 * 17
    assert sum(value == "gap" for value in assignments.values()) == 2 * 4
    assert sum(value == "test" for value in assignments.values()) == 2


def test_shared_source_event_between_development_and_test_rejected():
    qs = quarters()
    events = {qs[0]: {"shared"}, qs[-1]: {"shared", *[f"id-{i}" for i in range(49)]}}
    with pytest.raises(ValueError, match="both development and test"):
        choose_periods(qs, events)


def test_invalid_quarters_and_positive_linkage():
    with pytest.raises(ValueError, match="quarter-end"):
        quarter_index(date(2024, 1, 1))
    q = date(2023, 6, 30)
    known = {"fdic-1": event()}
    assert validate_positive(1, q, ["fdic-1"], known) == ("fdic-1", date(2024, 3, 31))
    for ids in ([], ["fdic-1", "fdic-1"], ["unknown"]):
        with pytest.raises(ValueError, match="source ID"):
            validate_positive(1, q, ids, known)
    for bad in (event(cert=2), event(when=q), event(when=date(2024, 9, 30)),
                OutcomeEvent("fdic-1", 1, date(2024, 3, 31),
                             "assistance_transaction", "chronological_candidate")):
        with pytest.raises(ValueError, match="target window"):
            validate_positive(1, q, ["fdic-1"], {"fdic-1": bad})
    assert validate_positive(1, q, ["fdic-1"],
        {"fdic-1": event(when=date(2024, 6, 30))})[1] == date(2024, 6, 30)


def _fixture(tmp_path, *, y=1, right=date(2024, 6, 30),
             label_disposition="positive", ids=None, model_cert=1,
             feature_history=True, membership="eligible"):
    q = date(2023, 6, 30)
    model_path = tmp_path / "model.parquet"
    label_path = tmp_path / "labels.parquet"
    model_row = {"CERT": model_cert, "Q": q, "Y": y,
                 **{name: None for name in MODEL_SCHEMA.names[3:]}}
    label_row = {"CERT": 1, "Q": q, "RIGHT_ENDPOINT": right,
                 "MEMBERSHIP": membership, "MEMBERSHIP_REASON": "synthetic",
                 "FEATURE_HISTORY_ELIGIBLE": feature_history,
                 "LABEL_DISPOSITION": label_disposition, "Y": y,
                 "LABEL_REASON": "synthetic", "QUALIFYING_SOURCE_IDS": ids if ids is not None else ["fdic-1"]}
    pq.write_table(pa.Table.from_pylist([model_row], schema=MODEL_SCHEMA), model_path)
    pq.write_table(pa.Table.from_pylist([label_row], schema=LABEL_SCHEMA), label_path)
    return ({"path": str(model_path), "quarter": "20230630", "rows": 1},
            {"path": str(label_path), "quarter": "20230630", "rows": 1})


def test_quarter_records_validate_keys_linkage_and_followup(tmp_path):
    model, labels = _fixture(tmp_path)
    records, links = _quarter_records(model, labels, {"fdic-1": event()})
    assert len(records) == len(links) == 1
    assert links[0]["SOURCE_ID"] == "fdic-1"


@pytest.mark.parametrize("change", [
    {"right": date(2024, 3, 31)},
    {"right": date(2026, 9, 30)},
    {"model_cert": 2},
    {"feature_history": False},
    {"membership": "uncertain"},
    {"label_disposition": "negative"},
])
def test_quarter_records_reject_inconsistent_model_label_or_followup(tmp_path, change):
    model, labels = _fixture(tmp_path, **change)
    with pytest.raises(ValueError, match="differs"):
        _quarter_records(model, labels, {"fdic-1": event()})


def test_negative_with_qualifying_id_rejected(tmp_path):
    model, labels = _fixture(tmp_path, y=0, label_disposition="negative")
    with pytest.raises(ValueError, match="Negative has qualifying"):
        _quarter_records(model, labels, {"fdic-1": event()})


def test_valid_negative_with_complete_followup(tmp_path):
    model, labels = _fixture(tmp_path, y=0, label_disposition="negative", ids=[])
    rows, links = _quarter_records(model, labels, {"fdic-1": event()})
    assert rows[0]["Y"] == 0 and links == []


def test_duplicate_model_key_and_duplicate_label_key_rejected(tmp_path):
    model, labels = _fixture(tmp_path)
    model_path = model["path"]
    original = pq.ParquetFile(model_path).read()
    pq.write_table(pa.concat_tables([original, original]), model_path)
    model["rows"] = 2
    with pytest.raises(ValueError, match="Duplicate"):
        _quarter_records(model, labels, {"fdic-1": event()})
    pq.write_table(original, model_path)
    model["rows"] = 1
    label_path = labels["path"]
    original_label = pq.ParquetFile(label_path).read()
    pq.write_table(pa.concat_tables([original_label, original_label]), label_path)
    labels["rows"] = 2
    with pytest.raises(ValueError, match="Duplicate label keys"):
        _quarter_records(model, labels, {"fdic-1": event()})


def test_prediction_after_followup_cutoff_rejected(tmp_path):
    model, labels = _fixture(tmp_path, y=0, label_disposition="negative", ids=[])
    late_q = date(2025, 9, 30)
    model_table = pq.ParquetFile(model["path"]).read().to_pylist()
    label_table = pq.ParquetFile(labels["path"]).read().to_pylist()
    model_table[0]["Q"] = late_q
    label_table[0]["Q"] = late_q
    label_table[0]["RIGHT_ENDPOINT"] = date(2026, 9, 30)
    model["quarter"] = labels["quarter"] = "20250930"
    pq.write_table(pa.Table.from_pylist(model_table, schema=MODEL_SCHEMA), model["path"])
    pq.write_table(pa.Table.from_pylist(label_table, schema=LABEL_SCHEMA), labels["path"])
    with pytest.raises(ValueError, match="follow-up differs"):
        _quarter_records(model, labels, {"fdic-1": event()})


def test_portable_split_rejects_historical_or_incomplete_selection(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/source_vintage_bundle.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="explicit nonhistorical"):
        split._accepted(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        split._accepted(tmp_path, {"labels": "portable-labels"})


def test_portable_split_rejects_cross_stage_lineage_before_output(tmp_path, monkeypatch):
    selected = {key: f"portable-{key}" for key in split.DEFAULT_RUNS}
    (tmp_path / "runs/source_vintage_bundle.json").parent.mkdir()
    (tmp_path / "runs/source_vintage_bundle.json").write_text("{}", encoding="utf-8")
    summaries = {
        ("task_4_3_feature_table", selected["model_table"]):
            {"source_runs": {"membership": selected["membership"], "labels": selected["labels"]}},
        ("task_4_3_feature_table", selected["model_repeat"]):
            {"source_runs": {"membership": selected["membership"], "labels": selected["labels"]}},
        ("task_4_2_production_labels", selected["labels"]):
            {"membership_run": selected["membership"], "repaired_event_run": "wrong-repair"},
        ("task_4_4_leakage_audit", selected["leakage_audit"]):
            {"source_runs": {"feature_table": selected["model_table"],
                             "independent_repeat": selected["model_repeat"],
                             "membership": selected["membership"], "labels": selected["labels"]}},
    }
    for (folder, run_id), payload in summaries.items():
        path = tmp_path / "runs" / folder / run_id / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(split, "verify_membership", lambda *args: None)
    monkeypatch.setattr(split, "verify_labels", lambda *args: None)
    monkeypatch.setattr(split, "verify_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(split, "verify_leakage", lambda *args, **kwargs: None)
    monkeypatch.setattr(split, "_verify_repair", lambda *args: ({}, [], {}))
    with pytest.raises(ValueError, match="lineage is inconsistent"):
        split._accepted(tmp_path, selected)


def test_split_cli_rejects_partial_explicit_selection(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["split", "run", "--project-root", str(tmp_path),
                                   "--run-id", "new-run", "--labels", "portable-labels"])
    with pytest.raises(SystemExit) as error:
        split.main()
    assert error.value.code == 2
    assert "All split upstream run IDs are required" in capsys.readouterr().err
