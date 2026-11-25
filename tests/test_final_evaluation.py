"""Synthetic acceptance checks for the one-use final-evaluation workflow."""
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bft65b.features.preprocessing import PREDICTOR_COLUMNS
from bft65b.modelling import final_evaluation as final
from bft65b.modelling.development import CANDIDATES
from bft65b.quality.audit import _sha


def _frame(quarter: date, labels: list[int], first_cert: int = 1) -> pd.DataFrame:
    rows = []
    for index, y in enumerate(labels):
        row = {name: float(index + position + 1) for position, name in enumerate(PREDICTOR_COLUMNS)}
        row.update(CERT=first_cert + index, Q=quarter, Y=y)
        rows.append(row)
    return pd.DataFrame(rows)


def test_threshold_equality_keys_and_cost_arithmetic():
    q = date(2014, 3, 31)
    frame = _frame(q, [1, 0, 0])
    class Fixed:
        def predict_proba(self, values):
            return np.column_stack((1 - np.array([0.5, 0.5, 0.1]), [0.5, 0.5, 0.1]))
    table = final._scores(frame, Fixed(), 0.5)
    expected = {(row.CERT, row.Q): row.Y for row in frame.itertuples()}
    final._validate_scores(table, expected, 0.5)
    result = final._metrics(table, {(1, q): "event-1"}, {
        "full_development_representative_cost_nominal_thousands_usd": 100.0,
        "reference": {"alpha": 0.1, "beta": 0.01}})
    assert (result["TP"], result["FP"], result["TN"], result["FN"]) == (1, 1, 1, 0)
    assert result["detected_events"] == 1
    assert result["false_alarm_surrogate_nominal_thousands_usd"] == 1
    assert result["reference_surrogate_per_row_nominal_thousands_usd"] == pytest.approx(1 / 3)
    bad = table.to_pylist()
    bad[1]["CERT"] = 1
    with pytest.raises(ValueError, match="keys"):
        final._validate_scores(final.pa.Table.from_pylist(bad, schema=final.SCORE_SCHEMA), expected, 0.5)
    bad = table.to_pylist()
    bad[0]["Y"] = 0
    with pytest.raises(ValueError, match="keys or labels"):
        final._validate_scores(final.pa.Table.from_pylist(bad, schema=final.SCORE_SCHEMA), expected, 0.5)
    bad = table.to_pylist()
    bad[0]["ALARM"] = False
    with pytest.raises(ValueError, match="alarm"):
        final._validate_scores(final.pa.Table.from_pylist(bad, schema=final.SCORE_SCHEMA), expected, 0.5)


def test_training_only_fit_and_forbidden_columns():
    train = _frame(date(2012, 12, 31), [0, 1] * 20)
    held_out = _frame(date(2014, 3, 31), [0, 1], first_cert=100)
    from bft65b.modelling.development import candidate_pipeline
    first = candidate_pipeline(CANDIDATES[1])
    second = candidate_pipeline(CANDIDATES[1])
    first.fit(train[list(PREDICTOR_COLUMNS)], train["Y"])
    altered = held_out.copy()
    altered[PREDICTOR_COLUMNS[0]] = 1e12
    altered["Y"] = 1 - altered["Y"]
    second.fit(train[list(PREDICTOR_COLUMNS)], train["Y"])
    assert np.array_equal(first.named_steps["preprocessing"].named_steps["imputer"].statistics_,
                          second.named_steps["preprocessing"].named_steps["imputer"].statistics_)
    assert np.array_equal(first.predict_proba(train[list(PREDICTOR_COLUMNS)]),
                          second.predict_proba(train[list(PREDICTOR_COLUMNS)]))
    assert list(PREDICTOR_COLUMNS) == list(first.named_steps["preprocessing"].get_feature_names_out())
    assert not {"CERT", "Q", "Y", "FAILDATE", "SOURCE_ID"} & set(PREDICTOR_COLUMNS)
    combined = pd.concat([train, altered], ignore_index=True)
    leaking = candidate_pipeline(CANDIDATES[1]).fit(combined[list(PREDICTOR_COLUMNS)], combined["Y"])
    assert not np.array_equal(first.named_steps["preprocessing"].named_steps["scaler"].mean_,
                              leaking.named_steps["preprocessing"].named_steps["scaler"].mean_)


def test_project_registry_lock_and_single_run_id(tmp_path):
    registry = tmp_path / "test_use_registry.json"
    lock = tmp_path / ".registry.lock"
    with final._ownership(lock):
        with pytest.raises(RuntimeError, match="owns"):
            with final._ownership(lock):
                pass
        final._transition(registry, "design", "first", "planned", plan_sha256="a")
    with final._ownership(lock):
        final._transition(registry, "design", "first", "test_access")
    with pytest.raises(RuntimeError, match="new run ID"):
        final._transition(registry, "design", "second", "planned")
    assert [row["state"] for row in final._registry(registry)["designs"]["design"]["history"]] == ["planned", "test_access"]


def test_atomic_scores_and_corruption(tmp_path):
    q = date(2014, 3, 31)
    class Fixed:
        def predict_proba(self, values):
            return np.column_stack(([0.1, 0.9], [0.9, 0.1]))
    table = final._scores(_frame(q, [1, 0]), Fixed(), 0.5)
    path, binding = tmp_path / "scores.parquet", tmp_path / "binding.json"
    digest = final._write_scores(path, table, binding, "model", "plan", final.time.perf_counter())
    assert digest == _sha(path) == json.loads(binding.read_text())["score_sha256"]
    assert pq.read_table(path).equals(table)
    with pytest.raises(FileExistsError):
        final._write_scores(path, table, binding, "model", "plan", final.time.perf_counter())
    path.write_bytes(b"truncated")
    with pytest.raises(Exception):
        pq.read_table(path)


def test_test_partition_membership_and_label_alignment(tmp_path):
    root = Path(tmp_path)
    q = date(2014, 3, 31)
    frame = _frame(q, [1, 0], first_cert=101)
    path = root / "data_processed/task_4_3_feature_table/featuretable20261120a/model_table/Q=20140331/part-000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    summary_path = root / "runs/task_4_3_feature_table/featuretable20261120a/summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps({"outputs": [{"dataset": "model_table", "quarter": "20140331",
                                                "path": str(path)}]}), encoding="utf-8")
    split_path = root / "runs/task_5_1_split/split20261120b/summary.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps({"periods": {"test": {"prediction_quarters": 1}}}), encoding="utf-8")
    spec = {"bounds": {"test_start": "2014-03-31", "test_end": "2014-03-31"},
            "source_hashes": {str(path): _sha(path)}}
    expected = {(101, q): 1, (102, q): 0}
    assert len(final._test_frame(root, expected, spec)) == 2
    with pytest.raises(ValueError, match="labels"):
        final._test_frame(root, {(101, q): 0, (102, q): 1}, spec)
    duplicate = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    pq.write_table(pa.Table.from_pandas(duplicate, preserve_index=False), path)
    spec["source_hashes"][str(path)] = _sha(path)
    with pytest.raises(ValueError, match="keys"):
        final._test_frame(root, expected, spec)


def test_invalid_probability_and_threshold_rejected():
    class Invalid:
        def predict_proba(self, values):
            return np.array([[np.nan, np.nan]])
    frame = _frame(date(2014, 3, 31), [1])
    with pytest.raises(ValueError, match="invalid"):
        final._scores(frame, Invalid(), 0.5)
    with pytest.raises(ValueError, match="invalid"):
        final._scores(frame, Invalid(), float("nan"))


def test_synthetic_plan_fit_evaluate_and_score_only_recovery(tmp_path, monkeypatch):
    root = Path(tmp_path)
    (root / "final_report_bft65b.pdf").write_bytes(b"synthetic authority marker")
    q_dev, q_test = date(2012, 12, 31), date(2014, 3, 31)
    train = _frame(q_dev, [0, 1] * 20)
    test = _frame(q_test, [1, 0, 0, 0], first_cert=101)
    expected = {(int(row.CERT), row.Q): int(row.Y) for row in test.itertuples()}
    event = {(101, q_test): "event-101"}
    lock_path = root / "runs/task_5_5_threshold/threshold20261122c/locked_policy.json"
    lock_path.parent.mkdir(parents=True)
    locked = {"configuration": CANDIDATES[1], "policy": {"kind": "score_ge", "threshold": 0.2,
        "score_comparison": "SCORE >= threshold"}, "reference": {"alpha": 0.1, "beta": 0.01},
        "full_development_representative_cost_nominal_thousands_usd": 100.0}
    lock_path.write_text(json.dumps(locked), encoding="utf-8")
    dev_path = root / "runs/task_5_2_development/dev20261121a/frozen_plan.json"
    dev_path.parent.mkdir(parents=True)
    dev_path.write_text("{}", encoding="utf-8")
    class Config:
        def to_dict(self):
            return {}
    monkeypatch.setattr(final, "load_config", lambda _: Config())
    split = {"bounds": {"development_start": "1984-03-31", "development_end": "2012-12-31",
                        "test_start": "2014-03-31", "test_end": "2025-06-30"},
             "periods": {"development": {"model_rows": 40, "positive_rows": 20}}}
    split_path = root / "runs/task_5_1_split/split20261120b/summary.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps(split), encoding="utf-8")
    accepted = {"design": "design", "lock": locked, "split": split,
                "sources": {str(lock_path): _sha(lock_path), str(dev_path): _sha(dev_path)}}
    monkeypatch.setattr(final, "_accepted", lambda _: accepted)
    monkeypatch.setattr(final, "_development_frame", lambda *_: (train, {(int(r.CERT), r.Q) for r in train.itertuples()}))
    monkeypatch.setattr(final, "_expected_test", lambda _: (expected, event))
    monkeypatch.setattr(final, "_test_frame", lambda *_: test)
    assert final.plan(root, "synthetic") ["status"] == "planned"
    assert final.fit(root, "synthetic")["rows"] == 40
    assert final.evaluate(root, "synthetic")["status"] == "complete"
    score_path = root / "data_processed/task_5_6_final/synthetic/test_scores.parquet"
    original = _sha(score_path)
    monkeypatch.setattr(final, "_scores", lambda *_: (_ for _ in ()).throw(AssertionError("rescore")))
    assert final.evaluate(root, "synthetic")["status"] == "complete"
    assert _sha(score_path) == original
    with pytest.raises(FileExistsError):
        final.plan(root, "another")
    evidence, _, registry, _ = final._paths(root, "synthetic")
    saved_registry = final._registry(registry)
    saved_registry["designs"]["design"]["state"] = "test_access"
    final._write_json_atomic(registry, saved_registry)
    (evidence / "summary.json").unlink()
    (evidence / "metadata.json").unlink()
    assert final.evaluate(root, "synthetic")["status"] == "complete"
    score_path.unlink()
    saved_registry = final._registry(registry)
    saved_registry["designs"]["design"]["state"] = "test_access"
    final._write_json_atomic(registry, saved_registry)
    with pytest.raises(RuntimeError, match="automatic rescore is forbidden"):
        final.evaluate(root, "synthetic")


@pytest.fixture
def completed_integrity_run(tmp_path, monkeypatch):
    """A complete synthetic run using the real plan/fit/evaluate/verifier path."""
    root = Path(tmp_path)
    (root / "final_report_bft65b.pdf").write_bytes(b"synthetic authority marker")
    q_dev, q_test = date(2012, 12, 31), date(2014, 3, 31)
    train = _frame(q_dev, [0, 1] * 20)
    test = _frame(q_test, [1, 0, 0, 0], first_cert=101)
    expected = {(int(row.CERT), row.Q): int(row.Y) for row in test.itertuples()}
    event = {(101, q_test): "event-101"}
    lock_path = root / "runs/task_5_5_threshold/threshold20261122c/locked_policy.json"
    lock_path.parent.mkdir(parents=True)
    locked = {"configuration": CANDIDATES[1], "policy": {"kind": "score_ge", "threshold": 0.2,
        "score_comparison": "SCORE >= threshold"}, "reference": {"alpha": 0.1, "beta": 0.01},
        "full_development_representative_cost_nominal_thousands_usd": 100.0}
    lock_path.write_text(json.dumps(locked), encoding="utf-8")
    dev_path = root / "runs/task_5_2_development/dev20261121a/frozen_plan.json"
    dev_path.parent.mkdir(parents=True)
    dev_path.write_text("{}", encoding="utf-8")
    split = {"bounds": {"development_start": "1984-03-31", "development_end": "2012-12-31",
                        "test_start": "2014-03-31", "test_end": "2025-06-30"},
             "periods": {"development": {"model_rows": 40, "positive_rows": 20}}}
    split_path = root / "runs/task_5_1_split/split20261120b/summary.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps(split), encoding="utf-8")
    class Config:
        def to_dict(self):
            return {}
    monkeypatch.setattr(final, "load_config", lambda _: Config())
    accepted = {"design": "design", "lock": locked, "split": split,
                "sources": {str(lock_path): _sha(lock_path), str(dev_path): _sha(dev_path)}}
    monkeypatch.setattr(final, "_accepted", lambda _: accepted)
    monkeypatch.setattr(final, "_development_frame", lambda *_: (train, {(int(r.CERT), r.Q) for r in train.itertuples()}))
    monkeypatch.setattr(final, "_expected_test", lambda _: (expected, event))
    monkeypatch.setattr(final, "_test_frame", lambda *_: test)
    assert final.plan(root, "integrity")["status"] == "planned"
    assert final.fit(root, "integrity")["status"] == "model_fitted"
    assert final.evaluate(root, "integrity")["status"] == "complete"
    assert final.verify_saved(root, "integrity")["status"] == "complete"
    return root


def test_verifier_rejects_changed_model_bytes_with_unchanged_records(completed_integrity_run):
    root = completed_integrity_run
    model_path = root / "data_processed/task_5_6_final/integrity/fitted_pipeline.joblib"
    original = _sha(model_path)
    with model_path.open("ab") as stream:
        stream.write(b"synthetic mutation after fit")
    assert _sha(model_path) != original
    with pytest.raises(ValueError, match="model|artifact|hash"):
        final.verify_saved(root, "integrity")


def test_verifier_rejects_recorded_binding_registry_and_manifest_conflicts(completed_integrity_run):
    root = completed_integrity_run
    evidence = root / "runs/task_5_6_final/integrity"
    registry_path = root / "runs/task_5_6_final/test_use_registry.json"
    summary_path = evidence / "summary.json"
    metadata_path = evidence / "metadata.json"
    binding_path = evidence / "model_binding.json"
    score_binding_path = evidence / "score_binding.json"
    manifest_path = evidence / "run_manifest.json"
    originals = {path: path.read_bytes() for path in
        (registry_path, summary_path, metadata_path, binding_path, score_binding_path, manifest_path)}

    def restore():
        for path, content in originals.items():
            path.write_bytes(content)
        assert final.verify_saved(root, "integrity")["status"] == "complete"

    # Keep summary metadata and registry digest consistent; isolate the
    # disagreement between its model reference and the accepted binding.
    summary = json.loads(originals[summary_path])
    summary["model_sha256"] = "0" * 64
    summary_digest = final._write_json_atomic(summary_path, summary)
    metadata = json.loads(originals[metadata_path])
    metadata["summary_sha256"] = summary_digest
    final._write_json_atomic(metadata_path, metadata)
    registry = json.loads(originals[registry_path])
    registry["designs"]["design"]["summary_sha256"] = summary_digest
    final._write_json_atomic(registry_path, registry)
    with pytest.raises(ValueError, match="manifest|artifact hash"):
        final.verify_saved(root, "integrity")
    restore()

    # Keep the registry's binding checksum current so the recorded model
    # disagreement, not a stale JSON-file checksum, is the failing gate.
    binding = json.loads(originals[binding_path])
    binding["model_sha256"] = "0" * 64
    binding_digest = final._write_json_atomic(binding_path, binding)
    registry = json.loads(originals[registry_path])
    registry["designs"]["design"]["binding_sha256"] = binding_digest
    final._write_json_atomic(registry_path, registry)
    with pytest.raises(ValueError, match="manifest|artifact hash"):
        final.verify_saved(root, "integrity")
    restore()

    score_binding = json.loads(originals[score_binding_path])
    score_binding["plan_sha256"] = "0" * 64
    final._write_json_atomic(score_binding_path, score_binding)
    with pytest.raises(ValueError, match="manifest|artifact hash"):
        final.verify_saved(root, "integrity")
    restore()

    for field in ("model_sha256", "binding_sha256", "score_sha256"):
        registry = json.loads(originals[registry_path])
        registry["designs"]["design"][field] = "0" * 64
        final._write_json_atomic(registry_path, registry)
        with pytest.raises(ValueError, match="manifest|artifact hash"):
            final.verify_saved(root, "integrity")
        restore()

    manifest = json.loads(originals[manifest_path])
    next(output for output in manifest["outputs"] if output["dataset"] == "final_model")["sha256"] = "0" * 64
    final._write_json_atomic(manifest_path, manifest)
    with pytest.raises(ValueError, match="manifest|artifact hash"):
        final.verify_saved(root, "integrity")
    restore()
