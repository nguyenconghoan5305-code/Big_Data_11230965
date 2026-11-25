"""P3.2 harness contracts independent of real-data timing."""

import os
import statistics
import subprocess
import sys

import pytest
import polars as pl

from bft65b.benchmarks import panel
from bft65b.benchmarks.panel import THREE_ENGINES, run_order, summarize
from bft65b.benchmarks.process_memory import sample_tree_rss


def test_rotation_has_five_fresh_process_slots_per_engine():
    order = run_order()
    assert len(order) == 10
    assert order == [(1, "pandas"), (1, "polars"), (2, "polars"), (2, "pandas"),
                     (3, "pandas"), (3, "polars"), (4, "polars"), (4, "pandas"),
                     (5, "pandas"), (5, "polars")]


def test_three_engine_rotation_has_fifteen_slots_without_engine_bias():
    order = run_order(THREE_ENGINES)
    assert len(order) == 15
    assert [engine for round_number, engine in order if round_number == 1] == list(THREE_ENGINES)
    assert [engine for round_number, engine in order if round_number == 2] == ["polars", "spark", "pandas"]
    assert [engine for round_number, engine in order if round_number == 3] == ["spark", "pandas", "polars"]
    assert {engine: sum(value == engine for _, value in order) for engine in THREE_ENGINES} == {
        "pandas": 5, "polars": 5, "spark": 5}


def test_summary_uses_all_five_durations_and_memory_samples():
    records = [
        {"engine": engine, "kind": "measured", "status": "pass",
         "duration_ns": duration, "peak_tree_rss_bytes": peak}
        for engine, durations, peaks in (
            ("pandas", (10, 11, 12, 13, 14), (100, 101, 102, 103, 104)),
            ("polars", (5, 6, 7, 8, 9), (50, 51, 52, 53, 54)))
        for duration, peak in zip(durations, peaks)]
    result = summarize(records)
    assert result["pandas"]["median_duration_ns"] == statistics.median((10, 11, 12, 13, 14))
    assert result["polars"]["speedup_vs_pandas_median"] == 12 / 7
    assert result["polars"]["memory_ratio_vs_pandas_median"] == 52 / 102
    assert result["polars"]["min_peak_tree_rss_bytes"] == 50
    assert result["polars"]["max_duration_ns"] == 9


def test_incomplete_or_failed_runs_cannot_be_summarized():
    records = [{"engine": "pandas", "kind": "measured", "status": "pass",
                "duration_ns": 1, "peak_tree_rss_bytes": 1}] * 4
    with pytest.raises(ValueError, match="Incomplete measured runs"):
        summarize(records)
    records.append({**records[0], "status": "failed"})
    with pytest.raises(ValueError, match="Incomplete measured runs"):
        summarize(records)


def test_process_tree_sampler_includes_current_process():
    rss, count, unreadable = sample_tree_rss(os.getpid())
    assert rss > 0
    assert count >= 1
    assert unreadable == 0


def test_process_tree_sampler_includes_descendant():
    script = ("import subprocess,sys; "
              "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(3)']); "
              "print(child.pid,flush=True); child.wait()")
    parent = subprocess.Popen([sys.executable, "-c", script],
                              stdout=subprocess.PIPE, text=True)
    try:
        child_pid = int(parent.stdout.readline().strip())
        assert child_pid != parent.pid
        rss, count, unreadable = sample_tree_rss(parent.pid)
        assert rss > 0
        assert count >= 2
        assert unreadable == 0
    finally:
        try:
            parent.wait(timeout=5)
        except subprocess.TimeoutExpired:
            parent.kill()
            parent.wait(timeout=5)


def test_parity_mismatch_names_first_field_and_row(tmp_path, monkeypatch):
    reference = tmp_path / "data_processed" / "task_3_1_polars" / panel.REFERENCE_RUN
    candidate = tmp_path / "candidate"
    reference.mkdir(parents=True)
    candidate.mkdir()
    pl.DataFrame({"CERT": [1], "P3ASSET": [2.0]}).write_parquet(reference / "observed.parquet")
    pl.DataFrame({"CERT": [1], "P3ASSET": [3.0]}).write_parquet(candidate / "observed.parquet")
    monkeypatch.setattr(panel, "compare", lambda *_: (_ for _ in ()).throw(ValueError("parity differs")))
    with pytest.raises(ValueError, match=r"observed: first mismatch row=0 field=P3ASSET"):
        panel.compare_diagnostic(tmp_path, candidate)
def test_portable_reference_does_not_fall_back(tmp_path):
    import pytest
    from bft65b.benchmarks.panel import input_contract

    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/source_vintage_bundle.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="explicit portable reference"):
        input_contract(tmp_path)
    with pytest.raises(ValueError, match="Invalid benchmark reference run ID"):
        input_contract(tmp_path, "../historical")

