"""Synthetic acceptance gates for the Part 3 evidence inventory."""

import pytest

from bft65b.benchmarks.panel import ENGINES, THREE_ENGINES, run_order
from bft65b.benchmarks.part3_review import validate_sequence


def _records(engines):
    order = ([('preflight', 0, engine) for engine in engines]
             + [('warmup', 0, engine) for engine in engines]
             + [('measured', number, engine) for number, engine in run_order(engines)])
    return [{'kind': kind, 'round': number, 'engine': engine,
             'status': 'pass', 'exit_code': 0, 'duration_ns': 10,
             'peak_tree_rss_bytes': 100, 'memory_samples': 2,
             'parity': {'status': 'exact_parity'}}
            for kind, number, engine in order]


@pytest.mark.parametrize('engines', (ENGINES, THREE_ENGINES))
def test_complete_rotated_process_inventory_passes(engines):
    validate_sequence(_records(engines), engines)


def test_missing_or_reordered_measurement_fails():
    rows = _records(THREE_ENGINES)
    with pytest.raises(ValueError, match='inventory/order'):
        validate_sequence(rows[:-1], THREE_ENGINES)
    rows[-1], rows[-2] = rows[-2], rows[-1]
    with pytest.raises(ValueError, match='inventory/order'):
        validate_sequence(rows, THREE_ENGINES)


def test_failed_worker_or_missing_memory_or_parity_fails():
    for field, value, message in (
        ('status', 'failed', 'Failed worker'),
        ('memory_samples', 0, 'Missing time or memory'),
        ('parity', {'status': 'failed'}, 'Saved parity'),
    ):
        rows = _records(ENGINES)
        rows[-1][field] = value
        with pytest.raises(ValueError, match=message):
            validate_sequence(rows, ENGINES)
