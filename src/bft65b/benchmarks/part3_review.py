"""Read-only Part 3 evidence gate and saved scaling summary.

This verifies accepted artifacts; it does not run a benchmark or infer a
cluster break-even point from local measurements.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from bft65b.benchmarks.panel import (
    ENGINES, NAMESPACE, REFERENCE_RUN, THREE_ENGINES, compare_diagnostic,
    input_contract, paths_for, run_order, summarize,
)
from bft65b.processing.polars_panel import _accepted_result
from bft65b.quality.audit import _read_json, _sha, _write_json


PRIOR_RUN = "panelbench20261115a"
THREE_ENGINE_RUN = "panelbench20261115b"
REVIEW_NAMESPACE = "task_3_4_review"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_sequence(records: list[dict[str, Any]], engines: tuple[str, ...]) -> None:
    """Reject missing, reordered, failed, or partially measured process runs."""
    expected = ([('preflight', 0, engine) for engine in engines]
                + [('warmup', 0, engine) for engine in engines]
                + [('measured', number, engine) for number, engine in run_order(engines)])
    actual = [(r.get('kind'), r.get('round'), r.get('engine')) for r in records]
    _require(actual == expected, f"Process run inventory/order differs: {actual}")
    for row in records:
        _require(row.get('status') == 'pass' and row.get('exit_code') == 0,
                 f"Failed worker: {row.get('kind')} {row.get('engine')}")
        _require(row.get('duration_ns', 0) > 0
                 and row.get('peak_tree_rss_bytes', 0) > 0
                 and row.get('memory_samples', 0) > 0,
                 f"Missing time or memory measurement: {row.get('engine')}")
        if row['kind'] != 'preflight':
            _require(row.get('parity', {}).get('status') == 'exact_parity',
                     f"Saved parity result missing: {row.get('engine')}")


def _accepted_paths(root: Path) -> dict[str, str]:
    """Hash small accepted evidence plus all source and saved result files."""
    paths = []
    p31 = root / 'runs' / 'task_3_1_polars' / REFERENCE_RUN
    paths += [p31 / name for name in ('run_manifest.json', 'summary.json',
                                     'metadata.json', 'contract.json')]
    for name in ('observed', 'gaps', 'quality'):
        paths.append(p31 / f'{name}_optimized_plan.txt')
        paths.append(root / 'data_processed' / 'task_3_1_polars'
                     / REFERENCE_RUN / f'{name}.parquet')
    contract = _read_json(p31 / 'contract.json')
    paths += [Path(row['path']) for row in contract['input_files']]
    for run_id in (PRIOR_RUN, THREE_ENGINE_RUN):
        evidence, _ = paths_for(root, run_id)
        paths += [evidence / name for name in ('summary.json', 'records.json',
                                              'parity.json', 'environment.json')]
        records = _read_json(evidence / 'records.json')['records']
        for row in records:
            paths += [Path(row['output_dir']) / f'{name}.parquet'
                      for name in ('observed', 'gaps', 'quality')]
        if run_id == THREE_ENGINE_RUN:
            plans = evidence / 'plans'
            paths.append(plans / 'settings.json')
            paths += [plans / f'{name}_{stage}.json'
                      for name in ('observed', 'gaps', 'quality')
                      for stage in ('before', 'after')]
    return {str(path): _sha(path) for path in paths}


def _verify_plan(root: Path) -> dict[str, Any]:
    p31 = _accepted_result(root, REFERENCE_RUN)
    expected_projections = {'observed': 'PROJECT 6/29 COLUMNS',
                            'gaps': 'PROJECT 2/29 COLUMNS',
                            'quality': 'PROJECT 5/29 COLUMNS'}
    for name, token in expected_projections.items():
        row = p31['plans'][name]
        path = Path(row['path'])
        _require(_sha(path) == row['sha256'] and token in path.read_text(encoding='utf-8'),
                 f'Polars {name} optimized plan differs')
    plan_dir = root / 'runs' / NAMESPACE / THREE_ENGINE_RUN / 'plans'
    settings = _read_json(plan_dir / 'settings.json')
    _require(settings['configuration']['spark.master'] == 'local[4]'
             and settings['configuration']['spark.sql.shuffle.partitions'] == '16'
             and settings['configuration']['spark.sql.adaptive.enabled'] == 'true'
             and settings['default_parallelism_effective'] == 4,
             'Effective Spark parallelism/AQE settings differ')
    projected = {'observed': ('CERT', 'REPDTE', 'P3ASSET', 'ID',
                              'SOURCE_MISSING_FIELDS', 'DQ_FLAGS'),
                 'gaps': ('CERT', 'REPDTE'),
                 'quality': ('CERT', 'REPDTE', 'P3ASSET',
                             'SOURCE_MISSING_FIELDS', 'DQ_FLAGS')}
    for name, columns in projected.items():
        before = _read_json(plan_dir / f'{name}_before.json')
        after = _read_json(plan_dir / f'{name}_after.json')
        physical = after['physical'].split('== Initial Plan ==')[0]
        scan = next((line for line in physical.splitlines()
                     if 'FileScan parquet [' in line), '')
        scan_columns = scan.split('FileScan parquet [', 1)[-1].split(']', 1)[0]
        _require(before['adaptive_is_final'] is False
                 and after['adaptive_is_final'] is True
                 and all(any(item.startswith(f'{column}#') for item in scan_columns.split(','))
                         for column in columns)
                 and len(scan_columns.split(',')) == len(columns),
                 f'Spark {name} executed plan or projection differs')
        required = {'observed': ('Sort [', 'Exchange'),
                    'gaps': ('Window [', 'Sort [', 'Exchange'),
                    'quality': ('HashAggregate', 'Exchange')}[name]
        _require(all(token in physical for token in required),
                 f'Spark {name} physical operators differ')
    return {'polars_plans': 3, 'spark_before_plans': 3,
            'spark_final_plans': 3, 'spark_settings': settings}


def _verify_run(root: Path, run_id: str,
                engines: tuple[str, ...], contract_hash: str,
                reference_run: str = REFERENCE_RUN) -> dict[str, Any]:
    evidence, data = paths_for(root, run_id)
    summary = _read_json(evidence / 'summary.json')
    records_doc = _read_json(evidence / 'records.json')
    parity = _read_json(evidence / 'parity.json')
    environment = _read_json(evidence / 'environment.json')
    rows = records_doc['records']
    _require(summary.get('status') == records_doc.get('status') == 'complete'
             and summary.get('run_id') == run_id
             and summary.get('reference_run') == reference_run
             and summary.get('input_contract_sha256') == contract_hash
             and summary.get('measured_runs') == len(engines) * 5,
             f'{run_id} summary/records/contract differ')
    validate_sequence(rows, engines)
    _require(summary['by_engine'] == summarize(rows, engines)
             and set(parity) == set(engines)
             and set(summary['parity']) == set(engines),
             f'{run_id} statistics or parity inventory differ')
    _require(environment['input_files'] == 170
             and environment['input_rows'] == 1682616
             and environment['input_stored_bytes'] == 191535956
             and environment['memory']['interval_ms'] == 50
             and 'perf_counter_ns' in environment['timer']
             and 'not flushed' in environment['cache'],
             f'{run_id} input scope or timing/cache record differs')
    if engines == THREE_ENGINES:
        _require(environment['engines'] == list(engines)
                 and environment['ordering']['measured'] == [
                     {'round': number, 'engine': engine}
                     for number, engine in run_order(engines)],
                 'Three-engine rotation/environment differs')
    for row in rows:
        output = Path(row['output_dir']).resolve()
        _require(output.is_relative_to(data.resolve())
                 and row['command'][1:4] == ['-m', 'bft65b.benchmarks.panel', '_worker']
                 and row['command'][4] == row['engine']
                 and Path(row['stdout']).is_file() and Path(row['stderr']).is_file(),
                 f'{run_id} worker command/path/log differs')
        actual = compare_diagnostic(root, output, reference_run)
        _require(actual['status'] == 'exact_parity'
                 and actual['rows_by_relation'] == {'observed': 1682616,
                                                     'gaps': 33, 'quality': 170},
                 f'{run_id} full output differs: {row["engine"]}')
        if row['kind'] == 'preflight':
            _require(parity[row['engine']] == actual
                     and summary['parity'][row['engine']] == actual,
                     f'{run_id} preflight parity evidence differs')
        else:
            _require(row['parity'] == actual,
                     f'{run_id} per-run parity evidence differs')
    return {'run_id': run_id, 'engines': list(engines),
            'full_output_comparisons': len(rows),
            'measured_runs': len(engines) * 5,
            'environment': environment, 'by_engine': summary['by_engine'],
            'input_contract_sha256': contract_hash}


def verify(root: Path, run_id: str) -> dict[str, Any]:
    output = root / 'runs' / REVIEW_NAMESPACE / run_id
    _require(run_id and all(c.isalnum() or c in '-_' for c in run_id),
             'Invalid review run ID')
    _require(not output.exists(), 'Review evidence run ID already exists')
    contract = input_contract(root)
    expected_hash = _sha(root / 'runs' / 'task_3_1_polars'
                         / REFERENCE_RUN / 'contract.json')
    _require(len(contract['input_files']) == 170, 'Accepted input inventory differs')
    before = _accepted_paths(root)
    plans = _verify_plan(root)
    prior = _verify_run(root, PRIOR_RUN, ENGINES, expected_hash)
    current = _verify_run(root, THREE_ENGINE_RUN, THREE_ENGINES, expected_hash)
    _require(prior['environment']['input_stored_bytes']
             == current['environment']['input_stored_bytes']
             and prior['environment']['python'] == current['environment']['python']
             and prior['environment']['versions'] == current['environment']['versions'],
             'Benchmark input or software environment differs across runs')
    input_contract(root)
    after = _accepted_paths(root)
    _require(before == after, 'Accepted upstream artifact checksum changed during review')
    ram = current['environment']['ram_bytes']
    _require(isinstance(ram, int) and ram > 0, 'Measured host RAM is unavailable')
    measured = current['by_engine']
    max_peak = max(max(row['peak_tree_rss_bytes']) for row in measured.values())
    result = {
        'status': 'complete', 'run_id': run_id,
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'workload_id': contract['workload_id'],
        'input_files': len(contract['input_files']),
        'input_rows': sum(row['rows'] for row in contract['input_files']),
        'input_stored_bytes': current['environment']['input_stored_bytes'],
        'contract_sha256': expected_hash,
        'accepted_artifact_hashes': before,
        'accepted_artifact_count': len(before),
        'full_output_comparisons': prior['full_output_comparisons']
                                   + current['full_output_comparisons'],
        'plans': plans,
        'prior_two_engine_run': {'run_id': PRIOR_RUN,
                                 'measured_runs': prior['measured_runs']},
        'three_engine_run': {'run_id': THREE_ENGINE_RUN,
                             'measured_runs': current['measured_runs'],
                             'by_engine': measured},
        'capacity_context': {
            'physical_ram_bytes': ram,
            'max_observed_sampled_process_tree_rss_bytes': max_peak,
            'max_sampled_rss_to_ram_fraction': max_peak / ram,
            'free_disk_bytes_before_three_engine_run':
                current['environment']['free_disk_bytes_before'],
            'free_disk_bytes_at_review': shutil.disk_usage(root).free,
            'cpu_utilization_measured': False,
            'cluster_crossover_measured': False,
        },
        'limitations': [
            'The OS file cache was not controlled; results are not cold-cache.',
            'Startup, three output writes, and Spark shutdown are included.',
            'Sampled process-tree RSS can miss short peaks and double-count shared pages.',
            'No larger workload, cluster execution, CPU utilization, or numerical crossover was measured.',
        ],
    }
    output.mkdir(parents=True)
    _write_json(output / 'review.json', result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Part 3 saved-evidence review')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('verify', 'show'):
        p = sub.add_parser(command)
        p.add_argument('--project-root', type=Path, default=Path.cwd())
        p.add_argument('--run-id', required=True)
    portable = sub.add_parser('verify-portable')
    portable.add_argument('--project-root', type=Path, required=True)
    portable.add_argument('--run-id', required=True)
    portable.add_argument('--reference-run', required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if args.command == 'verify-portable':
        input_contract(root, args.reference_run)
        contract = root / 'runs/task_3_1_polars' / args.reference_run / 'contract.json'
        result = _verify_run(root, args.run_id, THREE_ENGINES, _sha(contract), args.reference_run)
        print(json.dumps(result, indent=2))
        return
    if args.command == 'verify':
        result = verify(root, args.run_id)
    else:
        result = _read_json(root / 'runs' / REVIEW_NAMESPACE / args.run_id / 'review.json')
        _require(result.get('status') == 'complete'
                 and result.get('run_id') == args.run_id,
                 'Saved Part 3 review is incomplete')
    display = {key: result[key] for key in ('status', 'run_id', 'workload_id',
               'input_files', 'input_rows', 'input_stored_bytes',
               'accepted_artifact_count', 'full_output_comparisons',
               'capacity_context')}
    display['median_by_engine'] = {
        name: {'duration_ns': row['median_duration_ns'],
               'peak_tree_rss_bytes': row['median_peak_tree_rss_bytes']}
        for name, row in result['three_engine_run']['by_engine'].items()}
    print(json.dumps(display, indent=2))


if __name__ == '__main__':
    main()
