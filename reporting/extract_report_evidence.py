"""Read immutable evidence and export aggregate plotting inputs; never fit or score."""
from pathlib import Path
import json, hashlib
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/academic_report'
OUT.mkdir(parents=True, exist_ok=True)
sources = {}
def read(rel):
    p=ROOT/rel; sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding='utf-8'))
portable='tmp/p6_repro_20261123b/portable/'
data={
 'storage':read(portable+'runs/storage_portable20261125a/timings.json'),
 'storage_spec':read(portable+'runs/storage_portable20261125a/frozen_spec.json'),
 'storage_old':read('runs/task_1_6_storage/timings.json'),
 'processing':read('tmp/p6_portable_processing_20261125b_review.json'),
 'processing_old':read('runs/task_3_4_review/part3review20261115a/review.json'),
 'quality':read('runs/task_2_1_quality/audit20261114b/findings.json'),
 'development':read('runs/task_5_2_development/dev20261121a/summary.json'),
 'final':read('runs/task_5_6_final/final20261122a/summary.json'),
 'split':read('runs/task_5_1_split/split20261120b/summary.json'),
 'spark':read('runs/task_5_3_spark_mllib/sparkdemo20261122a/summary.json'),
 'features':read('runs/task_4_3_feature_table/featuretable20261120a/summary.json'),
}
score='data_processed/task_5_6_final/final20261122a/test_scores.parquet'
p=ROOT/score; sources[score]=hashlib.sha256(p.read_bytes()).hexdigest()
assert sources[score]==data['final']['score_sha256']
t=pq.read_table(p).to_pydict(); y=np.array(t['Y']); s=np.array(t['SCORE'])
precision,recall,_=precision_recall_curve(y,s)
fpr,tpr,_=roc_curve(y,s,drop_intermediate=True)
assert abs(average_precision_score(y,s)-data['final']['result']['average_precision'])<1e-12
assert abs(roc_auc_score(y,s)-data['final']['result']['roc_auc'])<1e-12
# Store complete threshold-free descriptive curves. Do not select a new operating point.
data['curves']={'pr':list(zip(recall.tolist(),precision.tolist())), 'roc':list(zip(fpr.tolist(),tpr.tolist()))}
data['sources']=sources
# Exclude bank identifiers, paths to individual raw partitions and individual predictions.
for key in ['development','spark','features','split']:
    data[key].pop('input_hashes',None)
(OUT/'plot_evidence.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
(OUT/'source_hashes.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
print('Read-only aggregate export complete:',len(sources),'sources')
