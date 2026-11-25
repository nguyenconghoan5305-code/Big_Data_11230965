"""Prepare public aggregate inputs from the original report extraction and local frozen evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission"
def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))
d = load("outputs/academic_report/plot_evidence.json")
for path, digest in d["sources"].items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
s = d["storage"]
p = d["processing"]
assert s["status"] == "complete" and len(s["records"]) == 30
assert p["run_id"] == "portable20261125bbench" and p["measured_runs"] == 15 and p["full_output_comparisons"] == 21
q = d["quality"]["datasets"]
quality = []
for dataset, field, reason in [("financials","P3ASSET","explicit_null"),("institutions","CHANGEC1","absent"),("failures","CERT","explicit_null")]:
    counts = q[dataset]["field_counts"][field]
    quality.append(dict(dataset=dataset,field=field,rule=reason,count=counts[reason],denominator=counts["denominator"]))
def storage_rows(obj):
    keys = ("format","workload","rows_read","columns_read","stored_bytes","median_seconds","min_seconds","max_seconds")
    return [{k:row[k] for k in keys} for row in obj["summary"]]
sources = dict(d["sources"])
extra = {}
for name,rel in [("lock","runs/task_5_5_threshold/threshold20261122c/locked_policy.json"),("sensitivity","runs/task_5_5_threshold/threshold20261122c/sensitivity.json")]:
    extra[name]=load(rel)
    sources[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
ref=next(row for row in extra["sensitivity"]["rows"] if row["alpha"]==0.1 and row["beta"]==0.01)
evidence = dict(storage=storage_rows(s),historical_storage=storage_rows(d["storage_old"]),engines=p["by_engine"],historical_engines=d["processing_old"]["three_engine_run"]["by_engine"],quality=quality,folds=d["development"]["folds"],fold_results=d["development"]["fold_results"],split=d["split"]["bounds"],periods=d["split"]["periods"],feature_nulls=d["features"]["model_nulls"],join=d["features"]["overall"],spark_metrics=d["spark"]["metrics"],lock=extra["lock"]["policy"],development_policy=ref,final=d["final"]["result"],curves=d["curves"],sources=sources)
OUT.mkdir(exist_ok=True)
(OUT/"aggregate_evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,separators=(",",":"))+"\n",encoding="utf-8")
print("Aggregate-only plotting evidence written;",len(sources),"source hashes verified")

