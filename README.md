# Bank Failure Early Warning

Nguyen Cong Hoan, 11230965. BFT65B, National Economics University.
Canonical repository: https://github.com/nguyenconghoan5305-code/Big_Data_11230965

The project builds an auditable FDIC bank-quarter pipeline with availability-aware features and chronological validation. The locked evaluation used 246,083 rows and 53 represented failures. AP was 0.172989; the fixed alarm policy had a higher declared cost surrogate than never alarming (0.955582 versus 0.782759 nominal USD thousands per row).

## Start here

- [Report PDF](submission/BFT65B_Final_Report.pdf) and [editable Word](submission/BFT65B_Final_Report.docx)
- [Source](src/bft65b), [synthetic tests](tests), [evidence register](EVIDENCE_REGISTER.csv)
- [Reproduction evidence](P6_REPRODUCIBILITY.md), [policies](PROJECT_PLAN.md), [tasks](TASKS.md)
- [Historical execution commands and milestone notes](docs/archive/README_milestones_20261126.md)

## A. Rebuild the report from public aggregate evidence

Use CPython 3.11 and a fresh virtual environment. This route uses only reporting/submission_manuscript.docx and submission/aggregate_evidence.json; it never calls collect_submission.py.

```powershell
py -3.11 -m venv .venv_report
.\.venv_report\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv_report\Scripts\python.exe -m pip check
$env:BFT_REPORT_OUTPUT = (Join-Path $PWD 'tmp/report_validation')
.\.venv_report\Scripts\python.exe reporting/submission_figures.py
.\.venv_report\Scripts\python.exe reporting/build_submission.py
```

Set BFT_REPORT_FONT to a TrueType file if Arial or DejaVu Sans cannot be found. Export the generated Word document to PDF using Microsoft Word; export and page-review evidence are recorded in the submission QA. PDF generation requires an installed renderer, separately from Python packages.

## B. Run synthetic checks

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-win-py311.txt
$env:PYTHONPATH = (Join-Path $PWD 'src')
.\.venv\Scripts\python.exe -m pytest -q --basetemp tmp/fresh_tests
```

Spark checks require JDK 17, the documented native libraries and current-process HADOOP_HOME and PYSPARK_PYTHON. See the archived Task 3.0 instructions for hashes and acceptance commands. A skipped Spark gate does not establish runtime acceptance.

## C. Replay captured data

Provision the immutable Financials, Failures, Institutions and supplementary History vintage using the source-vintage and membership-source-vintage CLIs documented in P6_REPRODUCIBILITY.md and the archived README. Select explicit upstream IDs at every stage and fresh output IDs. Required ignored snapshots and native tools are not included in a source checkout. New live API downloads can contain revisions and cannot promise historical byte equality.

Accepted portable stages and safe verification commands are recorded in P6_REPRODUCIBILITY.md. Execution writes data_processed/ and runs/. Preserve failed attempts and diagnose their logs before retrying. Do not overwrite occupied IDs or rewrite historical manifests.

The final20261122a test has already been used once. Use only its read-only verifier and saved-score metrics after completion. Do not execute final plan, fit or evaluate as a routine reproduction check.

## Report source roles

submission_figures.py and build_submission.py are the canonical report entry points. submission_manuscript.docx is their required editable manuscript. collect_submission.py and extract_report_evidence.py are local evidence-export helpers requiring provisioned ignored artifacts. generate_figures.py and build_academic_report.py are historical authoring prototypes retained for provenance.

## Measurements and limitations

The portable processing session measured pandas 72.692 s, Polars 4.116 s and local Spark 64.504 s on the same workload. Five fresh-process measurements per engine include startup, full materialization and three Parquet writes. Historical pandas 35.599 s belongs to a separate session. Cache state was uncontrolled; local Spark does not establish distributed scaling. The retrospective cohort and provisional 120-day availability lag do not reconstruct exact historical publication vintages.

## Sources and assistance

The course assignment is authoritative. FDIC API and dictionaries: https://api.fdic.gov/banks/docs/. Cite the individual measured runs through the evidence register. OpenAI Codex assisted with implementation, verification, figures and editing. Reference repository nahnhh/fdic-ccc was inspected at 46c36a4a3e43b85c0f00f89c4a1627a87c728359; no code was adapted. Its different workloads are not comparative performance evidence.
