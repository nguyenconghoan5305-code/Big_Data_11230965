"""Build the publication-grade academic report (DOCX) from verified aggregate evidence."""
from pathlib import Path
import json
import re
import statistics
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/academic_report'
FIG = OUT / 'figures'
D = json.loads((OUT / 'plot_evidence.json').read_text(encoding='utf-8'))

doc = Document()

# Enable automatic hyphenation in document settings
settings = doc.settings.element
hyp = OxmlElement('w:autoHyphenation')
hyp.set(qn('w:val'), 'true')
settings.append(hyp)

# A4 geometry with 2.5 cm academic margins
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.5)
sec.bottom_margin = Cm(2.5)
sec.left_margin = Cm(2.5)
sec.right_margin = Cm(2.5)
sec.header_distance = Cm(1.1)
sec.footer_distance = Cm(1.1)
sec.different_first_page_header_footer = True

NAVY_HEX = '102A43'
TEAL_HEX = '1D6F72'
SLATE_HEX = '627D98'
BODY_HEX = '1F2933'
SHADE_HEX = 'F8FAFC'

BODY_FONT = 'Georgia'
HEAD_FONT = 'Georgia'
MONO_FONT = 'Consolas'

# Typography styles
for name in ['Normal', 'Body Text']:
    s = doc.styles[name]
    s.font.name = BODY_FONT
    s.font.size = Pt(10.5)
    s.font.color.rgb = RGBColor.from_string(BODY_HEX)
    s.paragraph_format.space_after = Pt(4)
    s.paragraph_format.line_spacing = 1.20

for name, size, sb in [('Title', 18, 0), ('Heading 1', 13, 11), ('Heading 2', 11, 8), ('Heading 3', 10.5, 6)]:
    s = doc.styles[name]
    s.font.name = HEAD_FONT
    s.font.size = Pt(size)
    s.font.bold = True
    s.font.color.rgb = RGBColor.from_string(NAVY_HEX)
    s.paragraph_format.space_before = Pt(sb)
    s.paragraph_format.space_after = Pt(4)
    s.paragraph_format.keep_with_next = True

sub_style = doc.styles['Subtitle']
sub_style.font.name = HEAD_FONT
sub_style.font.size = Pt(10)
sub_style.font.color.rgb = RGBColor.from_string(SLATE_HEX)
sub_style.paragraph_format.space_after = Pt(2)

cap_style = doc.styles['Caption']
cap_style.font.name = BODY_FONT
cap_style.font.size = Pt(9)
cap_style.font.color.rgb = RGBColor.from_string(SLATE_HEX)
cap_style.paragraph_format.space_after = Pt(5)

doc.core_properties.author = 'Nguyen Cong Hoan'
doc.core_properties.title = 'A Big-Data Pipeline for Bank Failure Early Warning'
doc.core_properties.subject = 'BFT65B Final Individual Research Report'


def add_bottom_border(para, color=SLATE_HEX, sz='4'):
    pPr = para._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    b = OxmlElement('w:bottom')
    b.set(qn('w:val'), 'single')
    b.set(qn('w:sz'), sz)
    b.set(qn('w:space'), '3')
    b.set(qn('w:color'), color)
    pBdr.append(b)
    pPr.append(pBdr)


def add_field(para, instr, size=8.5, color=SLATE_HEX):
    run = para.add_run()
    run.font.name = BODY_FONT
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = f' {instr} '
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')
    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)


# Running header (pages 2+) with right tab at 16.0 cm (21 - 2.5 - 2.5)
hdr = sec.header.paragraphs[0]
hdr.paragraph_format.tab_stops.add_tab_stop(Cm(16.0), WD_ALIGN_PARAGRAPH.RIGHT)
hdr.paragraph_format.space_after = Pt(2)
rh1 = hdr.add_run('A Big-Data Pipeline for Bank Failure Early Warning\tBFT65B')
rh1.font.name = BODY_FONT
rh1.font.size = Pt(8.5)
rh1.font.color.rgb = RGBColor.from_string(SLATE_HEX)
add_bottom_border(hdr, SLATE_HEX, '4')

# Running footer (pages 2+ and page 1)
for ftr in [sec.footer.paragraphs[0], sec.first_page_footer.paragraphs[0]]:
    ftr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = ftr.add_run('Page ')
    r1.font.name = BODY_FONT
    r1.font.size = Pt(8.5)
    r1.font.color.rgb = RGBColor.from_string(SLATE_HEX)
    add_field(ftr, 'PAGE')
    r2 = ftr.add_run(' of ')
    r2.font.name = BODY_FONT
    r2.font.size = Pt(8.5)
    r2.font.color.rgb = RGBColor.from_string(SLATE_HEX)
    add_field(ftr, 'NUMPAGES')


TOKEN_RE = re.compile(r'(`[^`]+`|\*\*[^*]+\*\*)')


def add_rich_runs(para, text, base_size=10.5, base_color=BODY_HEX):
    for part in TOKEN_RE.split(text):
        if not part:
            continue
        if part.startswith('`') and part.endswith('`'):
            r = para.add_run(part[1:-1])
            r.font.name = MONO_FONT
            r.font.size = Pt(9.0)
            r.font.color.rgb = RGBColor.from_string(NAVY_HEX)
        elif part.startswith('**') and part.endswith('**'):
            r = para.add_run(part[2:-2])
            r.bold = True
            r.font.name = BODY_FONT
            r.font.size = Pt(base_size)
            r.font.color.rgb = RGBColor.from_string(base_color)
        else:
            r = para.add_run(part)
            r.font.name = BODY_FONT
            r.font.size = Pt(base_size)
            r.font.color.rgb = RGBColor.from_string(base_color)


def p(text, style=None, justify=True, space_after=None):
    para = doc.add_paragraph(style=style)
    if justify and style is None:
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if space_after is not None:
        para.paragraph_format.space_after = Pt(space_after)
    add_rich_runs(para, text, base_size=9.0 if style == 'Caption' else 10.5, base_color=SLATE_HEX if style == 'Caption' else BODY_HEX)
    return para


def eq(formula, number):
    para = doc.add_paragraph()
    para.paragraph_format.tab_stops.add_tab_stop(Cm(8.0), WD_ALIGN_PARAGRAPH.CENTER)
    para.paragraph_format.tab_stops.add_tab_stop(Cm(16.0), WD_ALIGN_PARAGRAPH.RIGHT)
    para.paragraph_format.space_before = Pt(3)
    para.paragraph_format.space_after = Pt(4)
    r_tab = para.add_run('\t')
    r_eq = para.add_run(formula)
    r_eq.font.name = BODY_FONT
    r_eq.font.size = Pt(10.0)
    r_eq.italic = True
    r_eq.font.color.rgb = RGBColor.from_string(NAVY_HEX)
    r_num = para.add_run(f'\t({number})')
    r_num.font.name = BODY_FONT
    r_num.font.size = Pt(9.5)
    r_num.font.color.rgb = RGBColor.from_string(SLATE_HEX)
    return para


def h(title, level=1):
    para = doc.add_heading(title, level)
    para.paragraph_format.keep_with_next = True
    return para


def page(title, level=1):
    doc.add_page_break()
    return h(title, level)


def fig(name, caption, width=6.25):
    pp = doc.add_paragraph()
    pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pp.paragraph_format.space_before = Pt(5)
    pp.paragraph_format.space_after = Pt(2)
    pp.paragraph_format.keep_with_next = True
    pp.add_run().add_picture(str(FIG / f'{name}.png'), width=Inches(width))
    pp._p.xpath('.//wp:docPr')[0].set('descr', caption)
    cp = p(caption, 'Caption', justify=True, space_after=6)
    return cp


def set_cell_border(cell, **edges):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for edge in ['top', 'bottom', 'left', 'right']:
        el = OxmlElement(f'w:{edge}')
        if edge in edges and edges[edge] is not None:
            val, sz, color = edges[edge]
            el.set(qn('w:val'), val)
            el.set(qn('w:sz'), str(sz))
            el.set(qn('w:color'), color)
        else:
            el.set(qn('w:val'), 'nil')
        tcBorders.append(el)
    tcPr.append(tcBorders)


def booktabs_table(title, headers, rows, widths=None, right_cols=None):
    cp = p(title, 'Caption', justify=False, space_after=2)
    cp.paragraph_format.space_before = Pt(5)
    cp.paragraph_format.keep_with_next = True

    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False

    all_rows = [headers] + list(rows)
    for i, v in enumerate(headers):
        t.rows[0].cells[i].text = ''
        add_rich_runs(t.rows[0].cells[i].paragraphs[0], str(v), base_size=8.8, base_color=NAVY_HEX)
        t.rows[0].cells[i].paragraphs[0].runs[0].bold = True

    for vals in rows:
        cells = t.add_row().cells
        for i, v in enumerate(vals):
            cells[i].text = ''
            add_rich_runs(cells[i].paragraphs[0], str(v), base_size=8.8, base_color=BODY_HEX)

    nrows = len(all_rows)
    for ri, row in enumerate(t.rows):
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement('w:cantSplit'))
        if ri == 0:
            trpr.append(OxmlElement('w:tblHeader'))
        for ci, c in enumerate(row.cells):
            if widths:
                c.width = Cm(widths[ci])
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcPr = c._tc.get_or_add_tcPr()

            # Subtle alternating row shading on data rows
            if ri > 0 and ri % 2 == 0:
                shd = OxmlElement('w:shd')
                shd.set(qn('w:fill'), SHADE_HEX)
                tcPr.append(shd)

            # Booktabs rules: 1.2pt top, 0.75pt midrule below header, 1.2pt bottom
            top_b = ('single', 10, NAVY_HEX) if ri == 0 else None
            bot_b = ('single', 6, NAVY_HEX) if ri == 0 else (('single', 10, NAVY_HEX) if ri == nrows - 1 else None)
            set_cell_border(c, top=top_b, bottom=bot_b, left=None, right=None)

            # Generous cell padding: top/bottom 4pt (80 dxa), left/right 6pt (120 dxa)
            margins = OxmlElement('w:tcMar')
            for edge, val in [('top', '80'), ('bottom', '80'), ('left', '120'), ('right', '120')]:
                el = OxmlElement(f'w:{edge}')
                el.set(qn('w:w'), val)
                el.set(qn('w:type'), 'dxa')
                margins.append(el)
            tcPr.append(margins)

            for pp in c.paragraphs:
                pp.paragraph_format.space_after = Pt(1.5)
                pp.paragraph_format.space_before = Pt(1.5)
                pp.paragraph_format.line_spacing = 1.05
                is_num = (right_cols is not None and ci in right_cols) or (
                    right_cols is None and ci > 0
                    and all(ch in '0123456789.,% /xâˆ’–-()+=' for ch in c.text.strip())
                    and any(ch.isdigit() for ch in c.text)
                )
                if is_num:
                    pp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                if ri == 0:
                    for r in pp.runs:
                        r.bold = True
                        r.font.color.rgb = RGBColor.from_string(NAVY_HEX)

    sp = p('', space_after=3)
    return t


# =========================================================================
# PAGE 1: TITLE, METADATA, ABSTRACT, TABLE 1
# =========================================================================
p('NATIONAL ECONOMICS UNIVERSITY  ·  BFT65B: BIG DATA ANALYTICS IN FINANCE', 'Subtitle', justify=False)
p('A Big-Data Pipeline for Bank Failure Early Warning', 'Title', justify=False)
p('Final Individual Research Report', 'Subtitle', justify=False)
p(
    '**Author:** Nguyen Cong Hoan (Student ID: 11230965)  ·  '
    '**Repository:** https://github.com/nguyenconghoan5305-code/Big_Data_11230965',
    justify=False,
    space_after=6,
)

# Abstract block
h('Abstract', 2)
p(
    'Early-warning supervision models fail in practice not for lack of classification algorithms, but when '
    'unreconciled reporting vintages, missingness heterogeneity, and look-ahead leakage distort historical panels. '
    'This study designs and audits a reproducible data-engineering pipeline across four decades of US Federal Deposit '
    'Insurance Corporation (FDIC) Call Reports (`1984Q1–2026Q2`, 1,682,616 bank-quarter observations). Quarter-partitioned '
    'ZSTD-3 Parquet reduces storage by 67.27% relative to CSV, and lazy Polars execution processes the panel 17.66x '
    'faster than pandas with exact parity against local PySpark, demonstrating that laptop-scale tooling suffices '
    'without a distributed cluster. Strict point-in-time availability rules (`REPDTE + 120d <= q`), transaction-level '
    'exit reconciliation, and fold-local preprocessing enforce temporal discipline. Under a single locked test '
    'evaluation (`2014Q1–2025Q2`; 53 distinct failures), bounded histogram gradient boosting achieves `AP = 0.1730` and '
    '`ROC-AUC = 0.8770` (detecting 38 of 53 failed banks), yet its development-selected alert policy (`SCORE >= 0.0321005`) '
    'incurs higher nominal surrogate costs than never alarming (`0.9556` vs. `0.7828` USDk/bank-quarter). The findings '
    'separate ranking discrimination from operational readiness and motivate cautious adaptation for Vietnamese banking supervision.',
    space_after=6,
)

booktabs_table(
    'Table 1. Study provenance, dataset scope, and chronological evaluation split [E1, E6]',
    ['Provenance / Split Dimension', 'Verified Scope and Measured Counts'],
    [
        ('Acquisition method & date', 'FDIC BankFind Suite public REST API; main snapshot captured 11 November 2026 (`real20261111b`)'),
        ('Financials Call Reports', '`1984Q1–2026Q2` (170 quarterly partitions); 1,682,616 bank-quarter rows; 26 logical + 2 provenance fields'),
        ('Failures & Institutions', '4,117 Failures/Assistance records (unique source `ID`); 27,835 Institutions records (unique `CERT`)'),
        ('Development period', '`1984Q1–2012Q4`; 1,329,451 model rows; 9,677 positive rows; 2,462 distinct represented failure events'),
        ('Excluded embargo gap', '`2013Q1–2013Q4`; 4 complete quarters (27,649 model rows; 79 positive rows; 38 distinct events)'),
        ('Locked final test period', '`2014Q1–2025Q2`; 246,083 model rows; 178 positive rows; 53 distinct represented failure events (>= 50 gate)')
    ],
    [4.6, 11.4],
)

# =========================================================================
# SECTION 1: INTRODUCTION
# =========================================================================
h('1. Introduction and Research Context')
p(
    'Predicting commercial bank failures is fundamentally a rare-event, longitudinal data-engineering problem. Across '
    'four decades of regulatory filings, institution identifiers merge or disappear, accounting definitions evolve, '
    'and financial statements become public weeks or months after quarter-end. When these structural realities are '
    'ignored, standard machine-learning pipelines routinely suffer from look-ahead leakage and survivorship bias, '
    'producing inflated validation metrics that collapse in live supervision [1].'
)
p(
    'Accordingly, this study treats the data pipeline—rather than model hyperparameter search—as the primary research '
    'contribution. Using public US FDIC Call Reports as a transparent empirical test bed, we address three central '
    'questions: **(Q1)** how four decades of bank filings should be stored and processed on a laptop, and under what '
    'measured conditions a Spark cluster becomes justified; **(Q2)** what quality defects exist in raw supervisory '
    'records and how they can be resolved without look-ahead bias; and **(Q3)** whether the resulting pipeline supports '
    'an honestly validated early-warning model. Throughout, main-text benchmarks reference the November 25 portable '
    'replay family (`storage_portable20261125a`, `portable20261125bbench`) for internal consistency, while Appendix A '
    'reconciles earlier November 15 executions.'
)

# =========================================================================
# SECTION 2: DATA SOURCES & ARCHITECTURE
# =========================================================================
h('2. Data Sources and Pipeline Architecture')
p(
    'The FDIC BankFind Suite API [2] exposes three relational tables linked by certificate number (`CERT`): *Financials* '
    '(1,682,616 quarterly Call Reports across `1984Q1–2026Q2`), *Failures* (4,117 historical failure and open-bank '
    'assistance transactions), and *Institutions* (27,835 institution-level structural records). Treating every record '
    'in *Failures* as a bank failure or every exit from *Financials* as distress would corrupt the target definition. '
    'As illustrated in **Figure 1**, the architecture enforces strict separation across immutable raw ingestion, '
    'non-destructive quality flagging, retrospective cohort reconstruction, point-in-time predictor selection, and '
    'chronologically isolated model evaluation.'
)
fig(
    'Fig1_pipeline',
    'Figure 1. End-to-end pipeline architecture and leakage-control boundaries. The dashed terracotta TEST LOCK '
    'isolates development-only model and threshold selection from single-use final test evaluation. Source: [E1–E8].',
    6.25,
)

h('2.1 Acquisition, Dictionary-Based Field Selection, and Integrity Controls', 2)
p(
    'Full acquisition followed a validated four-quarter `2023` pilot. Every one of the 170 *Financials* queries and two '
    'complete small-table queries passed API total reconciliation, typed Arrow read-back, partition-scope checks, key '
    'uniqueness, and SHA-256 verification prior to atomic promotion (`real20261111b`) [E1]. *Financials* is keyed by '
    '(`CERT`, `REPDTE`), whereas *Failures* is keyed by API source `ID` because `CERT` is null in 488 legacy records '
    '(`1934–1976`) and repeated across 44 multi-transaction resolution cases. Rather than ingesting redundant columns, '
    '26 logical *Financials* fields were selected from the official YAML dictionary to span CAMELS risk dimensions '
    '(Table 2), augmented by two physical provenance columns (`SOURCE_MISSING_FIELDS`, `DQ_FLAGS`).'
)
booktabs_table(
    'Table 2. Justification of the 26 selected Financials v2 fields by supervisory risk domain [E1, E5]',
    ['Risk Domain', 'Selected FDIC Codes', 'Analytical Rationale and Dictionary Caveats'],
    [
        ('Identity & geography', '`CERT`, `REPDTE`, `STNAME`', 'Bank-quarter natural key and state context; `REPDTE` is accounting quarter-end, not release date.'),
        ('Scale & liquidity', '`ASSET`, `CHBAL`, `SC`', 'Total assets (scale and ratio denominator) plus cash/due-from and securities buffers.'),
        ('Funding structure', '`DEP`, `DEPDOM`, `BRO`, `EINTEXP`', 'Total/domestic deposits, brokered-deposit reliance, and interest-expense pressure.'),
        ('Lending composition', '`LNLSGR`, `LNRE`, `LNCI`', 'Gross loan intensity, real-estate concentration, and commercial/industrial exposure.'),
        ('Asset quality', '`NCLNLS`, `LNATRES`, `P3ASSET`, `P9ASSET`, `NAASSET`, `NTLNLS`, `ELNATR`', 'Noncurrent loans, reserves, 30–89/90+ past-due, nonaccrual, charge-offs, and provisions; `P3ASSET` is null in 68 early quarters.'),
        ('Earnings', '`NETINC`, `ROA`, `NIMY`', 'Net income, annualized return on assets (%), and net interest margin (%).'),
        ('Capital adequacy', '`EQ`, `RBC1AAJ`, `RBCRWAJ`', 'Book equity, PCA leverage ratio (`RBCT1/AVASSETJ`, %), and risk-based capital ratio (%).')
    ],
    [3.3, 5.1, 7.6],
)

# =========================================================================
# SECTION 3: Q1 STORAGE & PROCESSING
# =========================================================================
h('3. Q1 — Storage and Processing Architecture')
h('3.1 Columnar Storage Efficiency and Read Latency', 2)
p(
    'Text-based CSV/JSON files lack native date, decimal, and null-versus-absent types and require full-row scanning '
    'even for narrow analytical queries. Partitioning typed Parquet by `REPDTE=YYYYMMDD` enables both column projection '
    'and partition pruning. Fixing the exact 1,682,616 rows and 26 logical columns across CSV, Snappy Parquet, and '
    'ZSTD-3 Parquet (510 files; six exact parity combinations), **Figure 2** quantifies the storage-latency trade-off [E2].'
)
fig(
    'Fig2_storage',
    'Figure 2. Storage footprint and read-and-materialize latency across 170 Financials partitions (`storage_portable20261125a`). '
    'Panel A compares total bytes (decimal MB); Panel B compares medians of five fresh-process reads after one warmup '
    'under OS-managed caches (Full: 26 columns; Selective: `2010Q1–2019Q4`, 262,595 rows x 8 columns). Source: [E2].',
    6.25,
)
p(
    'ZSTD-3 compresses the four-decade panel to 153.57 MB—32.73% of CSVâ€™s 469.21 MB (a 67.27% reduction)—while Snappy '
    'requires 203.33 MB (43.33%). Conversely, Snappy achieves the lowest read latency for both full scans (2.11 s vs. '
    '2.46 s ZSTD-3 and 7.34 s CSV) and selective queries (0.091 s vs. 0.156 s ZSTD-3 and 0.441 s CSV). Because CSV '
    'cannot exploit native column projection, selective reads capture the combined format and access-path advantage. '
    'ZSTD-3 is therefore selected as the archival default, with Snappy reserved for latency-critical interactive caches.'
)

h('3.2 Three-Engine Computational Parity and Benchmark', 2)
p(
    'To evaluate single-node scalability fairly, pandas, Polars, and local PySpark executed an identical six-column '
    'observed-panel workload across all 170 treated partitions: sorting 1,682,616 unique `(CERT, REPDTE)` keys, '
    'detecting resumed reporting gaps via per-bank lag (33 gaps across 32 banks; 92 missing intervening quarters), and '
    'aggregating 170 quarterly quality summaries. All three engines passed 21/21 exact schema, key, null, and value '
    'parity checks without numeric tolerance (`portable20261125bbench`) [E3].'
)
fig(
    'Fig3_compute',
    'Figure 3. Portable three-engine benchmark on identical 170-quarter inputs (`portable20261125bbench`; five fresh '
    'processes + one warmup per engine). Panel A shows median wall time with min–max range bars; Panel B shows '
    '50 ms sampled process-tree peak RSS against the 15.63 GiB host ceiling. Source: [E3].',
    6.25,
)
e = D['processing']['by_engine']
booktabs_table(
    'Table 3. Portable three-engine wall time and sampled process-tree memory [E3]',
    ['Engine', 'Median Wall (s)', 'Min–Max Range (s)', 'Speedup vs. pandas', 'Median Peak RSS (MiB)'],
    [
        (
            'pandas' if n == 'pandas' else ('Polars' if n == 'polars' else 'PySpark (local)'),
            f"{e[n]['median_duration_ns'] / 1e9:.3f}",
            f"{e[n]['min_duration_ns'] / 1e9:.3f}–{e[n]['max_duration_ns'] / 1e9:.3f}",
            f"{(e['pandas']['median_duration_ns'] / e[n]['median_duration_ns']):.2f}x",
            f"{e[n]['median_peak_tree_rss_bytes'] / 2**20:,.1f}",
        )
        for n in ['pandas', 'polars', 'spark']
    ],
    [3.2, 2.8, 3.8, 3.0, 3.2],
    right_cols={1, 2, 3, 4},
)

h('3.3 Execution-Plan Interpretation and the Cluster Decision', 2)
p(
    'Inspecting the captured Polars optimized plans and PySpark physical plans (Table 4) explains the 17.66x wall-time '
    'advantage of Polars (4.116 s) over pandas (72.692 s) and 15.67x advantage over local PySpark (64.504 s) [E3]. '
    'Polars pushes column projections (`6/29`, `2/29`, `5/29`) directly into multi-threaded Arrow Parquet scans with '
    'zero inter-process serialization. By contrast, PySpark incurs JVM startup, `RangePartitioning` shuffle exchanges '
    'around window `lag(REPDTE)` and `HashAggregate`, and Arrow-backed `toPandas()` driver collection.'
)
booktabs_table(
    'Table 4. Physical execution-plan operators and resource bounds: Polars vs. local PySpark [E3]',
    ['Dimension', 'Polars (`LazyFrame`)', 'Local PySpark (`local[4]`, AQE enabled)'],
    [
        ('Scan projection', 'Projects 6/29 (observed), 2/29 (gaps), and 5/29 (quality) columns directly into columnar buffers.', 'Projects identical 6/29, 2/29, and 5/29 columns at `FileScan parquet` across all 170 explicit paths.'),
        ('Window & aggregation', 'Multi-threaded `Sort` + `shift(1).over(CERT)` for gaps; parallel hash group-by `REPDTE`.', '`RangePartitioning Exchange` + `Sort` for global order and `lag(REPDTE)`; partial/final `HashAggregate` shuffles.'),
        ('Adaptation & overhead', 'Single native Rust/Arrow process; zero JVM or driver-collection overhead.', '`AdaptiveSparkPlan` transitions `isFinalPlan=false -> true` (coalesced shuffle reads); JVM & 7-process tree overhead.'),
        ('Peak RAM utilization', 'Max sampled peak: 1,026.5 MiB (6.41% of 15.63 GiB host RAM).', 'Max sampled peak: 2,342.5 MiB (14.64% of 15.63 GiB host RAM).')
    ],
    [3.2, 6.4, 6.4],
)
p(
    '**Cluster decision:** A Spark cluster is empirically unjustified for this workload. Total treated input is only '
    '191.5 MB on disk, and local PySpark peaked at 14.64% of host RAM without disk spill. Distributing 191.5 MB across '
    'a cluster would introduce network shuffle and provisioning overhead exceeding local execution time. A cluster '
    'would be warranted only if a larger or multi-user workload exceeds laptop memory or latency SLAs after local Polars '
    'optimization, verified via end-to-end cost-and-shuffle benchmarking.'
)

# =========================================================================
# SECTION 4: Q2 DATA QUALITY & LOOK-AHEAD BIAS
# =========================================================================
h('4. Q2 — Data Quality, Exits, and Look-Ahead Control')
p(
    'Automated raw audits (`audit20261114b`) reveal structurally distinct missingness regimes across endpoints '
    '(**Figure 4**, Table 5) [E4]. An explicit JSON null (e.g., an uncollected historical ratio) is analytically '
    'different from a field omitted from an API payload, and a temporary filing gap differs from institutional failure. '
    'Rather than imputing zeros or dropping historical banks absent from a contemporary *Institutions* snapshot, the '
    'flag-only treatment preserves all 1,682,616 *Financials*, 4,117 *Failures*, and 27,835 *Institutions* rows verbatim.'
)
fig(
    'Fig4_quality',
    'Figure 4. Verified missingness and schema provenance across distinct FDIC endpoint denominators (`audit20261114b`). '
    'Teal bars represent explicit JSON nulls; slate denotes fields absent from API responses. Source: [E4].',
    6.1,
)
booktabs_table(
    'Table 5. Verified quality findings and non-destructive treatment rules [E4]',
    ['Quality Finding', 'Affected / Source Records', 'Analytical Treatment and Methodological Consequence'],
    [
        ('`P3ASSET` explicit null', '971,925 / 1,682,616 (57.8%)', '68 early quarters (`1984Q1–2000Q4`) are 100% null; excluded from the 7-feature model set.'),
        ('`COST` explicit null', '637 / 4,117 (15.5%)', 'Missingness preserved; development-event `COST` coverage audited separately (2,439/2,462 usable).'),
        ('`CERT` explicit null', '488 / 4,117 (11.9%)', '`1934–1976` records retain unique source `ID`; never fabricated or heuristically matched.'),
        ('`CHANGEC1` absent', '3,482 / 27,835 (12.5%)', 'Absence provenance preserved; never inferred as a merger or regulatory closure.')
    ],
    [3.8, 4.1, 8.1],
)

h('4.1 Exit Reconciliation and Retrospective Activity Cohort', 2)
p(
    'The observed panel preserves 1,682,616 unique (`CERT`, `REPDTE`) keys and logs 33 resumed reporting gaps across '
    '32 banks without synthesizing zero-filled quarters. Reconciling `RESTYPE`/`RESTYPE1` transaction codes (`repair20261115f`) '
    'distinguishes qualifying failures from open-bank assistance and charter-uncertain cases: 2,534 confirmed failure '
    'banks (revised down from a preliminary 2,775 count after reclassifying 241 assistance cases as ambiguous), 17,217 '
    'ambiguous exits, and 4,313 boundary-observed banks. Ordinary mergers are never labelled as failures [E4]. Over 166 '
    'prediction quarters (`1984Q1–2025Q2`), the retrospective cohort evaluates 24,064 observed `CERT`s (3,994,624 '
    'decisions): 1,660,450 eligible, 2,298,952 ineligible, and 35,222 uncertain (with 54,585 eligible decisions lacking '
    'prior report history). Because structural dates derive from current-vintage *Institutions*/*History* records, the '
    'modelling sample represents a reconstructed historical cohort rather than a certified real-time supervisory registry.'
)

h('4.2 Point-in-Time Availability, Feature Design, and Leakage Controls', 2)
p(
    'Because `REPDTE` denotes an accounting period-end rather than public release, a report becomes eligible as a '
    'predictor only after a conservative 120-calendar-day lag (or later when verified revision timing requires it). '
    'For bank *i* at prediction quarter-end *q*, the latest eligible report *r**(i, q) and forward failure label '
    '*Y_{i,q}* are formally defined as:'
)
eq('r*(i, q) = max { r : REPDTE(i, r) + 120d <= q  and  revision_available(i, r) <= q }', 1)
eq('Y_{i,q} = I( FAILDATE_i in (q, q + 4Q] ),   for q <= 2025Q2', 2)
p(
    'Observation quarters end at `2025Q2` to guarantee complete four-quarter follow-up through `2026Q2` [E4, E5]. Seven '
    'leak-free predictors are anchored strictly at *r**(i, q) (Table 6). `SimpleImputer` (median) and `StandardScaler` '
    'are wrapped in an unfitted scikit-learn pipeline and fitted exclusively within each development training fold. '
    'The temporal-leakage audit (`p44audit20261120a`) verified all 1,603,183 model rows across 166 quarters, recomputed '
    '24 fixed-seed bank histories from raw inputs, and confirmed invariance under future-report and held-out '
    'perturbations while flagging deliberate combined-period-fit negative controls [E5].'
)
booktabs_table(
    'Table 6. Seven point-in-time predictors anchored at eligible report r* [E5]',
    ['Predictor', 'Formula / Definition', 'Economic Rationale and Missingness Policy'],
    [
        ('Asset size', '`ASSET[r*]` (USD thousands)', 'Institution scale; single eligible report.'),
        ('Equity to assets', '`100 x EQ[r*] / ASSET[r*]`', 'Book capital buffer; null if `ASSET[r*] <= 0`; negative equity retained.'),
        ('Deposits to assets', '`100 x DEP[r*] / ASSET[r*]`', 'Core funding structure; null if `ASSET[r*] <= 0`.'),
        ('Reported ROA', '`ROA[r*]` (already %)', 'Annualized profitability; not rescaled by 100.'),
        ('PCA leverage ratio', '`RBC1AAJ[r*]` (already %)', 'Regulatory leverage adequacy; not rescaled by 100.'),
        ('Quarterly asset growth', '`100 x (ASSET[r*]/ASSET[r*-1] - 1)`', 'Balance-sheet growth; requires consecutive quarters.'),
        ('Four-quarter ROA mean', '`Mean(ROA[r*]...ROA[r*-3])`', 'Earnings persistence; requires four consecutive quarters.')
    ],
    [3.9, 5.3, 6.8],
)

# =========================================================================
# SECTION 5: Q3 MODELLING, VALIDATION & ECONOMIC DECISION
# =========================================================================
h('5. Q3 — Chronological Validation, Modelling, and Decision Analysis')
p(
    'The final model table comprises 1,603,183 bank-quarters: 9,934 positive (0.620%) and 1,593,249 negative. Split '
    'boundaries were frozen solely from event counts before any model fitting (**Figure 5**): development spans '
    '`1984Q1–2012Q4` (1,329,451 rows; 2,462 distinct failures), `2013Q1–2013Q4` forms a four-quarter embargo gap '
    '(eliminating outcome-window overlap), and test spans `2014Q1–2025Q2` (246,083 rows; 178 positive rows representing '
    '53 distinct failed banks, satisfying the `>= 50` failures criterion) [E6].'
)
fig(
    'Fig5_timeline',
    'Figure 5. Expanding-window development folds (`2001–2004`, `2005–2008`, `2009–2012`) with four-quarter embargo '
    'gaps and the untouched `2014Q1–2025Q2` final test period. Source: `dev20261121a` and `split20261120b` [E6].',
    6.25,
)

dev = D['development']
rows = []
for cid, label in [('logistic_baseline', 'Logistic baseline (`C=1`, `liblinear`)'), ('bounded_hist_gradient_boosting', 'Histogram gradient boosting (`seed=42`)')]:
    fs = [x for x in dev['fold_results'] if x['candidate_id'] == cid]
    rows.append((label, *[f"{x['average_precision']:.4f}" for x in fs], f"{statistics.mean(x['average_precision'] for x in fs):.4f}"))
booktabs_table(
    'Table 7. Development-fold Average Precision (AP) used for model selection [E6]',
    ['Candidate Model', 'Fold 1 (2001–04)', 'Fold 2 (2005–08)', 'Fold 3 (2009–12)', 'Mean Fold AP'],
    rows,
    [5.8, 2.5, 2.5, 2.5, 2.7],
    right_cols={1, 2, 3, 4},
)
sp_m = D['spark']['metrics']
sp_t = D['spark']['timings_seconds']
p(
    f"Because positive prevalence is below 1%, classification accuracy is uninformative; stepwise Average Precision (AP) "
    f"served as the primary development selection metric. Across three expanding folds (409,094 OOF rows), bounded "
    f"histogram gradient boosting (40 iterations, `lr=0.05`, `max_depth=3`, `max_leaf_nodes=15`, `L2=1`) outperformed "
    f"the logistic baseline on every fold (mean AP `0.1284` vs. `0.0794`, Table 7) and was selected. Separately, a "
    f"PySpark MLlib demonstration (`sparkdemo20261122a`) fitted `VectorAssembler`, `approxQuantile` median imputation, "
    f"`StandardScaler`, and `LogisticRegression` (`regParam=0.1`) on Fold 3 (1,173,760 train / 122,293 validation rows), "
    f"achieving `AP = {sp_m['average_precision']:.4f}`, `ROC-AUC = {sp_m['roc_auc']:.4f}`, and `Brier = {sp_m['brier']:.4f}` "
    f"(pipeline fit `{sp_t['spark_pipeline_fit']:.2f} s`, transform+collect `{sp_t['spark_transform_collect']:.2f} s`), "
    f"confirming reproducible distributed-API execution without replacing the selected scikit-learn estimator [E6]."
)

h('5.1 Economic Threshold Selection and Single-Use Final Test Evaluation', 2)
p(
    'To avoid arbitrary probability cutoffs or test-set tuning, the alarm threshold `tau` was selected exclusively on '
    'development OOF predictions (`threshold20261122c`) by minimizing the fold-level economic surrogate loss [E7]:'
)
eq('L_f(tau) = M_f * ( alpha * FN_f(tau) + beta * FP_f(tau) ),   with alpha = 0.10,  beta = 0.01', 3)
p(
    'Here `M_f` is the fold-training median of observed FDIC Estimated Loss (`COST`, in nominal USD thousands across '
    '2,439 nonnegative development events; `M = 10,821.559` on full development), `alpha = 0.10` is the assumed '
    'preventable loss fraction, and `beta = 0.01` is the assumed supervisory examination-cost fraction. Evaluating '
    '32,388 empirical policies across 9 sensitivity cells froze `SCORE >= 0.0321005`. After refitting once on all '
    '1,329,451 development rows, a durable registry recorded one test-access transition (`final20261122a`) over the '
    '246,083 untouched test rows (0.07233% positive prevalence), yielding **Figure 6** [E7].'
)
fig(
    'Fig6_evaluation',
    'Figure 6. Locked final-test evaluation (`2014Q1–2025Q2`; `final20261122a`). Panel A: Precision–Recall curve '
    '(`AP = 0.1730` vs. `0.000723` prevalence). Panel B: ROC curve (`AUC = 0.8770`). Panel C: Confusion counts at '
    '`SCORE >= 0.0321005` and nominal surrogate cost (`0.9556` vs. `0.7828` USDk/bank-quarter). Source: [E7].',
    6.3,
)
p(
    'On the final test set, the model achieves `AP = 0.172989` and `ROC-AUC = 0.876992`, demonstrating informative '
    'risk ranking. At the locked threshold (`TP=109`, `FP=1,483`, `TN=244,422`, `FN=69`), row recall is 61.24% and '
    'precision is 6.85% (alert rate 0.65%), while at the institution-event level 38 of 53 failed banks (71.70%) trigger '
    'at least one prior alarm. Although `Brier = 0.0006591` beats the all-zero baseline (`0.0007233`), extreme imbalance '
    'prevents interpreting raw scores as calibrated probabilities. Crucially, the locked policy yields a nominal '
    'surrogate cost of `0.955582` USDk/bank-quarter—22.08% *higher* than never alarming (`0.782759`)—because 1,483 '
    'false alarms (`160,483.72` USDk) outweigh 69 missed positives (`74,668.76` USDk). Strong ranking discrimination '
    'thus coexists with an economically inferior fixed threshold, underscoring why honest reporting prohibits '
    'post-test threshold re-optimization [E7].'
)

# =========================================================================
# SECTION 6: TRANSFERABILITY TO VIETNAMESE BANKING SUPERVISION
# =========================================================================
h('6. Transferability to Vietnamese Banking Supervision')
p(
    'While the empirical evidence is US-based, the pipelineâ€™s engineering discipline transfers directly to the State '
    'Bank of Vietnam (SBV) supervisory context—whereas US model coefficients, the 120-day lag, FDIC `COST` scales, and '
    'the `0.0321005` threshold do not [1]. Under Vietnamese prudential frameworks such as SBV Circular 41/2016/TT-NHNN '
    '(capital adequacy ratios) and statutory early-intervention / special-control regimes, supervisory filings remain '
    'non-public and institutional distress often resolves via restructuring or state-led consolidation rather than '
    'open-market liquidation. Deploying this architecture locally therefore requires authorized institutional governance '
    'and empirical verification of every boundary in Table 8, without assuming unverified SBV schemas.'
)
booktabs_table(
    'Table 8. Transferable pipeline practices vs. local SBV verification requirements',
    ['Transferable Methodological Practice', 'Local Verification Required for Vietnamese Deployment'],
    [
        ('Versioned ingestion & SHA-256 manifests', 'Authorized SBV data governance, historical filing extracts, and revision retention.'),
        ('Stable bank & event identifiers', 'Distinguish ordinary M&A from special control, compulsory transfer, and restructuring.'),
        ('As-of predictor joins (Eq. 1)', 'Actual regulatory submission/revision timestamps; verify reporting lag empirically vs. 120 days.'),
        ('Chronological split & embargo gaps', 'Measure distinct local distress events and complete follow-up before freezing holdout dates.'),
        ('Cost-sensitive threshold (Eq. 3)', 'Calibrate to SBV on-site inspection capacity, intervention efficacy, and VND loss units.'),
        ('Single-node Polars vs. cluster scaling', 'Benchmark on local filing volume, concurrent supervisory users, and hardware SLAs.')
    ],
    [5.6, 10.4],
)

# =========================================================================
# SECTION 7: SUPERVISORY IMPLICATIONS, LIMITATIONS & FUTURE WORK
# =========================================================================
h('7. Supervisory Implications, Limitations, and Future Research')
p(
    '**Synthesis of Q1–Q3:** First (Q1), typed quarter-partitioned ZSTD-3 Parquet and lazy Polars execution handle four '
    'decades of supervisory filings on a standard laptop with exact PySpark parity and minimal RAM headroom, rendering '
    'a Spark cluster unnecessary at this scale. Second (Q2), non-destructive quality flagging, transaction-type exit '
    'reconciliation, and a 120-day point-in-time cutoff eliminate known look-ahead channels in predictors. Third (Q3), '
    'the pipeline supports an auditable chronological validation design, but extreme class imbalance (`0.072%` test '
    'prevalence) causes a cost-selected threshold to generate excessive false alarms relative to never alarming.'
)
p(
    '**Methodological Limitations & Future Work:** Four constraints qualify these findings: (i) historical public-release '
    'vintages are unobserved, forcing reliance on a fixed 120-day assumption; (ii) the activity cohort is reconstructed '
    'retrospectively from current-vintage structural tables; (iii) `COST` reflects nominal DIF resolution loss rather '
    'than causal preventable loss; and (iv) the 46-quarter test window spans shifting macroeconomic regimes without '
    'post-hoc recalibration. Future supervisory research should prioritize logging true filing timestamps, fitting '
    'fold-local probability calibration (e.g., isotonic/Platt scaling), and evaluating event-level alert capacity '
    'rather than row-level binary triggers.'
)

# =========================================================================
# SECTION 8: REPRODUCIBILITY & REFERENCES
# =========================================================================
h('8. Reproducibility and Archival Availability')
p(
    'Captured-vintage portable replays regenerate the pipeline from raw audit through storage, three-engine processing, '
    'cohort/label construction, feature tables, leakage audit, split, development validation, Spark MLlib demo, COST '
    'audit, and threshold selection with pinned dependencies and fixed `seed=42`; the clean sequential suite passed '
    '500/500 tests with zero skips [E8]. Final test scores (`final20261122a`) are verified read-only from immutable '
    'persisted artifacts to preserve the single-use test guarantee. Raw and derived datasets are excluded from version '
    'control via `.gitignore`; full archival reproducibility from a fresh clone requires committing the local `src/`, '
    '`tests/`, `reporting/`, and `config/` trees to the public repository.'
)

h('References', 2)
refs = [
    '[1] National Economics University. *BFT65B Big Data Analytics in Finance: Final Individual Report — A Big-Data Pipeline for Bank Failure Early Warning*. Authoritative course assignment, pp. 1–3.',
    '[2] Federal Deposit Insurance Corporation (FDIC). *BankFind Suite API Documentation and Financials, Failures, and Institutions YAML Dictionaries*. https://api.fdic.gov/banks/docs/ (accessed 25 November 2026).',
    '[3] Federal Deposit Insurance Corporation (FDIC). *Failures Data Dictionary (failure_properties.yaml) and Resolutions Handbook*. https://www.fdic.gov/bank-failures/resolutions-handbook.pdf.',
    '[4] scikit-learn developers. *average_precision_score (v1.6.1) Documentation*. https://scikit-learn.org/1.6/modules/generated/sklearn.metrics.average_precision_score.html.',
    '[5] Nguyen Cong Hoan. *BFT65B Implementation, Evidence Register, and Reproducibility Documentation*. https://github.com/nguyenconghoan5305-code/Big_Data_11230965.',
    '[6] Course-provided `fdic_download_starter.py`. Optional acquisition reference; superseded by typed Arrow storage and manifest recovery.',
    '[7] AI Assistance Disclosure: OpenAI Codex and Antigravity (Google DeepMind) supported pipeline engineering, verification scripting, and editorial refinement; all numerical claims are verified directly against immutable saved project artifacts.'
]
for ref in refs:
    rp = p(ref, justify=True, space_after=2.5)
    for run in rp.runs:
        run.font.size = Pt(9.0)

# =========================================================================
# APPENDIX A: BENCHMARK RECONCILIATION
# =========================================================================
page('Appendix A: Benchmark Reconciliation')
p(
    'An earlier five-page review draft (`BFT65B_report_draft.pdf`) inadvertently mixed measurements from two valid, '
    'parity-verified benchmark sessions: its opening tables used the November 25 portable replays, whereas its Q1 prose '
    'cited the November 15 historical runs. To guarantee strict internal consistency, the main text reports only the '
    'November 25 portable family; both executions are reconciled in Tables A1 and A2.'
)
old = D['processing_old']['three_engine_run']['by_engine']
new = D['processing']['by_engine']
booktabs_table(
    'Table A1. Processing benchmark medians across historical vs. portable executions',
    ['Engine', 'Historical Nov 15 (`panelbench20261115b`)', 'Portable Nov 25 (`portable20261125bbench`)'],
    [
        (
            'pandas' if n == 'pandas' else ('Polars' if n == 'polars' else 'PySpark (local)'),
            f"{old[n]['median_duration_ns'] / 1e9:.3f} s",
            f"{new[n]['median_duration_ns'] / 1e9:.3f} s",
        )
        for n in ['pandas', 'polars', 'spark']
    ],
    [4.0, 6.0, 6.0],
    right_cols={1, 2},
)
oldst = {(x['format'], x['workload']): x for x in D['storage_old']['summary']}
srows = []
for x in D['storage']['summary']:
    srows.append((x['format'].upper(), x['workload'], f"{oldst[x['format'], x['workload']]['median_seconds']:.6f} s", f"{x['median_seconds']:.6f} s"))
booktabs_table(
    'Table A2. Storage read-and-materialize medians across historical vs. portable executions',
    ['Format', 'Workload', 'Historical Nov 15 (`task_1_6_storage`)', 'Portable Nov 25 (`storage_portable20261125a`)'],
    srows,
    [2.8, 3.0, 5.1, 5.1],
    right_cols={2, 3},
)
p(
    'Both run families execute identical logical contracts and pass exact output parity. Wall-time differences reflect '
    'separate directory roots, source snapshots, and uncontrolled OS cache/background conditions rather than a '
    'controlled regression. Storage timers measure read-and-materialize only, whereas processing timers span interpreter '
    'startup, transformations, and three ZSTD Parquet writes.'
)

# =========================================================================
# APPENDIX B: ENVIRONMENT, EVIDENCE & RUBRIC TRACEABILITY
# =========================================================================
page('Appendix B: Environment, Evidence, and Rubric Traceability')
p(
    '**Execution Environment:** All portable benchmarks and clean-suite verifications ran on Windows 11 '
    '(build `10.0.26200`, 16 logical processors, 15.63 GiB RAM) using Python `3.11.9`, `pandas 2.2.3`, `polars 1.35.2`, '
    '`pyarrow 18.1.0`, `pyspark 3.5.7` (Temurin JDK `17.0.21`, Hadoop `3.3.5` natives), `scikit-learn 1.6.1`, and '
    '`numpy 2.1.3`, pinned via `requirements.txt` and `requirements-lock-win-py311.txt`. Every quantitative claim maps '
    'to immutable SHA-256 inventories in `EVIDENCE_REGISTER.csv` and `outputs/academic_report/source_hashes.json`.'
)
booktabs_table(
    'Table B1. Evidence labels and immutable run identifiers',
    ['Label', 'Accepted Run IDs and Artifacts', 'Report Placement'],
    [
        ('E1', '`real20261111b`; `task_1_5_full_real20261111b_acceptance.json`', 'Tables 1–2; Sections 1–2'),
        ('E2', '`storage_portable20261125a`; `timings.json`, `frozen_spec.json`', 'Figure 2; Section 3.1; Table A2'),
        ('E3', '`portable20261125apolars`, `portable20261125bbench`', 'Figure 3; Tables 3–4; Sections 3.2–3.3; Table A1'),
        ('E4', '`audit20261114b`, `treatment20261114a`, `panel20261115b`, `repair20261115f`, `pit20261115f`, `membership20261120a`', 'Figure 4; Table 5; Section 4'),
        ('E5', '`featuretable20261120a/b`, `p44audit20261120a`, `FEATURE_DICTIONARY.md`', 'Table 2; Table 6; Section 4.2'),
        ('E6', '`split20261120b`, `dev20261121a/b`, `sparkdemo20261122a`', 'Figure 5; Table 7; Section 5'),
        ('E7', '`costaudit20261122d`, `threshold20261122c`, `final20261122a`', 'Figure 6; Section 5.1'),
        ('E8', 'Portable replay manifests, `p6bfull.xml` (500/500), `P6_REPRODUCIBILITY.md`', 'Section 8; Appendix B')
    ],
    [1.4, 9.6, 5.0],
)
booktabs_table(
    'Table B2. Assignment rubric coverage matrix',
    ['Rubric ID', 'Verification Gate and Evidence', 'Report Placement'],
    [
        ('P1.1–P1.3', '172-job acquisition gate; 26-field schema; 510-file storage parity & 30 reads', 'Sections 2–3.1; Tables 1–2, A2; Fig 2'),
        ('P2.1–P2.4', 'Automated audit; flag-only treatment; transaction repair; PIT 120-day & revision gates', 'Section 4; Fig 4; Table 5'),
        ('P3.1–P3.4', 'Polars/pandas/Spark exact parity (21 checks); 15 timed runs; AQE plan & capacity analysis', 'Sections 3.2–3.3; Fig 3; Tables 3–4, A1'),
        ('P4.1–P4.2', '7-feature leak-free table (1,603,183 rows); train-only sklearn pipeline; perturbation tests', 'Section 4.2; Table 6'),
        ('P5.1–P5.4', 'Chronological split (53 test failures); 3 forward folds; MLlib demo; COST/threshold; 1-use test', 'Section 5; Figs 5–6; Table 7'),
        ('P6.1', 'Evidence-based Q1–Q3 synthesis and Vietnam SBV transferability analysis', 'Sections 6–7; Table 8'),
        ('R.1; S.1–S.4', 'Clean 500/500 suite; portable replay; citations; archival GitHub status', 'Section 8; Appendix A–B')
    ],
    [2.4, 9.0, 4.6],
)

out_docx = OUT / 'paper.docx'
doc.save(out_docx)
texts = [' '.join(el.itertext()) for el in doc.element.body]
(OUT / 'report_content.txt').write_text('\n'.join(texts), encoding='utf-8')
print('Saved:', out_docx)

