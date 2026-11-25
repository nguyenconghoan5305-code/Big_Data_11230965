"""Generate 6 consolidated, publication-grade 300-DPI figures (PDF/SVG/PNG).

Run extract_report_evidence.py with the pinned project environment first.
Run this script with the isolated report Python environment (reportlab, pypdfium2).
No model inference, threshold selection, or individual record export occurs.
"""
from pathlib import Path
import json
from reportlab.graphics.shapes import Drawing, Rect, String, Line, PolyLine, Circle
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/academic_report'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
D = json.loads((OUT / 'plot_evidence.json').read_text(encoding='utf-8'))

# Unified academic color palette
NAVY = '#102A43'
TEAL = '#1D6F72'
SLATE = '#627D98'
ALERT = '#B85038'
GRID = '#E6ECF1'
LIGHT = '#F1F5F9'
TEXT = '#1F2933'
MUTED = '#52606D'

FONT_REG = 'Times-Roman'
FONT_BOLD = 'Times-Bold'


def txt(d, x, y, s, size=9.5, color=TEXT, anchor='start', bold=False):
    d.add(String(
        x, y, str(s),
        fontName=FONT_BOLD if bold else FONT_REG,
        fontSize=size,
        fillColor=HexColor(color),
        textAnchor=anchor,
    ))


def ln(d, x1, y1, x2, y2, color=SLATE, width=0.7, dash=None):
    kw = {'strokeColor': HexColor(color), 'strokeWidth': width}
    if dash:
        kw['strokeDashArray'] = dash
    d.add(Line(x1, y1, x2, y2, **kw))


def box(d, x, y, w, h, fill, stroke=None, sw=0.6):
    d.add(Rect(
        x, y, w, h,
        fillColor=HexColor(fill) if fill else None,
        strokeColor=HexColor(stroke) if stroke else None,
        strokeWidth=sw if stroke else 0,
    ))


def save_fig(d, name):
    renderPDF.drawToFile(d, str(FIG / f'{name}.pdf'))
    renderSVG.drawToFile(d, str(FIG / f'{name}.svg'))
    pdf = pdfium.PdfDocument(str(FIG / f'{name}.pdf'))
    pdf[0].render(scale=300 / 72).to_pil().save(FIG / f'{name}.png', dpi=(300, 300))
    pdf.close()


def hbars(d, x, y, w, h, labels, values, title, unit, colors=None, extras=None, err_ranges=None, ref_line=None):
    txt(d, x, y + h + 16, title, 10.5, NAVY, bold=True)
    max_val = max(values)
    if err_ranges:
        max_val = max(max_val, max(hi for _, hi in err_ranges))
    if ref_line:
        max_val = max(max_val, ref_line[0])
    maxv = max_val * 1.32
    for j in range(5):
        xx = x + w * j / 4
        ln(d, xx, y, xx, y + h, GRID, 0.6)
        txt(d, xx, y - 12, f'{maxv * j / 4:.1f}', 8.2, MUTED, 'middle')
    ln(d, x, y, x, y + h, SLATE, 0.8)
    ln(d, x, y, x + w, y, SLATE, 0.8)
    n = len(values)
    slot = h / n
    bh = min(17, slot * 0.55)
    for i, (lab, v) in enumerate(zip(labels, values)):
        yy = y + h - (i + 0.5) * slot - bh / 2
        txt(d, x - 7, yy + bh / 2 - 3, lab, 9.0, NAVY, 'end')
        bw = w * v / maxv
        c = (colors or [TEAL] * n)[i]
        box(d, x, yy, bw, bh, c)
        if err_ranges:
            lo, hi = err_ranges[i]
            xlo, xhi = x + w * lo / maxv, x + w * hi / maxv
            ymid = yy + bh / 2
            ln(d, xlo, ymid, xhi, ymid, NAVY, 1.0)
            ln(d, xlo, ymid - 3.5, xlo, ymid + 3.5, NAVY, 1.0)
            ln(d, xhi, ymid - 3.5, xhi, ymid + 3.5, NAVY, 1.0)
            anchor_x = xhi + 5
        else:
            anchor_x = x + bw + 5
        txt(d, anchor_x, yy + bh / 2 - 3, extras[i] if extras else f'{v:,.2f}', 8.6, TEXT)
    if ref_line:
        rx = x + w * ref_line[0] / maxv
        ln(d, rx, y, rx, y + h, ALERT, 1.0, dash=[3, 2])
        txt(d, rx - 3, y + h + 4, ref_line[1], 7.8, ALERT, 'end')
    txt(d, x + w / 2, y - 25, unit, 8.5, MUTED, 'middle')


# =========================================================================
# Figure 1: End-to-End Pipeline & Leakage-Control Architecture
# =========================================================================
d = Drawing(600, 228)
stages = [
    ('1. DATA PREPARATION', NAVY, ['FDIC REST API', 'Immutable Parquet', 'Quality Audit', 'Observed Panel']),
    ('2. TIME & COHORT DISCIPLINE', TEAL, ['PIT As-of Features', 'Retrospective Cohort', '4-Quarter Labels', 'Chronological Split']),
    ('3. MODELLING & EVALUATION', SLATE, ['Fold-Only Fitting', 'OOF Model + Threshold', 'Locked Test Eval', 'Cost Interpretation']),
]
for j, (stitle, col, items) in enumerate(stages):
    y = 172 - j * 68
    txt(d, 8, y + 31, stitle, 9.2, col, bold=True)
    for i, item in enumerate(items):
        x = 8 + i * 149
        box(d, x, y, 136, 25, LIGHT, col, 0.7)
        txt(d, x + 68, y + 9.5, item, 8.8, NAVY, 'middle')
        if i < 3:
            ln(d, x + 136, y + 12.5, x + 148, y + 12.5, col, 0.9)
            txt(d, x + 148, y + 9.2, '>', 9.0, col, 'end', bold=True)
for y_top, col in [(172, NAVY), (104, TEAL)]:
    ln(d, 523, y_top, 523, y_top - 12, col, 0.8)
    ln(d, 523, y_top - 12, 76, y_top - 12, col, 0.8)
    ln(d, 76, y_top - 12, 76, y_top - 18, col, 0.8)
    txt(d, 76, y_top - 22, 'v', 8.0, col, 'middle', bold=True)

ln(d, 299, 30, 299, 73, ALERT, 1.2, dash=[3, 2])
txt(d, 299, 77, 'TEST LOCK', 7.8, ALERT, 'middle', bold=True)
txt(d, 8, 6, 'Safeguards: SHA-256 manifests  |  REPDTE + 120d <= q  |  Train-fold-only fit  |  Single-use test registry', 8.5, TEAL, bold=True)
save_fig(d, 'Fig1_pipeline')


# =========================================================================
# Figure 2: Combined 2-Panel Storage Footprint & Read Latency
# =========================================================================
formats = ['CSV', 'Snappy', 'ZSTD-3']
full = [x for x in D['storage']['summary'] if x['workload'] == 'full']
sel = [x for x in D['storage']['summary'] if x['workload'] == 'selective']
sizes_mb = [x['stored_bytes'] / 1e6 for x in full]

d = Drawing(600, 195)
hbars(
    d, 68, 42, 200, 118, formats, sizes_mb,
    'Panel A: Stored Footprint (170 Quarters)', 'Decimal MB (1 MB = 10^6 bytes)',
    colors=[SLATE, TEAL, NAVY],
    extras=[f'{v:.1f} MB ({sizes_mb[0]/v:.2f}x)' for v in sizes_mb],
)

# Panel B: Grouped horizontal bars (Full vs Selective read times)
bx, by, bw, bh = 372, 42, 200, 118
txt(d, bx, by + bh + 16, 'Panel B: Median Read Latency', 10.5, NAVY, bold=True)
f_times = [r['median_seconds'] for r in full]
s_times = [r['median_seconds'] for r in sel]
maxt = max(f_times) * 1.32
for j in range(5):
    xx = bx + bw * j / 4
    ln(d, xx, by, xx, by + bh, GRID, 0.6)
    txt(d, xx, by - 12, f'{maxt * j / 4:.1f}', 8.2, MUTED, 'middle')
ln(d, bx, by, bx, by + bh, SLATE, 0.8)
ln(d, bx, by, bx + bw, by, SLATE, 0.8)
slot = bh / 3
for i, (lab, ft, st) in enumerate(zip(formats, f_times, s_times)):
    ymid = by + bh - (i + 0.5) * slot
    txt(d, bx - 7, ymid - 3, lab, 9.0, NAVY, 'end')
    box(d, bx, ymid + 1, bw * ft / maxt, 10, NAVY)
    txt(d, bx + bw * ft / maxt + 4, ymid + 3, f'{ft:.2f} s', 8.0, TEXT)
    box(d, bx, ymid - 11, bw * st / maxt, 10, TEAL)
    txt(d, bx + bw * st / maxt + 4, ymid - 9, f'{st:.3f} s', 8.0, TEAL)
box(d, bx + 95, by + bh + 2, 8, 7, NAVY)
txt(d, bx + 106, by + bh + 2, 'Full (26 cols)', 7.8, TEXT)
box(d, bx + 158, by + bh + 2, 8, 7, TEAL)
txt(d, bx + 169, by + bh + 2, 'Selective (8 cols)', 7.8, TEAL)
txt(d, bx + bw / 2, by - 25, 'Seconds (median of 5 fresh reads)', 8.5, MUTED, 'middle')
save_fig(d, 'Fig2_storage')


# =========================================================================
# Figure 3: Combined 2-Panel Compute Engine Benchmark
# =========================================================================
eng = D['processing']['by_engine']
names = ['pandas', 'polars', 'spark']
labels_e = ['pandas', 'Polars', 'PySpark']
med_s = [eng[n]['median_duration_ns'] / 1e9 for n in names]
rng_s = [(eng[n]['min_duration_ns'] / 1e9, eng[n]['max_duration_ns'] / 1e9) for n in names]
med_gb = [eng[n]['median_peak_tree_rss_bytes'] / 2**30 for n in names]

d = Drawing(600, 195)
hbars(
    d, 68, 42, 200, 118, labels_e, med_s,
    'Panel A: End-to-End Wall Time (Min–Max)', 'Seconds (5 fresh processes + 1 warmup)',
    colors=[SLATE, TEAL, NAVY],
    extras=[f'{v:.2f} s' for v in med_s],
    err_ranges=rng_s,
)
hbars(
    d, 372, 42, 200, 118, labels_e, med_gb,
    'Panel B: Sampled Process-Tree Peak RSS', 'GiB (median of 5 peaks; 50 ms sampling)',
    colors=[SLATE, TEAL, NAVY],
    extras=[f'{v:.2f} GiB ({100 * v / 15.63:.1f}%)' for v in med_gb],
)
txt(d, 472, 4, 'Host physical RAM ceiling: 15.63 GiB (Spark max peak: 14.64%)', 8.0, MUTED, 'middle')
save_fig(d, 'Fig3_compute')


# =========================================================================
# Figure 4: Source Missingness & Schema Provenance
# =========================================================================
quality = D['quality']['datasets']
qrows = [
    ('Financials P3ASSET (null)', 'financials', 'P3ASSET', 'explicit_null', TEAL),
    ('Failures COST (null)', 'failures', 'COST', 'explicit_null', TEAL),
    ('Institutions CHANGEC1 (absent)', 'institutions', 'CHANGEC1', 'absent', SLATE),
    ('Failures CERT (null)', 'failures', 'CERT', 'explicit_null', TEAL),
]
qvals, qextras, qcols = [], [], []
for _, ds, fld, key, col in qrows:
    c = quality[ds]['field_counts'][fld]
    pct = 100.0 * c[key] / c['denominator']
    qvals.append(pct)
    qextras.append(f"{pct:.1f}%  ({c[key]:,} / {c['denominator']:,})")
    qcols.append(col)

d = Drawing(600, 185)
hbars(
    d, 165, 38, 300, 115, [r[0] for r in qrows], qvals,
    'Source Missingness and Schema Provenance by Endpoint Denominator',
    'Percent of records within the named FDIC source table',
    colors=qcols, extras=qextras,
)
save_fig(d, 'Fig4_quality')


# =========================================================================
# Figure 5: Expanding-Window Chronological Validation & Embargo Design
# =========================================================================
d = Drawing(600, 205)
left, width = 92, 488


def yr(s):
    return int(s[:4]) + (int(s[5:7]) - 1) / 12


def seg(y, start, end, col):
    x1 = left + (yr(start) - 1984) / 42 * width
    w = (yr(end) - yr(start) + 0.25) / 42 * width
    box(d, x1, y, w, 14, col)


txt(d, 10, 185, 'Expanding-Window Development Folds, 4-Quarter Embargo Gaps, and Locked Test', 10.5, NAVY, bold=True)
for i, f in enumerate(D['development']['folds']):
    y = 148 - i * 31
    txt(d, 82, y + 3.5, f'Fold {i + 1}', 9.2, NAVY, 'end', bold=True)
    seg(y, f['train_start'], f['train_end'], TEAL)
    seg(y, f['gap_start'], f['gap_end'], SLATE)
    seg(y, f['validation_start'], f['validation_end'], NAVY)

txt(d, 82, 56, 'Final Split', 9.2, NAVY, 'end', bold=True)
seg(53, '1984-03-31', '2012-12-31', TEAL)
seg(53, '2013-03-31', '2013-12-31', SLATE)
seg(53, '2014-03-31', '2025-06-30', ALERT)

for v in [1984, 1992, 2000, 2004, 2008, 2013, 2020, 2025]:
    txt(d, left + (v - 1984) / 42 * width, 34, v, 8.2, MUTED, 'middle')

legend = [
    ('Training / development', TEAL),
    ('4-quarter embargo gap', SLATE),
    ('Development validation', NAVY),
    ('Locked final test', ALERT),
]
for i, (label, col) in enumerate(legend):
    lx = 12 + i * 148
    box(d, lx, 9, 9, 9, col)
    txt(d, lx + 13, 10.5, label, 8.2, TEXT)
save_fig(d, 'Fig5_timeline')


# =========================================================================
# Figure 6: Combined 3-Panel Final Test Evaluation
# =========================================================================
r = D['final']['result']
d = Drawing(600, 225)


def draw_curve(dx, dy, dw, dh, pts, title, xlab, ylab, marker):
    txt(d, dx, dy + dh + 16, title, 9.8, NAVY, bold=True)
    for a in [0, 0.25, 0.5, 0.75, 1.0]:
        ln(d, dx, dy + a * dh, dx + dw, dy + a * dh, GRID, 0.5)
        txt(d, dx - 4, dy + a * dh - 3, f'{a:g}', 7.5, MUTED, 'end')
        txt(d, dx + a * dw, dy - 10, f'{a:g}', 7.5, MUTED, 'middle')
    ln(d, dx, dy, dx + dw, dy, SLATE, 0.8)
    ln(d, dx, dy, dx, dy + dh, SLATE, 0.8)
    flat = []
    for a, b in pts:
        flat.extend([dx + a * dw, dy + b * dh])
    d.add(PolyLine(flat, strokeColor=HexColor(TEAL), strokeWidth=1.2, fillColor=None))
    d.add(Circle(dx + marker[0] * dw, dy + marker[1] * dh, 3.2, fillColor=HexColor(ALERT), strokeColor=None))
    txt(d, dx + dw / 2, dy - 22, xlab, 8.2, MUTED, 'middle')
    txt(d, dx, dy + dh + 5, ylab, 7.8, MUTED)


draw_curve(32, 44, 152, 138, D['curves']['pr'], f"Panel A: PR (AP = {r['average_precision']:.4f})", 'Recall', 'Precision', (r['recall'], r['precision']))
ln(d, 32, 44 + r['prevalence'] * 138, 184, 44 + r['prevalence'] * 138, SLATE, 0.7, dash=[2, 2])

draw_curve(220, 44, 152, 138, D['curves']['roc'], f"Panel B: ROC (AUC = {r['roc_auc']:.4f})", 'False-positive rate', 'True-positive rate', (r['false_positive_rate'], r['recall']))
ln(d, 220, 44, 372, 182, SLATE, 0.7, dash=[2, 2])

# Panel C: Confusion matrix + surrogate cost comparison
cx, cy = 408, 44
txt(d, cx, 198, 'Panel C: Locked Policy & Cost', 9.8, NAVY, bold=True)
txt(d, cx + 62, 179, 'Alarm', 8.0, NAVY, 'middle', bold=True)
txt(d, cx + 142, 179, 'No alarm', 8.0, NAVY, 'middle', bold=True)
cm = [('Pos', [r['TP'], r['FN']]), ('Neg', [r['FP'], r['TN']])]
for ri, (rlab, vals) in enumerate(cm):
    yy = 144 - ri * 34
    txt(d, cx + 18, yy + 11, rlab, 8.0, NAVY, 'end', bold=True)
    for ci, v in enumerate(vals):
        xx = cx + 24 + ci * 78
        box(d, xx, yy, 74, 28, '#DCEAEB' if (ri + ci) % 2 == 0 else LIGHT, SLATE, 0.5)
        txt(d, xx + 37, yy + 10, f'{v:,}', 10.5, NAVY, 'middle', bold=True)

# Cost comparison bars below confusion matrix
txt(d, cx, 88, 'Surrogate cost (USDk / bank-qtr):', 8.2, NAVY, bold=True)
c_never = r['never_alarm_surrogate_per_row_nominal_thousands_usd']
c_lock = r['reference_surrogate_per_row_nominal_thousands_usd']
maxc = c_lock * 1.38
for idx, (clab, cv, col, ext) in enumerate([
    ('Never alarm', c_never, TEAL, f'{c_never:.4f}'),
    ('Locked policy', c_lock, ALERT, f'{c_lock:.4f} (+22.1%)'),
]):
    yy = 62 - idx * 22
    txt(d, cx + 56, yy + 4, clab, 7.8, NAVY, 'end')
    box(d, cx + 60, yy, 92 * cv / maxc, 13, col)
    txt(d, cx + 63 + 92 * cv / maxc, yy + 4, ext, 7.6, TEXT)

txt(d, 32, 8, 'Terracotta markers denote locked threshold (SCORE >= 0.0321005) selected on development OOF; no test tuning.', 8.2, ALERT)
save_fig(d, 'Fig6_evaluation')
print('Generated 6 consolidated 300-DPI figures (PDF/SVG/PNG).')
