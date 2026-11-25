"""Draw the ten report panels from aggregate measurements with no model or raw-data access."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/"submission/figures"
FIG.mkdir(parents=True,exist_ok=True)
evidence = json.loads((ROOT/"submission/aggregate_evidence.json").read_text(encoding="utf-8"))
storage = {"summary":evidence["storage"]}
m, folds, bounds, curves = (evidence[key] for key in ("final","folds","split","curves"))
NAVY, TEAL, GREY, LIGHT = "#172f43", "#237f80", "#617487", "#edf4f6"
def canvas(height=460):
    im = Image.new("RGB", (1500, height), "white")
    d = ImageDraw.Draw(im)
    def label(x, y, value, size=34, fill=NAVY, **kw):
        d.text((x, y), str(value), font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size), fill=fill, **kw)
    return im, d, label

def bars(name, labels, values, unit, axis_max=None, fmt=".2f"):
    im, d, label = canvas(140 + len(labels)*90)
    lo, hi = 390, 1245
    end = axis_max or max(values)*1.12
    for i in range(5):
        x = lo + (hi-lo)*i/4
        d.line((x, 14, x, 25+len(labels)*90), fill="#d8e2e8", width=2)
        label(x, 35+len(labels)*90, f"{end*i/4:g}", 27, anchor="mt")
    for i,(key,v) in enumerate(zip(labels,values)):
        y = 38+i*90
        label(5,y,key,32)
        x = lo+(hi-lo)*v/end
        d.rectangle((lo,y,x,y+44),fill=[TEAL,"#4c8eae",GREY][i%3])
        label(x+14,y+3,format(v,fmt),29)
    label((lo+hi)/2,84+len(labels)*90,unit,29,anchor="mt")
    im.save(FIG / f"{name}.png")

fmts = list(dict.fromkeys(x["format"] for x in storage["summary"]))
display = {"csv":"CSV","snappy":"Parquet Snappy","zstd":"Parquet ZSTD-3"}
s = {(x["format"],x["workload"]):x for x in storage["summary"]}
bars("storage_size",[display[x] for x in fmts],[s[x,"full"]["stored_bytes"]/1e6 for x in fmts],"Stored size (decimal MB)",500)
for role, end in [("full",8),("selective",.5)]:
    bars("read_"+role,[display[x] for x in fmts],[s[x,role]["median_seconds"] for x in fmts],"Median read time (seconds)",end,".3f")
engines = ["pandas","polars","spark"]
e = evidence["engines"]
bars("processing_time",["pandas","Polars","Local PySpark"],[e[x]["median_duration_ns"]/1e9 for x in engines],"Median end-to-end wall time (seconds)",80)
bars("processing_memory",["pandas","Polars","Local PySpark"],[e[x]["median_peak_tree_rss_bytes"]/2**20 for x in engines],"Median sampled peak process-tree RSS (MiB)",2600,".0f")
bars("quality",["P3ASSET explicit null","CHANGEC1 absent","Failure CERT null"],[x["count"] / x["denominator"] * 100 for x in evidence["quality"]],"Share of records in the relevant source table (%)",100)
bars("cost",["Locked alarm policy","Never alarm"],[m["reference_surrogate_per_row_nominal_thousands_usd"],m["never_alarm_surrogate_per_row_nominal_thousands_usd"]],"Nominal USD thousands per test bank-quarter",1.1,".3f")

im,d,label=canvas(510)
stages=[
    ("01  DATA PREPARATION",["FDIC snapshots → Quarters, source IDs and SHA-256","Audit and treatment → Observed bank-quarter panel"]),
    ("02  CHRONOLOGICAL DEVELOPMENT",["Eligible history + four-quarter outcomes → one-to-one join","Train within folds → Select model and threshold on OOF scores"]),
    ("03  FINAL EVALUATION",["Refit on development → Score test once → Save decisions","Ranking and event detection → Cost surrogate comparison"]),
]
for i,(title,lines) in enumerate(stages):
    y=8+i*170
    d.rounded_rectangle((10,y,1490,y+142),radius=12,fill=LIGHT)
    label(35,y+8,title,30,TEAL)
    for j,line in enumerate(lines):
        label(35,y+51+j*38,line,30)
    if i<2: label(750,y+143,"↓",25,anchor="mt")
im.save(FIG/"pipeline.png")

im,d,label=canvas(465)
def qnum(day):
    return (int(day[:4])-1984)*4+(int(day[5:7])-1)//3
def xpos(q):
    return 310+27*q/4
periods=[]
for f in folds:
    a,b,c,e1=[qnum(f[k]) for k in ("train_start","train_end","validation_start","validation_end")]
    periods.append((f["id"].replace("_"," ").title(),[(a,b+1,TEAL),(b+1,c,"#bac9d4"),(c,e1+1,"#528eb4")]))
periods.append(("Final evaluation",[(qnum(bounds["development_start"]),qnum(bounds["development_end"])+1,TEAL),(qnum(bounds["gap_start"]),qnum(bounds["gap_end"])+1,"#bac9d4"),(qnum(bounds["test_start"]),qnum(bounds["test_end"])+1,NAVY)]))
for i,(key,parts) in enumerate(periods):
    y=35+i*80
    label(10,y,key,30)
    for a,b,c in parts: d.rectangle((xpos(a),y,xpos(b),y+39),fill=c)
for year in [1984,1994,2004,2014,2024]:
    label(xpos((year-1984)*4),355,year,27,anchor="mt")
for i,(key,c) in enumerate([("Training / development",TEAL),("4-quarter gap","#bac9d4"),("Validation","#528eb4"),("Locked test",NAVY)]):
    x=[20,444,790,1100][i]
    d.rectangle((x,407,x+26,435),fill=c)
    label(x+36,402,key,27)
im.save(FIG/"timeline.png")

im=Image.new('RGB',(1650,520),'white');draw=ImageDraw.Draw(im)
def txt(x,y,s,size=29,anchor=None):draw.text((x,y),s,font=ImageFont.truetype('C:/Windows/Fonts/calibri.ttf',size),fill='#183748',anchor=anchor)
for i,(pairs,yname,xname,base,point) in enumerate([(curves['pr'],'Precision','Recall',m['prevalence'],(m['recall'],m['precision'])),(curves['roc'],'True positive rate','False positive rate',None,(m['false_positive_rate'],m['recall']))]):
 left=94+i*820
 def xy(x,y):return(left+x*640,406-y*295)
 for tick in [0,.25,.5,.75,1]:
  x,y=xy(tick,tick);draw.line((left,y,left+640,y),fill='#d3dcdf',width=2);txt(left-12,y,f'{tick:g}',23,'rm');txt(x,415,f'{tick:g}',23,'mt')
 draw.line([xy(x,y) for x,y in pairs],fill='#167e80',width=4)
 for a in range(0,100,3):draw.line([xy(a/100,base if base else a/100),xy((a+1.7)/100,base if base else (a+1.7)/100)],fill='#995e49',width=3)
 x,y=xy(*point);draw.ellipse([x-7,y-7,x+7,y+7],fill='#091821')
 txt(left,15,yname,33);txt(left,63,('Dashed: prevalence (0.0723%)' if i==0 else 'Dashed: chance diagonal'),26);txt(left+320,451,xname,29,'mt')
im.save(FIG/'ranking_curves.png')
