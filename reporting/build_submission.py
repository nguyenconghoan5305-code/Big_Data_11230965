"""Build the editable submission from the editorial manuscript and measured figure assets."""
from pathlib import Path
import os
from docx import Document
from docx.shared import Cm

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get("BFT_REPORT_OUTPUT", ROOT/"submission"))
OUT.mkdir(parents=True,exist_ok=True)
doc=Document(ROOT/"reporting/submission_manuscript.docx")
figures=0
for paragraph in doc.paragraphs:
    if not paragraph.text.startswith("[[figure "):
        continue
    name,width=paragraph.text.removeprefix("[[figure ").removesuffix("]]").split()
    paragraph.clear()
    paragraph.add_run().add_picture(str(OUT/"figures"/(name+".png")),width=Cm(float(width)))
    paragraph.paragraph_format.keep_with_next=True
    figures+=1
assert figures==10 and len(doc.tables)==10
doc.save(OUT/"BFT65B_Final_Report.docx")
print("Saved final DOCX with",figures,"figure panels and",len(doc.tables),"tables")

