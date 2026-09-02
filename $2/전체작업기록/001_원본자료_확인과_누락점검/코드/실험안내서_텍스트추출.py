from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

pptx = Path(__file__).resolve().parent.parent / "HAI_Experiment_UWB_GuideLine.pptx"
ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}

with ZipFile(pptx) as zf:
    slides = sorted(
        (name for name in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
        key=lambda name: int(re.search(r"\d+", Path(name).stem).group()),
    )
    for number, name in enumerate(slides, 1):
        root = ET.fromstring(zf.read(name))
        texts = [node.text or "" for node in root.findall(".//a:t", ns)]
        print(f"\n===== SLIDE {number} =====")
        print("\n".join(texts))
