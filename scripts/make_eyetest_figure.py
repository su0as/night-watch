"""docs/assets/eyetest_side_vs_head.png: what Gemma is shown, and what it said (from runs/eyetest)."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/eyetest")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/assets/eyetest_side_vs_head.png")
d = root / "1024x768" / "frames"
rows = [json.loads(x) for x in (root / "eyetest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def said(mode: str, sheet: str) -> str:
    hits = [r for r in rows if r.get("mode") == mode and r.get("sheet") == sheet and r.get("config") == "1024x768"]
    return hits[-1]["posture"].replace("_", " ") if hits else "?"


W, H = 512, 384
cells = [
    (d / "upright_0_3.png", "side camera, truly upright", "Gemma: " + said("side", "frames/upright_0_sheet.png")),
    (d / "upside_down_0_3.png", "side camera, truly UPSIDE DOWN", "Gemma: " + said("side", "frames/upside_down_0_sheet.png")),
    (d / "upright_0_head3.png", "head camera, truly upright", "Gemma: " + said("head", "frames/upright_0_sheet.png")),
    (d / "upside_down_0_head3.png", "head camera, truly UPSIDE DOWN", "Gemma: " + said("head", "frames/upside_down_0_sheet.png")),
]
sheet = Image.new("RGB", (2 * W, 2 * H), "white")
for i, (p, top, bottom) in enumerate(cells):
    im = Image.open(p).convert("RGB").resize((W, H))
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, W, 24], fill=(0, 0, 0))
    dr.text((8, 6), top, fill=(255, 255, 255))
    ok = ("UPSIDE" in top) == ("upside" in bottom)
    dr.rectangle([0, H - 26, W, H], fill=(20, 110, 40) if ok else (170, 30, 30))
    dr.text((8, H - 19), bottom + ("   (right)" if ok else "   (wrong)"), fill=(255, 255, 255))
    sheet.paste(im, ((i % 2) * W, (i // 2) * H))
out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out)
print(out)
