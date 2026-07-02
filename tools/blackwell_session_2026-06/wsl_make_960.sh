#!/usr/bin/env bash
set -e
source ~/artifixer-venv/bin/activate
python - <<'PY'
import os, shutil
from PIL import Image
src = "/mnt/d/REPOS/HYPERclone/apex-sfm/_dataset/Laptop/train/data"
dst = os.path.expanduser("~/laptop_960")
os.makedirs(dst + "/images", exist_ok=True)
if os.path.isdir(dst + "/sparse/0"): shutil.rmtree(dst + "/sparse/0")
shutil.copytree(src + "/sparse/0", dst + "/sparse/0")
TW, TH = 1712, 963   # = 16*107 x 9*107 : EXACT 16:9 of 3840x2160 (preserves camera aspect; ~960p in-distribution)
n = 0
for f in sorted(os.listdir(src + "/images")):
    if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
    im = Image.open(src + "/images/" + f).convert("RGB")
    im.resize((TW, TH), Image.LANCZOS).save(dst + "/images/" + f, quality=95); n += 1
ww, hh = Image.open(dst + "/images/" + sorted(os.listdir(dst + "/images"))[0]).size
print(f"wrote {n} images at {ww}x{hh} -> {dst}")
PY
