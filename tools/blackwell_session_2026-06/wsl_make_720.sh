#!/usr/bin/env bash
set -e
source ~/artifixer-venv/bin/activate
python - <<'PY'
import os, shutil
from PIL import Image
src = "/mnt/d/REPOS/HYPERclone/apex-sfm/_dataset/Laptop/train/data"
dst = os.path.expanduser("~/laptop_720")
os.makedirs(dst + "/images", exist_ok=True)
if os.path.isdir(dst + "/sparse/0"):
    shutil.rmtree(dst + "/sparse/0")
shutil.copytree(src + "/sparse/0", dst + "/sparse/0")
LONG = 1280
n = 0
for f in sorted(os.listdir(src + "/images")):
    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    im = Image.open(src + "/images/" + f).convert("RGB")
    w, h = im.size
    s = LONG / max(w, h)
    im.resize((round(w * s), round(h * s)), Image.LANCZOS).save(dst + "/images/" + f, quality=95)
    n += 1
print(f"wrote {n} images at long-side {LONG} -> {dst}")
ww, hh = Image.open(dst + "/images/" + sorted(os.listdir(dst + "/images"))[0]).size
print("sample resized size:", ww, "x", hh)
PY
