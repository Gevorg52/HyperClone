"""Render every model version at the SAME held-out views + emit a self-contained
HTML blink-comparator. Arrow keys change VIEW; number keys / tabs flick between
VERSIONS instantly on the same view (best way to spot subtle differences).

Usage:
  python render_viewer.py data=<train/data> out=<dir> [nviews=8] [factor=2] \
      GT base=<ckpt> v2=<ckpt> wd=<ckpt> ...
Open <dir>/viewer.html in a browser.  (`GT` with no =ckpt adds the captured frames.)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats

SH = 3


def load_splats(p, device):
    ck = torch.load(p, weights_only=True, map_location=device)
    return {k: v.to(device) for k, v in ck["splats"].items()}


def main():
    device = "cuda"
    data_dir = "_dataset/Laptop/train/data"
    out = Path("_dataset/Laptop/viewer")
    nviews, factor = 8, 2
    versions = []  # list of (label, ckpt_or_None)
    for a in sys.argv[1:]:
        if a.startswith("data="):
            data_dir = a.split("=", 1)[1]
        elif a.startswith("out="):
            out = Path(a.split("=", 1)[1])
        elif a.startswith("nviews="):
            nviews = int(a.split("=", 1)[1])
        elif a.startswith("factor="):
            factor = int(a.split("=", 1)[1])
        elif a == "GT":
            versions.append(("GT", None))
        elif "=" in a:
            label, ck = a.split("=", 1)
            versions.append((label, ck))
    out.mkdir(parents=True, exist_ok=True)

    data = load_colmap_data(data_dir, factor=factor, test_every=8, normalize=True)
    test = list(data["test_indices"])
    pick = [test[i] for i in np.linspace(0, len(test) - 1, nviews).astype(int)]

    def save(img_bgr, j, label):
        cv2.imwrite(str(out / f"v{j:02d}_{label}.jpg"), img_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

    for label, ck in versions:
        if ck is None:  # GT
            for j, idx in enumerate(pick):
                gt = (load_image(data, idx)["image"].numpy() * 255).astype(np.uint8)
                save(gt[..., ::-1], j, label)
        else:
            sp = load_splats(ck, device)
            with torch.no_grad():
                for j, idx in enumerate(pick):
                    b = load_image(data, idx)
                    K = b["K"].to(device); c2w = b["c2w"].to(device)
                    gt = b["image"]; H, W = gt.shape[:2]
                    r, _, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH,
                                            torch.zeros(3, device=device), "antialiased")
                    img = (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    save(img[..., ::-1], j, label)
            del sp
            torch.cuda.empty_cache()
        print(f"rendered {label} ({nviews} views)", flush=True)

    labels = [v[0] for v in versions]
    (out / "viewer.html").write_text(_HTML.replace("__LABELS__", str(labels))
                                     .replace("__NVIEWS__", str(nviews)), encoding="utf-8")
    print(f"\nDONE -> open {out / 'viewer.html'}")


_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Splat Compare</title>
<style>
 html,body{margin:0;height:100%;background:#111;color:#ddd;font:14px system-ui;overflow:hidden}
 #bar{position:fixed;top:0;left:0;right:0;padding:8px 12px;background:rgba(0,0,0,.7);z-index:10}
 #tabs{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
 .tab{padding:4px 10px;border:1px solid #444;border-radius:6px;cursor:pointer;user-select:none}
 .tab.on{background:#2d7;color:#000;border-color:#2d7;font-weight:600}
 #stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center}
 #img{max-width:100vw;max-height:100vh;object-fit:contain;image-rendering:auto}
 #hint{position:fixed;bottom:8px;left:12px;color:#888}
 b{color:#fff}
</style></head><body>
<div id="bar">
  <span>view <b id="vw">1</b>/__NVIEWS__ &nbsp;|&nbsp; version: <b id="lb"></b>
  &nbsp;|&nbsp; <span style="color:#888">←/→ view · ↑/↓ or 1-9 version · G=GT flick</span></span>
  <div id="tabs"></div>
</div>
<div id="stage"><img id="img"></div>
<div id="hint"></div>
<script>
const LABELS=__LABELS__, NV=__NVIEWS__;
let v=0, k=0, lastNonGT=0;
const img=document.getElementById('img'), tabs=document.getElementById('tabs');
const cache={};
function src(j,l){return `v${String(j).padStart(2,'0')}_${l}.jpg`;}
// preload
for(let j=0;j<NV;j++)for(const l of LABELS){const i=new Image();i.src=src(j,l);cache[src(j,l)]=i;}
LABELS.forEach((l,i)=>{const t=document.createElement('div');t.className='tab';t.textContent=(i+1)+' '+l;
  t.onclick=()=>{k=i;draw();};tabs.appendChild(t);});
function draw(){
  img.src=src(v,LABELS[k]);
  document.getElementById('vw').textContent=v+1;
  document.getElementById('lb').textContent=LABELS[k];
  [...tabs.children].forEach((t,i)=>t.classList.toggle('on',i===k));
  if(LABELS[k]!=='GT') lastNonGT=k;
}
addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'){v=(v+1)%NV;draw();}
  else if(e.key==='ArrowLeft'){v=(v-1+NV)%NV;draw();}
  else if(e.key==='ArrowUp'){k=(k-1+LABELS.length)%LABELS.length;draw();}
  else if(e.key==='ArrowDown'){k=(k+1)%LABELS.length;draw();}
  else if(e.key>='1'&&e.key<='9'){const i=+e.key-1;if(i<LABELS.length){k=i;draw();}}
  else if(e.key.toLowerCase()==='g'){const gi=LABELS.indexOf('GT');if(gi>=0)k=(k===gi?lastNonGT:gi);draw();}
});
draw();
</script></body></html>"""


if __name__ == "__main__":
    main()
