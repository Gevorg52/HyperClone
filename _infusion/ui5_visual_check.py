"""UI-5 visual validation gate (playwright).

Two-phase audit of the gui2 wizard in BOTH browser color schemes (emulated
prefers-color-scheme light AND dark) at 1920x1080 and 2560x1440.

PHASE A - LIVE server (real gui2.app: build_theme() + theme.css + /viewer +
/scenes, same mount as gui2.app.launch): the surfaces present in the initial DOM
- Step 1 (New scan), the Utilities panel (expanded), and the Advanced drawer
(expanded). Real Gradio components (inputs, dropdowns, checkboxes, radios, tabs,
accordions, buttons, markdown).

PHASE B - STANDALONE faithful render: Gradio lazily renders `visible=False`
columns, so Steps 2/3/4 are NOT in the initial DOM and cannot be driven without a
full pipeline run. Their color risk lives entirely in our custom-HTML surfaces
(progress bar, stage list, quality card, deliverable cards, honest-flag box). We
render those with the IDENTICAL tokens: build_theme()._get_theme_css() (byte for
byte what Gradio injects) + theme.css + the real renderer output, wrapped in the
same `.gradio-container`. The Gradio component chrome inside Steps 2/3/4 is the
same components validated live in Phase A.

Every screenshot is audited on the live DOM:
  1. CONTRAST  - each visible text node's computed color vs effective background
     clears WCAG-AA (>=4.5:1 normal, >=3:1 large text).
  2. EM-DASH   - document.body.innerText contains zero U+2014.
  3. FULL-WIDTH- .gradio-container fills >=95% of the viewport width.

Re-runnable:  python _infusion/ui5_visual_check.py
Persists: docs/gui2_screens/*.png, _infusion/ui5_visual_check.json
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(r"D:\REPOS\HYPERclone\apex-sfm")
sys.path.insert(0, str(REPO))
DATASET = REPO / "_dataset"
SCREENS_DIR = REPO / "docs" / "gui2_screens"
OUT_JSON = REPO / "_infusion" / "ui5_visual_check.json"
PORT = 7863
OBJ_RUN = DATASET / "UI4_SMOKE"          # object-mode run with a real delivery record

VIEWPORTS = [(1920, 1080), (2560, 1440)]
SCHEMES = ["light", "dark"]
LIVE_SCREENS = ["step1", "utilities", "advanced"]
STD_SCREENS = ["step2", "step3", "step4"]


# --------------------------------------------------------------------------
# Real rendered HTML (from the live renderers) for every custom surface.
# --------------------------------------------------------------------------
def build_injections() -> dict:
    from gui2 import deliver as DV
    from gui2.app import (_progress_html, _stages_html, _deliverables_html,
                          _stepper_html, _viewer_iframe_html)
    from gui2.runner import STAGE_WEIGHTS, STAGE_LABELS

    card = DV.quality_card(str(OBJ_RUN), mode="object")
    quality = DV.quality_card_html(card)

    items = []
    for p in sorted((OBJ_RUN / "deliverables").glob("*")):
        if p.suffix.lower() in (".sog", ".ply"):
            items.append({"name": p.name, "path": str(p),
                          "size_mb": round(p.stat().st_size / 1e6, 2)})
    deliv = _deliverables_html(items, str(OBJ_RUN))

    progress = _progress_html(62.0, "Training (12,400 / 30,000)", 4215)
    order = [k for k, _ in STAGE_WEIGHTS]
    snap = {"stage_order": order, "stage_labels": STAGE_LABELS, "stages": {}}
    for i, k in enumerate(order):
        st = ("done" if i < 3 else ("running" if i == 3 else "pending"))
        snap["stages"][k] = {"status": st, "frac": (0.5 if st == "running" else 0.0),
                             "desc": ("step 12,400" if st == "running" else "")}
    stages = _stages_html(snap)

    return {
        "quality": quality, "deliv": deliv, "progress": progress, "stages": stages,
        "stepper2": _stepper_html(2, "object", done={1}),
        "stepper3": _stepper_html(3, "object", done={1, 2}),
        "stepper4": _stepper_html(4, "object", done={1, 2, 3}),
        "sharpness_label": card.get("sharpness", {}).get("label"),
        "n_flags": len(card.get("flags", [])),
    }


# --------------------------------------------------------------------------
# Standalone doc for the custom (lazily-rendered) surfaces, with the EXACT
# theme tokens + theme.css inlined.
# --------------------------------------------------------------------------
def theme_and_css() -> tuple[str, str]:
    from gui2.app import _HERE
    from gui2.theme import build_theme
    theme_css = build_theme()._get_theme_css()
    app_css = (_HERE / "theme.css").read_text(encoding="utf-8")
    return theme_css, app_css


_HEADER = ('<div class="apex-header"><h1>Apex Reality</h1>'
           '<p>Clean, usable Gaussian splats from ordinary captures.</p></div>')


def standalone_html(screen: str, inj: dict, scheme: str,
                    theme_css: str, app_css: str) -> str:
    if screen == "step2":
        body = (inj["stepper2"] + '<h3 style="color:var(--apex-text)">2 &middot; '
                'Processing</h3>' + inj["progress"] + inj["stages"])
    elif screen == "step3":
        body = (inj["stepper3"] + '<h3 style="color:var(--apex-text)">3 &middot; '
                'Confirm the object cut-out</h3>'
                '<div class="apex-panel">If the red matte covers just your subject, '
                'choose <b>Looks right</b>. If it grabbed the stand, choose '
                '<b>Fix it</b>.</div>')
    else:  # step4
        body = (inj["stepper4"] + '<h3 style="color:var(--apex-text)">4 &middot; '
                'Your splat is ready</h3>'
                '<h4 style="color:var(--apex-text)">Quality report</h4>'
                + inj["quality"]
                + '<h4 style="color:var(--apex-text)">Deliverables</h4>'
                + inj["deliv"]
                + '<div style="font-size:12px;color:var(--apex-muted);margin-top:10px;">'
                  'Drag to orbit &middot; scroll to zoom &middot; right-drag to pan.</div>')
    root_cls = ' class="dark"' if scheme == "dark" else ""
    return (
        f"<!doctype html><html{root_cls}><head><meta charset='utf-8'><style>\n"
        f"{theme_css}\n{app_css}\n"
        "html,body{margin:0;padding:0;background:var(--body-background-fill,#0e0f13);}\n"
        "</style></head><body>\n"
        f'<div class="gradio-container">{_HEADER}{body}</div>\n'
        "</body></html>"
    )


# --------------------------------------------------------------------------
# In-process server (mirrors gui2.app.launch exactly)
# --------------------------------------------------------------------------
def serve():
    import uvicorn
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from gui2.app import create_app, _HERE
    from gui2.theme import build_theme

    DATASET.mkdir(parents=True, exist_ok=True)
    fastapi_app = FastAPI()
    fastapi_app.mount("/viewer",
                      StaticFiles(directory=str(_HERE / "viewer")), name="viewer")
    fastapi_app.mount("/scenes", StaticFiles(directory=str(DATASET)), name="scenes")
    demo = create_app()
    demo.queue(default_concurrency_limit=4)
    css = (_HERE / "theme.css").read_text(encoding="utf-8")
    app = gr.mount_gradio_app(fastapi_app, demo, path="/",
                              theme=build_theme(), css=css)
    cfg = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(120):
        if server.started:
            break
        time.sleep(0.25)
    return server


# --------------------------------------------------------------------------
# Browser-side audit (contrast + em-dash + width).
# --------------------------------------------------------------------------
AUDIT_JS = r"""
(() => {
  const parse = (s) => {
    if (!s) return null;
    const m = s.match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const p = m[1].split(',').map(x => parseFloat(x.trim()));
    return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1];
  };
  const blend = (fg, bg) => {
    const a = fg[3];
    return [fg[0]*a + bg[0]*(1-a), fg[1]*a + bg[1]*(1-a), fg[2]*a + bg[2]*(1-a)];
  };
  const lum = (c) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v/12.92 : Math.pow((v+0.055)/1.055, 2.4); };
    return 0.2126*f(c[0]) + 0.7152*f(c[1]) + 0.0722*f(c[2]);
  };
  const ratio = (a, b) => { const L1 = lum(a), L2 = lum(b); const hi = Math.max(L1,L2), lo = Math.min(L1,L2); return (hi+0.05)/(lo+0.05); };
  const APP_BG = [14,15,19];
  const effBg = (el) => {
    let node = el, guard = 0;
    while (node && guard++ < 60) {
      const st = getComputedStyle(node);
      const c = parse(st.backgroundColor);
      if (c && c[3] > 0) {
        if (c[3] >= 0.999) return [c[0],c[1],c[2]];
        const pbg = node.parentElement ? effBg(node.parentElement) : APP_BG;
        return blend(c, pbg);
      }
      node = node.parentElement;
    }
    return APP_BG;
  };
  const visible = (el) => {
    const st = getComputedStyle(el);
    if (st.display === 'none' || st.visibility === 'hidden') return false;
    if (parseFloat(st.opacity) === 0) return false;
    const r = el.getBoundingClientRect();
    return r.width > 1 && r.height > 1;
  };
  const SKIP = new Set(['SCRIPT','STYLE','NOSCRIPT','SVG','PATH','IFRAME','CANVAS','TEXTAREA','INPUT']);
  const fails = [];
  let scanned = 0;
  for (const el of document.querySelectorAll('*')) {
    if (SKIP.has(el.tagName)) continue;
    let txt = '';
    for (const n of el.childNodes) if (n.nodeType === 3) txt += n.textContent;
    txt = txt.replace(/\s+/g, ' ').trim();
    if (!txt) continue;
    if (!visible(el)) continue;
    const st = getComputedStyle(el);
    const bg = effBg(el);
    let fg = parse(st.color);
    if (!fg) continue;
    if (fg[3] < 1) fg = blend(fg, bg); else fg = [fg[0],fg[1],fg[2]];
    const cr = ratio(fg, bg);
    const px = parseFloat(st.fontSize);
    const bold = (parseInt(st.fontWeight) || 400) >= 700;
    const large = px >= 24 || (px >= 18.66 && bold);
    const need = large ? 3.0 : 4.5;
    scanned++;
    if (cr + 1e-6 < need) {
      fails.push({tag: el.tagName, cls: (el.className || '').toString().slice(0, 60),
        text: txt.slice(0, 50), fg: fg.map(x => Math.round(x)), bg: bg.map(x => Math.round(x)),
        ratio: Math.round(cr * 100) / 100, need, px: Math.round(px), bold});
    }
  }
  const body = document.body.innerText || '';
  const emIdx = body.indexOf('—');
  const attrEm = [];
  for (const el of document.querySelectorAll('[placeholder],[aria-label],[title]')) {
    for (const a of ['placeholder','aria-label','title']) {
      const v = el.getAttribute(a);
      if (v && v.includes('—')) attrEm.push({attr: a, val: v.slice(0,60)});
    }
  }
  const gc = document.querySelector('.gradio-container');
  const cw = gc ? gc.getBoundingClientRect().width : 0;
  return {scanned, fails, emdash_in_body: emIdx >= 0,
    emdash_context: emIdx >= 0 ? body.slice(Math.max(0, emIdx-30), emIdx+30) : '',
    attr_emdash: attrEm, container_width: cw, viewport_width: window.innerWidth,
    width_ratio: gc ? cw / window.innerWidth : 0};
})()
"""

EXPAND_JS = r"""
(needle) => {
  for (const h of document.querySelectorAll('.label-wrap')) {
    if ((h.textContent || '').includes(needle)) {
      const isOpen = h.classList.contains('open');
      if (!isOpen) h.click();
      return true;
    }
  }
  return false;
}
"""


def audit_and_shot(page, key, phase):
    shot = SCREENS_DIR / f"{key}.png"
    page.screenshot(path=str(shot), full_page=True)
    a = page.evaluate(AUDIT_JS)
    nf = len(a["fails"])
    wr = a["width_ratio"]
    wfail = wr < 0.95
    rec = {"phase": phase, "screenshot": str(shot.relative_to(REPO)),
           "scanned": a["scanned"], "n_contrast_fail": nf,
           "contrast_fails": a["fails"][:25], "emdash_in_body": a["emdash_in_body"],
           "emdash_context": a["emdash_context"], "attr_emdash": a["attr_emdash"],
           "container_width": round(a["container_width"], 1),
           "viewport_width": a["viewport_width"], "width_ratio": round(wr, 4),
           "width_ok": not wfail}
    ok = (nf == 0 and not a["emdash_in_body"] and not wfail)
    print(f"[{'OK' if ok else 'FAIL'}] {key}: contrast_fail={nf} "
          f"emdash={a['emdash_in_body']} width={wr:.3f} scanned={a['scanned']}")
    return rec, nf, a["emdash_in_body"], wfail


def main():
    from playwright.sync_api import sync_playwright

    SCREENS_DIR.mkdir(parents=True, exist_ok=True)
    inj = build_injections()
    theme_css, app_css = theme_and_css()
    print(f"[inj] quality label={inj['sharpness_label']} flags={inj['n_flags']}")

    server = serve()
    print(f"[serve] gui2 wizard on http://127.0.0.1:{PORT}")
    time.sleep(1.0)

    results = {"port": PORT, "run_dir": str(OBJ_RUN), "screens": {}, "summary": {},
               "injection": {k: inj[k] for k in ("sharpness_label", "n_flags")}}
    total_fail = total_emdash = total_width_fail = n_screens = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--force-color-profile=srgb"])
        for (w, h) in VIEWPORTS:
            for scheme in SCHEMES:
                ctx = browser.new_context(viewport={"width": w, "height": h},
                                          color_scheme=scheme, device_scale_factor=1)
                page = ctx.new_page()

                # ---- Phase A: live server ----
                page.goto(f"http://127.0.0.1:{PORT}/",
                          wait_until="domcontentloaded", timeout=60000)
                page.wait_for_selector("#apex-step1", timeout=30000)
                page.wait_for_timeout(1200)
                for screen in LIVE_SCREENS:
                    if screen == "utilities":
                        page.evaluate(EXPAND_JS, "Utilities")
                    elif screen == "advanced":
                        page.evaluate(EXPAND_JS, "Advanced settings")
                    page.wait_for_timeout(500)
                    key = f"{screen}_{scheme}_{w}x{h}"
                    rec, nf, em, wf = audit_and_shot(page, key, "live")
                    results["screens"][key] = rec
                    n_screens += 1
                    total_fail += nf
                    total_emdash += int(em)
                    total_width_fail += int(wf)

                # ---- Phase B: standalone faithful render ----
                for screen in STD_SCREENS:
                    doc = standalone_html(screen, inj, scheme, theme_css, app_css)
                    page.set_content(doc, wait_until="load")
                    page.wait_for_timeout(300)
                    key = f"{screen}_{scheme}_{w}x{h}"
                    rec, nf, em, wf = audit_and_shot(page, key, "standalone")
                    results["screens"][key] = rec
                    n_screens += 1
                    total_fail += nf
                    total_emdash += int(em)
                    total_width_fail += int(wf)

                ctx.close()
        browser.close()

    passed = (total_fail == 0 and total_emdash == 0 and total_width_fail == 0)
    results["summary"] = {"n_screens": n_screens, "total_contrast_fails": total_fail,
                          "screens_with_emdash": total_emdash,
                          "screens_failing_width": total_width_fail, "PASS": passed}
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== UI-5 VISUAL CHECK ===")
    print(f"screens: {n_screens}  contrast_fails: {total_fail}  "
          f"emdash_screens: {total_emdash}  width_fails: {total_width_fail}")
    print(f"PASS={passed}")
    print("json ->", OUT_JSON)
    print("screens ->", SCREENS_DIR)
    try:
        server.should_exit = True
    except Exception:
        pass
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
