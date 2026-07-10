"""UI-4 validation gate — drives the FULL wizard programmatically on the mini smoke
config in OBJECT mode (UI1_SMOKE 39-frame Laptop subset, factor 4, 500 steps, object
extraction ON) so it exercises Step③ presence + Step④ with extraction, then exercises
every UI-4 Deliver-screen + Utilities surface the app wires:

  1. quality card renders REAL numbers from the delivery record (gui2.deliver.quality_card)
  2. View-in-3D serves the object SOG  (HTTP 200 on the /scenes URL + iframe URL well-formed)
  3. Improve-further re-capture report produces the coverage map (gui2.deliver.coverage_report_for)
  4. one Utilities item end-to-end: Capture QC standalone (gui2.qc.run_qc)
  5. deliverable cards carry Gaussian count + SH + scene URL

Guarded under __main__ (hloc's Windows-spawn DataLoader workers re-import this module).
Re-runnable:  python _infusion/ui4_gate_driver.py
"""
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(r"D:\REPOS\HYPERclone\apex-sfm")
sys.path.insert(0, str(REPO))
DATASET = REPO / "_dataset"
INPUT = DATASET / "UI1_SMOKE" / "input_images"
RUN_DIR = DATASET / "UI4_SMOKE"
INFUSION = REPO / "_infusion"
PORT = 7862


def _http_status(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read()
            return int(getattr(r, "status", r.getcode())), len(body)
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _serve_static(port):
    """Mount /viewer + /scenes exactly like gui2.app.launch() and serve on a thread."""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    app = FastAPI()
    app.mount("/viewer", StaticFiles(directory=str(REPO / "gui2" / "viewer")),
              name="viewer")
    app.mount("/scenes", StaticFiles(directory=str(DATASET)), name="scenes")
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    return server


def main():
    from gui2 import defaults as D
    from gui2 import deliver as DV
    from gui2 import qc as QC
    from gui2.runner import PipelineRunner, RunConfig
    from gui2.app import (next_step_after_processing, _export_ready_for_review,
                          _viewer_iframe_html)

    trace = {"config": {}, "checks": {}, "descriptions": {}}

    def check(name, cond, info=""):
        trace["checks"][name] = {"pass": bool(cond), "info": str(info)[:600]}
        print(f"[{'PASS' if cond else 'FAIL'}] {name}  {str(info)[:160]}")
        return bool(cond)

    # Fresh run dir so extraction + QA reflect THIS run.
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR, ignore_errors=True)

    settings = D.defaults_for("object", False)
    settings["cap_max"] = 100_000
    cfg = RunConfig(
        scan_name="UI4_SMOKE", mode="object", input_dir=str(INPUT),
        run_dir=str(RUN_DIR), settings=settings, source_mp=8.29, subject="ui4_obj",
        detail=False, max_steps_override=500, data_factor_override=4,
        delivery_refit=False)          # skip 3000-step refit -> GPU stays bursty
    trace["config"] = {
        "mode": "object", "input": str(INPUT), "run_dir": str(RUN_DIR),
        "max_steps": 500, "data_factor": 4, "cap_max": 100_000,
        "extract_object": True, "delivery_refit": False, "delivery_no_qa": False,
        "seed": settings.get("seed"),
        "argv_delivery_flags": D.assemble_delivery_flags(settings)}

    runner = PipelineRunner(cfg)
    step3_ready = {"seen": False, "at_frac": None}

    t0 = time.time()
    runner.start()
    while runner.is_alive():
        if not step3_ready["seen"] and _export_ready_for_review(str(RUN_DIR)):
            step3_ready["seen"] = True
            step3_ready["at_frac"] = runner.snapshot()["global_frac"]
        time.sleep(1.0)
    snap = runner.snapshot()
    elapsed = round(time.time() - t0, 1)
    print(f"\n=== run finished: status={snap['status']} elapsed={elapsed}s ===")

    run_done = check("run_done", snap["status"] == "done",
                     f"status={snap['status']}")

    # --- Step③ presence: object flow routes through Review, and Review became
    #     reachable during the run (export/images complete). ---
    check("step3_in_object_flow", next_step_after_processing("object") == 3,
          "next_step_after_processing('object')==3")
    check("step3_became_reachable", step3_ready["seen"],
          f"export-ready during run at frac={step3_ready['at_frac']}")

    # --- Step④(1): quality card renders REAL numbers from the delivery record ---
    card = DV.quality_card(str(RUN_DIR), mode="object")
    qhtml = DV.quality_card_html(card)
    (INFUSION / "ui4_quality_card.html").write_text(qhtml, encoding="utf-8")
    sh = card.get("sharpness") or {}
    ge = card.get("geometry") or {}
    vw = card.get("views") or {}
    fl = card.get("floaters") or {}
    lp = sh.get("lpips")
    real_sharp = (card.get("has_record") and lp is not None
                  and 0.0 < float(lp) < 2.0 and sh.get("stars") is not None)
    check("quality_card_has_record", bool(card.get("has_record")))
    check("quality_card_sharpness_real", real_sharp,
          f"obj-LPIPS={lp} stars={sh.get('stars')} label={sh.get('label')}")
    check("quality_card_geometry_real", ge.get("value_pct") is not None,
          f"{ge.get('kind')}={ge.get('value_pct')} ({ge.get('text')})")
    # STRONG assertion: total must be the 39 TRUE inputs (not the 38 registered) — the
    # external folder-drop flow has no input_images under the run dir, so this catches the
    # old total=registered fallback bug (confidently-wrong 38/38 = 100%).
    reg_v, tot_v = vw.get("registered"), vw.get("total")
    check("quality_card_views_real",
          isinstance(reg_v, int) and reg_v > 0 and tot_v == 39 and reg_v <= tot_v,
          f"registered={reg_v} total={tot_v} (expect total==39 true inputs, reg<=total)")
    check("quality_card_floaters_real", fl.get("text") not in (None, "not scored"),
          f"{fl.get('label')} :: {fl.get('text')}")
    trace["descriptions"]["step4_quality_card"] = card

    # --- Step④(2): deliverable cards carry count + SH + scene URL ---
    items = snap["deliverables"]
    cards = DV.deliverable_cards(items, str(RUN_DIR))
    (INFUSION / "ui4_deliverable_cards.html").write_text(
        DV.deliverable_cards_html(cards), encoding="utf-8")
    obj_card = next((c for c in cards if (c.get("mode") == "object" and c.get("sog"))),
                    None)
    check("deliverable_object_card", obj_card is not None,
          f"cards={[c['base'] for c in cards]}")
    if obj_card:
        check("deliverable_card_meta",
              obj_card.get("n_gaussians", 0) and obj_card.get("sh_degree") is not None,
              f"n={obj_card.get('n_gaussians')} SH={obj_card.get('sh_degree')} "
              f"sog_mb={obj_card.get('sog_mb')}")
    trace["descriptions"]["step4_deliverable_cards"] = cards

    # --- Step④(3): View-in-3D serves the object SOG (HTTP 200 + iframe URL) ---
    sog = obj_card.get("sog") if obj_card else None
    scene_url = DV.scene_url_for(sog) if sog else None
    iframe = _viewer_iframe_html(sog) if sog else ""
    server = _serve_static(PORT)
    time.sleep(0.5)
    base_st, base_info = _http_status(f"http://127.0.0.1:{PORT}/viewer/index.html")
    scene_st, scene_info = _http_status(f"http://127.0.0.1:{PORT}{scene_url}"
                                        if scene_url else f"http://127.0.0.1:{PORT}/scenes/")
    check("viewer_bundle_http_200", base_st == 200, f"/viewer/index.html -> {base_st} ({base_info}B)")
    check("object_sog_http_200", scene_st == 200,
          f"{scene_url} -> {scene_st} ({scene_info}B)")
    check("iframe_url_wellformed",
          bool(scene_url) and scene_url in iframe and "/viewer/index.html?content=" in iframe,
          f"scene_url={scene_url}")
    trace["descriptions"]["step4_view_in_3d"] = {
        "sog": sog, "scene_url": scene_url,
        "iframe_src_prefix": iframe[:180]}
    try:
        server.should_exit = True
    except Exception:
        pass

    # --- Improve-further: re-capture / coverage report produces the map ---
    cov_png = str(RUN_DIR / "improve" / "coverage_map.png")
    cov_ok, cov_text, cov_pct = False, "", None
    try:
        cov_text, cov_png_out, cov_pct = DV.coverage_report_for(
            runner.data_dir, int(runner.data_factor), cov_png)
        cov_ok = os.path.isfile(cov_png_out)
        if cov_ok:
            shutil.copy2(cov_png_out, str(INFUSION / "ui4_coverage_map.png"))
    except Exception as e:  # noqa: BLE001
        cov_text = f"{type(e).__name__}: {e}"
    check("recapture_coverage_map", cov_ok,
          f"coverage={None if cov_pct is None else round(cov_pct*100,1)}% -> {cov_png}")
    trace["descriptions"]["improve_recapture"] = {
        "coverage_pct": (round(cov_pct * 100, 1) if cov_pct is not None else None),
        "report": cov_text, "map_png": cov_png}

    # --- Utilities: Capture QC standalone (end-to-end) ---
    rep = QC.run_qc(str(INPUT), "object")
    qc_ok = rep.ok and rep.n_frames > 0 and rep.hist_png and os.path.isfile(rep.hist_png)
    if qc_ok:
        shutil.copy2(rep.hist_png, str(INFUSION / "ui4_util_qc_hist.png"))
    check("util_capture_qc", bool(qc_ok),
          f"ok={rep.ok} n={rep.n_frames} {rep.width}x{rep.height} "
          f"factor={rep.data_factor} hist={bool(rep.hist_png)}")
    trace["descriptions"]["util_capture_qc"] = {
        "n_frames": rep.n_frames, "res": f"{rep.width}x{rep.height}",
        "megapixels": rep.megapixels, "data_factor": rep.data_factor,
        "training_mp": rep.training_mp}

    # --- delivery record + object-extraction evidence ---
    rec = DV.primary_record(str(RUN_DIR), "object")
    if rec:
        vote = (rec.get("stages") or {}).get("vote") or {}
        deliv = rec.get("deliverable") or {}
        trace["descriptions"]["delivery_record"] = {
            "master_n": (rec.get("stages", {}).get("master", {}) or {}).get("n"),
            "vote_kept": vote.get("n_kept"), "vote_kept_pct": vote.get("kept_pct"),
            "delivered_n": deliv.get("n_gaussians"), "sh": deliv.get("sh_degree"),
            "sog_mb": deliv.get("sog_mb"),
            "record": f"{RUN_DIR}/deliverables/UI4_SMOKE_object_delivery.json"}
        check("object_extraction_ran",
              bool(vote.get("n_kept")) and vote.get("n_kept") < (rec.get("stages", {})
                   .get("master", {}) or {}).get("n", 10**12),
              f"vote kept {vote.get('n_kept')} of master "
              f"{(rec.get('stages', {}).get('master', {}) or {}).get('n')}")

    trace["status"] = snap["status"]
    trace["elapsed_s"] = elapsed
    passed = all(c["pass"] for c in trace["checks"].values())
    trace["PASS"] = passed
    trace["n_checks"] = len(trace["checks"])
    trace["n_pass"] = sum(c["pass"] for c in trace["checks"].values())

    out = INFUSION / "ui4_gate_trace.json"
    out.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    print("\n=== UI-4 GATE ===")
    print(json.dumps({k: v["pass"] for k, v in trace["checks"].items()}, indent=2))
    print(f"{trace['n_pass']}/{trace['n_checks']} checks passed; PASS={passed}")
    print("trace ->", out)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
