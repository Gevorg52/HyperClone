"""UI-2 validation gate — mini smoke (39-frame Laptop subset, 500 steps, save_all
quarterly -> ckpts at 125/250/375/500). Drives the runner directly and verifies:
  - strips auto-appear at each checkpoint (>=3)
  - 'Preview now' works mid-run
  - the interactive viewer SOG exports from the final model
  - trainer stall from preview contention stays <5s
Guarded under __main__ (hloc's Windows-spawn DataLoader workers re-import this module;
without the guard they'd re-run the whole harness — the UI-1 fork-bomb bug).
"""
import sys, time, json, os, statistics
from pathlib import Path

REPO = Path(r"D:\REPOS\HYPERclone\apex-sfm")
sys.path.insert(0, str(REPO))
DATASET = REPO / "_dataset"
INPUT = DATASET / "UI1_SMOKE" / "input_images"     # existing 39-frame every-8th subset
RUN_DIR = DATASET / "UI2_SMOKE"


def main():
    from gui2 import defaults as D
    from gui2.runner import PipelineRunner

    step_times = []      # (step, wall_time) per training step
    spawn_events = []    # preview subprocess launches

    _orig_cb = PipelineRunner._train_cb
    def _patched_cb(self, key):
        cb = _orig_cb(self, key)
        def wrap(step=0, total=1, loss=0.0, psnr=0.0):
            step_times.append((int(step), time.time()))
            return cb(step, total, loss, psnr)
        return wrap
    PipelineRunner._train_cb = _patched_cb

    _orig_spawn = PipelineRunner._spawn_preview
    def _patched_spawn(self, ckpt, step, height=360, cols=3, blocking=False, timeout=30.0):
        spawn_events.append({"t": time.time(), "step": int(step), "blocking": blocking})
        return _orig_spawn(self, ckpt, step, height, cols, blocking, timeout)
    PipelineRunner._spawn_preview = _patched_spawn

    from gui2.runner import RunConfig
    settings = D.defaults_for("environment", False)
    settings["cap_max"] = 100000
    cfg = RunConfig(
        scan_name="UI2_SMOKE", mode="environment",
        input_dir=str(INPUT), run_dir=str(RUN_DIR), settings=settings,
        source_mp=8.29, subject="", detail=False,
        max_steps_override=500, data_factor_override=4, delivery_no_qa=True)

    runner = PipelineRunner(cfg)

    seen = {}            # filename -> {t, size, step}
    def scan():
        pdir = RUN_DIR / "preview"
        if not pdir.is_dir():
            return
        for p in sorted(pdir.glob("strip_*.png")):
            if p.name not in seen:
                try:
                    st = int(p.stem.split("_")[1])
                except (IndexError, ValueError):
                    st = -1
                seen[p.name] = {"t": time.time(), "size": p.stat().st_size, "step": st}

    preview_now = None
    did_pn = False

    t0 = time.time()
    runner.start()
    while runner.is_alive():
        snap = runner.snapshot()
        scan()
        tr = snap["stages"]["training"]
        if (not did_pn and snap["current_stage"] == "training"
                and tr["status"] == "running" and tr["frac"] > 0.45 and len(seen) >= 1):
            latest = runner._latest_ckpt(runner.ckpt_dir)
            mt_before = None
            if latest:
                st = int(Path(latest).name.split("_")[1])
                cand = RUN_DIR / "preview" / f"strip_{st}.png"
                mt_before = cand.stat().st_mtime if cand.is_file() else None
            path_pn, msg_pn = runner.preview_now(wait=True, timeout=60)
            mt_after = os.path.getmtime(path_pn) if path_pn and os.path.isfile(path_pn) else None
            preview_now = {
                "path": path_pn, "msg": msg_pn,
                "exists": bool(path_pn and os.path.isfile(path_pn)),
                "ckpt": latest, "at_frac": round(tr["frac"], 3),
                "mtime_before": mt_before, "mtime_after": mt_after,
                "mtime_advanced": bool(mt_before is None or (mt_after and mt_after > mt_before)),
            }
            did_pn = True
        time.sleep(1.0)

    scan()
    snap = runner.snapshot()

    # export the interactive viewer SOG from the final model
    sog_path, sog_msg = runner.export_viewer()
    export_res = {
        "sog": sog_path, "msg": sog_msg,
        "exists": bool(sog_path and os.path.isfile(sog_path)),
        "size_mb": round(os.path.getsize(sog_path) / 1e6, 2)
        if sog_path and os.path.isfile(sog_path) else None,
        "scene_url": ("/scenes/" + Path(sog_path).resolve().relative_to(DATASET.resolve()).as_posix())
        if sog_path and os.path.isfile(sog_path) else None,
    }

    # stall analysis
    dts = [(step_times[i][1] - step_times[i - 1][1]) for i in range(1, len(step_times))]
    gaps = [{"to_step": step_times[i][0], "dt": round(step_times[i][1] - step_times[i - 1][1], 3)}
            for i in range(1, len(step_times))]
    gaps_sorted = sorted(gaps, key=lambda g: g["dt"], reverse=True)
    # max gap that OVERLAPS a preview render window (attribution): a preview launched at
    # T occupies roughly [T, T+12s]; find the biggest step gap starting in that window.
    def gap_in_windows():
        worst = 0.0
        for i in range(1, len(step_times)):
            gs, ge = step_times[i - 1][1], step_times[i][1]
            for sp in spawn_events:
                if sp["t"] - 1.0 <= gs <= sp["t"] + 14.0:
                    worst = max(worst, ge - gs)
                    break
        return round(worst, 3)
    stall = {
        "n_steps_logged": len(step_times),
        "median_step_s": round(statistics.median(dts), 4) if dts else None,
        "p95_step_s": round(sorted(dts)[int(len(dts) * 0.95)], 4) if len(dts) > 1 else None,
        "max_step_gap_s": round(max(dts), 3) if dts else None,
        "max_gap_at_step": gaps_sorted[0]["to_step"] if gaps_sorted else None,
        "max_gap_in_preview_window_s": gap_in_windows(),
        "top5_gaps": gaps_sorted[:5],
    }

    strip_steps = sorted({v["step"] for v in seen.values()})
    gate = {
        "run_done": snap["status"] == "done",
        "strips_ge_3": len(seen) >= 3,
        "preview_now_ok": bool(preview_now and preview_now["exists"]),
        "viewer_export_ok": export_res["exists"],
        "trainer_stall_under_5s": bool(stall["max_step_gap_s"] is not None
                                       and stall["max_step_gap_s"] < 5.0),
    }
    gate["PASS"] = all(gate.values())

    result = {
        "config": {"mode": "environment", "input": str(INPUT), "run_dir": str(RUN_DIR),
                   "max_steps": 500, "data_factor": 4, "cap_max": 100000,
                   "delivery_no_qa": True, "save_all": True, "seed": settings.get("seed")},
        "status": snap["status"], "elapsed_s": round(time.time() - t0, 1),
        "n_strips": len(seen), "strip_steps": strip_steps,
        "strips": seen, "spawn_events": spawn_events,
        "preview_now": preview_now, "export_viewer": export_res,
        "trainer_stall": stall, "deliverables": snap["deliverables"], "gate": gate,
    }
    out = REPO / "_infusion" / "ui2_gate_trace.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("=== UI-2 GATE ===")
    print(json.dumps(gate, indent=2))
    print("n_strips:", len(seen), "steps:", strip_steps)
    print("max_step_gap_s:", stall["max_step_gap_s"], "at step", stall["max_gap_at_step"])
    print("max_gap_in_preview_window_s:", stall["max_gap_in_preview_window_s"])
    print("median_step_s:", stall["median_step_s"])
    print("export sog:", export_res["sog"], export_res["size_mb"], "MB")
    print("status:", snap["status"], "elapsed", result["elapsed_s"], "s")
    print("trace ->", out)


if __name__ == "__main__":
    main()
