"""gui2 — pipeline runner (design §3 ②, §5 UI-1 scope).

Orchestrates the FULL chain headless in a background thread, reusing the proven
WP-T1 stage handlers (gui/stages/*) and core modules:

    QC -> features -> matching -> reconstruction -> export
       -> priors+dense-init -> training -> delivery

with a single UNIFIED weighted progress bar (design §3 ② weights), a streaming
log tail, per-stage status, and cancel + resume-from-stage. Training progress is
taken from the trainer's `step=N` stdout (already parsed by GsplatBackend into the
progress callback). Delivery runs as a subprocess whose stdout is streamed.

Driveable two ways:
  * from Gradio (gui2/app.py polls `snapshot()` on a timer), and
  * programmatically (the UI-1 smoke calls `start()` then polls `snapshot()`),
so the runner has NO Gradio dependency.

State lives in a per-scan run dir (mirrors the SONY_TEST layout):
    <run_dir>/
      input_images/            raw images (or extracted video frames)
      apex_<scan>/             project dir (sfm/, feats h5, pipeline_state.json)
      export/                  3DGS layout (images/, sparse/0/)
      training/gsplat/data/    prepared data (sparse, images_<f>, priors/)
      training/gsplat/output/  ckpts/, ply/, stats/
      deliverables/            <scan>_<mode>.sog / .ply / _delivery.json
      run_state.json           runner status snapshot (resume source)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import gradio as gr  # noqa: F401  (stage handlers import/raise gr.*)

from core.pipeline_state import PipelineState
from core.utils import create_project_dir, get_image_list
from gui.stages.extraction import run_extraction
from gui.stages.matching import run_matching
from gui.stages.reconstruction import run_reconstruction
from gui.stages.export import run_export, run_refinement
from core.training import TrainingConfig, TrainingInterrupted, get_backend
from gui2 import defaults as D

_REPO = str(Path(__file__).resolve().parent.parent)

# Weighted stages (design §3 ②). Sum = 100.
STAGE_WEIGHTS: list[tuple[str, int]] = [
    ("qc", 2),
    ("features", 8),
    ("matching", 25),
    ("reconstruction", 20),
    ("export", 5),
    ("priors_dense", 10),
    ("training", 28),
    ("delivery", 2),
]
STAGE_LABELS = {
    "qc": "Capture check",
    "features": "Feature extraction",
    "matching": "Feature matching",
    "reconstruction": "3D reconstruction",
    "export": "Export",
    "priors_dense": "Priors + dense init",
    "training": "Training the splat",
    "delivery": "Delivering",
}
_TOTAL_WEIGHT = sum(w for _, w in STAGE_WEIGHTS)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    scan_name: str
    mode: str                       # "object" | "environment"
    input_dir: str                  # folder of images (already extracted if video)
    run_dir: str                    # <repo>/_dataset/<scan_name>
    settings: dict[str, Any]        # from defaults_for(mode) + advanced overrides
    source_mp: float = 0.0
    subject: str = ""
    detail: bool = False
    # smoke / speed knobs (production leaves these at defaults)
    max_steps_override: Optional[int] = None
    data_factor_override: Optional[int] = None
    delivery_no_qa: bool = False
    delivery_refit: Optional[bool] = None   # None = per-mode default; True/False overrides
    resume: bool = False


# ---------------------------------------------------------------------------
# Log tee — surface in-process print() from the stage handlers into the UI tail
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, real, sink: Callable[[str], None]):
        self._real = real
        self._sink = sink
        self._buf = ""

    def write(self, s: str):
        try:
            self._real.write(s)
        except Exception:
            pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._sink(line.rstrip())
        return len(s)

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """Runs the full chain in a background thread with unified progress."""

    def __init__(self, config: RunConfig):
        self.cfg = config
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._cancelled = threading.Event()
        self._active_proc: Optional[subprocess.Popen] = None
        self._backend = None

        self.status = "idle"        # idle|running|done|failed|cancelled
        self.message = ""
        self.current_stage = STAGE_WEIGHTS[0][0]
        # per-stage: {"status": pending|running|done|skipped|failed, "frac": 0..1, "desc": str}
        self.stages: dict[str, dict] = {
            k: {"status": "pending", "frac": 0.0, "desc": ""} for k, _ in STAGE_WEIGHTS
        }
        self._log = deque(maxlen=4000)
        self.deliverables: list[dict] = []
        self.export_dir: Optional[str] = None
        self.data_dir: Optional[str] = None
        self.ckpt_path: Optional[str] = None
        self.data_factor: int = 1
        self.max_steps: int = int(config.settings.get("max_steps", 30_000))
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    # ---- public API -------------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.status = "running"
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancelled.set()
        with self._lock:
            self.message = "Cancelling — stopping the current stage cleanly..."
        if self._backend is not None:
            try:
                self._backend.request_stop()
            except Exception:
                pass
        # Delivery spawns a NESTED self-refit trainer subprocess; a plain terminate on
        # the delivery process orphans it. Kill the whole process TREE.
        _terminate_tree(self._active_proc)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> dict:
        with self._lock:
            gfrac = self._global_frac()
            return {
                "status": self.status,
                "message": self.message,
                "current_stage": self.current_stage,
                "current_stage_label": STAGE_LABELS.get(self.current_stage, self.current_stage),
                "global_frac": gfrac,
                "global_pct": round(100.0 * gfrac, 1),
                "stages": {k: dict(v) for k, v in self.stages.items()},
                "stage_order": [k for k, _ in STAGE_WEIGHTS],
                "stage_labels": dict(STAGE_LABELS),
                "log_tail": "\n".join(list(self._log)[-40:]),
                "deliverables": list(self.deliverables),
                "elapsed_s": (time.time() - self.started_at) if self.started_at else 0.0,
            }

    # ---- progress helpers -------------------------------------------------

    def _global_frac(self) -> float:
        acc = 0.0
        for k, w in STAGE_WEIGHTS:
            st = self.stages[k]
            if st["status"] in ("done", "skipped"):
                acc += w
            elif st["status"] in ("running", "failed"):
                acc += w * max(0.0, min(1.0, st["frac"]))
        return acc / _TOTAL_WEIGHT

    def _log_line(self, line: str):
        with self._lock:
            self._log.append(line)

    def _begin(self, key: str, desc: str = ""):
        with self._lock:
            self.current_stage = key
            self.stages[key]["status"] = "running"
            self.stages[key]["frac"] = 0.0
            self.stages[key]["desc"] = desc
            self.message = f"{STAGE_LABELS.get(key, key)}: {desc}".rstrip(": ")
        self._log_line(f"=== {STAGE_LABELS.get(key, key)} ===")
        self._persist()

    def _tick(self, key: str, frac: float, desc: str = ""):
        with self._lock:
            self.stages[key]["frac"] = max(0.0, min(1.0, float(frac)))
            if desc:
                self.stages[key]["desc"] = desc
                self.message = f"{STAGE_LABELS.get(key, key)}: {desc}"

    def _done(self, key: str, desc: str = ""):
        with self._lock:
            self.stages[key]["status"] = "done"
            self.stages[key]["frac"] = 1.0
            if desc:
                self.stages[key]["desc"] = desc
        self._persist()

    def _skip(self, key: str, desc: str = "already present"):
        with self._lock:
            self.stages[key]["status"] = "skipped"
            self.stages[key]["frac"] = 1.0
            self.stages[key]["desc"] = desc
        self._log_line(f"--- {STAGE_LABELS.get(key, key)}: skipped ({desc}) ---")

    def _persist(self):
        try:
            snap = {
                "scan_name": self.cfg.scan_name, "mode": self.cfg.mode,
                "status": self.status, "current_stage": self.current_stage,
                "stages": self.stages, "deliverables": self.deliverables,
                "data_dir": self.data_dir, "export_dir": self.export_dir,
                "ckpt": self.ckpt_path, "data_factor": self.data_factor,
                "settings": self.cfg.settings,
            }
            Path(self.cfg.run_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.cfg.run_dir) / "run_state.json").write_text(
                json.dumps(snap, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

    def _check_cancel(self):
        if self._cancelled.is_set():
            raise _Cancelled()

    # ---- the run ----------------------------------------------------------

    def _run(self):
        real_stdout = sys.stdout
        sys.stdout = _Tee(real_stdout, self._log_line)
        try:
            self._run_inner()
            with self._lock:
                self.status = "done"
                self.finished_at = time.time()
                self.message = "Done. Your deliverables are ready in Step 4."
            self._persist()
        except _Cancelled:
            with self._lock:
                self.status = "cancelled"
                self.finished_at = time.time()
                self.message = ("Cancelled. You can resume from the last finished "
                                "stage.")
            self._persist()
        except Exception as e:
            tb = traceback.format_exc()
            self._log_line(tb)
            with self._lock:
                self.status = "failed"
                self.finished_at = time.time()
                self.stages[self.current_stage]["status"] = "failed"
                self.message = f"Failed in {self.current_stage}: {e}"
            self._persist()
        finally:
            sys.stdout = real_stdout

    def _run_inner(self):
        cfg = self.cfg
        run_dir = Path(cfg.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        resume = cfg.resume

        # ---- QC (already ran in Step 1; account for its weight instantly) ----
        self._begin("qc", "input verified")
        n_in = len(get_image_list(cfg.input_dir))
        self.data_factor = (cfg.data_factor_override
                            or D.resolve_data_factor(cfg.settings, cfg.source_mp))
        self.max_steps = int(cfg.max_steps_override or cfg.settings.get("max_steps", 30_000))
        self._log_line(f"[qc] {n_in} input images; data_factor={self.data_factor}; "
                       f"max_steps={self.max_steps}; mode={cfg.mode}")
        self._done("qc", f"{n_in} images")

        # ---- PipelineState / project dir ----
        state = PipelineState()
        state.image_dir = cfg.input_dir
        state.project_dir = str(create_project_dir(str(run_dir), cfg.scan_name))
        all_imgs = get_image_list(cfg.input_dir)
        state.image_list = [str(p) for p in all_imgs]
        state.excluded_images = []
        state.num_images = len(all_imgs)
        state.save()

        # ---- Stage 2: features ----
        self._begin("features", "ALIKED")
        if resume and _has(state.project_dir, "feats-*.h5"):
            state.discover_outputs()
            self._skip("features")
        else:
            cb = self._frac_cb("features")
            run_extraction("aliked", 8192, 3, False, 1600, False, state, cb)
            self._log_line(f"[features] {state.feature_path} "
                           f"(avg kpts {getattr(state, 'avg_keypoints', '?')})")
            self._done("features")
        self._check_cancel()

        # ---- Stage 3: matching (exhaustive <=300, else retrieval) ----
        self._begin("matching")
        strategy = "exhaustive" if state.num_images <= 300 else "retrieval"
        self._log_line(f"[matching] strategy={strategy} (n={state.num_images})")
        if resume and _has(state.project_dir, "*matches*.h5"):
            state.discover_outputs()
            self._skip("matching")
        else:
            cb = self._frac_cb("matching")
            run_matching(strategy, True, 0.1, 50, state, cb)
            self._log_line(f"[matching] pairs={getattr(state, 'num_pairs', '?')} "
                           f"avg_matches={getattr(state, 'avg_matches_per_pair', '?')}")
            self._done("matching")
        self._check_cancel()

        # ---- Stages 4-5: reconstruction + final BA ----
        self._begin("reconstruction", "incremental SfM")
        if resume and state.sfm_dir and _has_dir_bin(state.sfm_dir):
            state.discover_outputs()
            self._skip("reconstruction")
        else:
            cb = self._frac_cb("reconstruction", hi=0.8)
            run_reconstruction("incremental", "OPENCV", True, 15, 40, 100,
                               True, True, True, state, cb)
            reg = state.num_registered or 0
            self._log_line(f"[recon] registered {reg}/{state.num_images}, "
                           f"3D points {state.num_points3d}, "
                           f"reproj {state.mean_reprojection_error:.3f}px")
            self._tick("reconstruction", 0.85, "final bundle adjustment")
            run_refinement(200, True, True, True, state, self._frac_cb("reconstruction", lo=0.85))
            self._done("reconstruction")
        self._check_cancel()

        # ---- Stage 7: export ----
        self._begin("export", "3DGS layout")
        self.export_dir = str(run_dir / "export")
        if resume and _has_dir(Path(self.export_dir) / "sparse" / "0"):
            self._skip("export")
        else:
            cb = self._frac_cb("export")
            run_export(self.export_dir, "binary", True, "y", True, False, False,
                       "bw_masks", state, cb)
            self._log_line(f"[export] {state.export_dir}")
            self._done("export")
        self._check_cancel()

        # ---- Stage 6b: priors + dense-init (prepare_data does dense_init) ----
        self._begin("priors_dense", "preparing training data")
        self._backend = get_backend("gsplat")
        extra = D.assemble_training_extra(cfg.settings, self.data_factor, self.max_steps)
        train_dir = Path(state.project_dir) / "training" / "gsplat"
        train_dir.mkdir(parents=True, exist_ok=True)
        tconf = TrainingConfig(
            backend="gsplat", max_steps=self.max_steps,
            data_factor=self.data_factor, test_every=int(cfg.settings.get("test_every", 8)),
            save_steps=[max(1, self.max_steps // 4), self.max_steps], extra=extra,
        )
        self.data_dir = str(train_dir / "data")
        if resume and _prepared_ready(self.data_dir, self.data_factor,
                                      bool(extra.get("dense_init"))):
            self._log_line("[prepare] reusing prepared data dir (late-artifact check ok)")
        else:
            # prepare_data reports per-image progress; map onto 0..0.55 of this stage
            pcb = self._count_cb("priors_dense", lo=0.0, hi=0.55)
            self.data_dir = self._backend.prepare_data(
                export_dir=self.export_dir, mask_dir=None,
                output_dir=str(train_dir), config=tconf, progress_cb=pcb)
            self._log_line(f"[prepare] data_dir={self.data_dir} "
                           f"(dense_init={'on' if extra.get('dense_init') else 'off'})")
        self._tick("priors_dense", 0.6, "building MoGe-2 depth priors")
        self._check_cancel()
        if extra.get("mono_depth_lambda", 0.0) > 0:
            need = not _priors_ready(self.data_dir)
            if resume and not need:
                self._log_line("[priors] cache present")
            else:
                from core.priors import build_cache
                self._log_line(f"[priors] MoGe-2 cache @ factor {self.data_factor} ...")
                stats = build_cache(self.data_dir, factor=self.data_factor)
                self._log_line(f"[priors] built ({stats.get('mode', '?')} mode)")
        else:
            self._log_line("[priors] mono_depth off — skipping MoGe cache")
        self._done("priors_dense")
        self._check_cancel()

        # ---- Stage 8: training (with warm-start resume) ----
        # Resume-from-cancel: the trainer's --init_ckpt is a WARM-START (loads the
        # trained Gaussians, restarts the step counter at 1), so we warm-start from the
        # latest saved checkpoint and train only the REMAINING steps — instead of
        # throwing away all prior work and restarting from a fresh init at step 0.
        ckpt_dir = str(train_dir / "output" / "ckpts")
        init_ckpt: Optional[str] = None
        eff_max_steps = self.max_steps
        already_done = False
        if resume:
            latest = self._latest_ckpt(ckpt_dir)
            if latest:
                # file name ckpt_{step-1}_rank0.pt -> completed steps = (step-1)+1
                done_steps = int(Path(latest).name.split("_")[1]) + 1
                if done_steps >= self.max_steps:
                    self.ckpt_path, already_done = latest, True
                else:
                    init_ckpt = latest
                    eff_max_steps = self.max_steps - done_steps

        if already_done:
            self._begin("training", "already complete (resume)")
            self._log_line(f"[resume] training already at {self.max_steps} steps — "
                           f"reusing {Path(self.ckpt_path).name}")
            self._done("training", "resumed")
        else:
            # save_all=True writes intermediate checkpoints too (default backend
            # behavior writes ONLY the final ckpt), so a mid-training cancel is
            # resumable from the latest ckpt.
            train_extra = dict(extra)
            train_extra["save_all"] = True
            if init_ckpt:
                self._begin("training", f"resume: {eff_max_steps} steps remain")
                self._log_line(f"[resume] warm-start from {Path(init_ckpt).name}; "
                               f"{eff_max_steps} steps remain (densify continued, "
                               f"coarse off)")
                # keep MCMC densifying for the whole resumed run (init_ckpt otherwise
                # defaults refine_stop=1 = no growth) and skip coarse on a warm model.
                train_extra.update(init_ckpt=init_ckpt, coarse_factor=0,
                                   refine_stop_iter=eff_max_steps)
            else:
                self._begin("training", f"0/{eff_max_steps} steps")
            q = max(1, eff_max_steps // 4)
            save_steps = sorted(set([q, 2 * q, 3 * q, eff_max_steps]))
            train_conf = TrainingConfig(
                backend="gsplat", max_steps=eff_max_steps,
                data_factor=self.data_factor,
                test_every=int(cfg.settings.get("test_every", 8)),
                save_steps=save_steps, extra=train_extra)
            tcb = self._train_cb("training")
            try:
                result = self._backend.train(
                    data_dir=self.data_dir, output_dir=str(train_dir),
                    config=train_conf, progress_cb=tcb)
            except TrainingInterrupted:
                raise _Cancelled()
            # Race guard: request_stop() terminates the trainer, which can EOF the
            # stdout stream so backend.train RETURNS normally (progress-cb stop not
            # hit). Re-check cancel so we don't fall through into delivery.
            self._check_cancel()
            self.ckpt_path = self._latest_ckpt(result.checkpoint_dir)
            self._log_line(f"[train] done: {result.steps_completed} steps, "
                           f"psnr={result.psnr:.2f}, ckpt={self.ckpt_path}")
            self._done("training", f"{result.steps_completed} steps")
        self._check_cancel()

        # ---- Stage 9: delivery ----
        self._begin("delivery", f"{cfg.mode} mode")
        self._run_delivery()
        self._done("delivery")

    # ---- delivery (subprocess, streamed) ----------------------------------

    def _run_delivery(self):
        cfg = self.cfg
        out_dir = str(Path(cfg.run_dir) / "deliverables")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if not self.ckpt_path or not os.path.isfile(self.ckpt_path):
            raise RuntimeError(f"no checkpoint to deliver ({self.ckpt_path})")
        cmd = [sys.executable, "-m", "core.delivery",
               "--ckpt", self.ckpt_path, "--data_dir", self.data_dir,
               "--name", cfg.scan_name, "--factor", str(self.data_factor),
               "--out", out_dir, "--gpu", "0"]
        cmd += D.assemble_delivery_flags(cfg.settings)
        if cfg.delivery_refit is not None:
            cmd.append("--refit" if cfg.delivery_refit else "--no-refit")
        if cfg.delivery_no_qa:
            cmd.append("--no_qa")
        self._log_line("[delivery] " + " ".join(cmd[1:]))
        markers = {"[master]": 0.15, "[extract]": 0.25, "[prune]": 0.4,
                   "[reduce]": 0.55, "[refit]": 0.7, "[deliver]": 0.85, "[qa]": 0.95}
        env = dict(os.environ)
        env["PYTHONPATH"] = _REPO + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.Popen(cmd, cwd=_REPO, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        self._active_proc = proc
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self._log_line(line)
                for m, f in markers.items():
                    if m in line:
                        self._tick("delivery", f)
                        break
                if self._cancelled.is_set():
                    _terminate_tree(proc)      # kill delivery + nested refit trainer
                    raise _Cancelled()
            proc.wait()
        finally:
            self._active_proc = None
        if proc.returncode != 0:
            # A cancel tree-kills the delivery process -> non-zero exit; surface it as a
            # cancellation, not a failure.
            if self._cancelled.is_set():
                raise _Cancelled()
            raise RuntimeError(f"delivery failed (exit {proc.returncode})")
        self._collect_deliverables(out_dir)

    def _collect_deliverables(self, out_dir: str):
        items = []
        for p in sorted(Path(out_dir).glob("*")):
            if p.suffix.lower() in (".sog", ".ply") and p.is_file():
                items.append({"name": p.name, "path": str(p),
                              "size_mb": round(p.stat().st_size / 1e6, 2)})
        with self._lock:
            self.deliverables = items
        self._log_line(f"[delivery] {len(items)} deliverable(s) in {out_dir}")

    # ---- progress callback adapters ---------------------------------------

    def _frac_cb(self, key: str, lo: float = 0.0, hi: float = 1.0):
        """Adapter for the gr.Progress-style stage handlers: progress(frac, desc=...)."""
        def cb(frac=0.0, desc="", *a, **k):
            try:
                f = float(frac)
            except (TypeError, ValueError):
                f = 0.0
            self._tick(key, lo + (hi - lo) * max(0.0, min(1.0, f)), str(desc or ""))
        return cb

    def _count_cb(self, key: str, lo: float, hi: float):
        """Adapter for prepare_data-style progress_cb: (cur, tot, loss, psnr)."""
        def cb(cur=0, tot=1, *a, **k):
            try:
                frac = float(cur) / max(float(tot), 1.0)
            except (TypeError, ValueError):
                frac = 0.0
            self._tick(key, lo + (hi - lo) * max(0.0, min(1.0, frac)),
                       f"image {cur}/{tot}")
        return cb

    def _train_cb(self, key: str):
        """Adapter for backend.train progress_cb: (step, max_steps, loss, psnr)."""
        def cb(step=0, total=1, loss=0.0, psnr=0.0):
            if self._cancelled.is_set():
                raise TrainingInterrupted("cancel requested")
            frac = float(step) / max(float(total), 1.0)
            self._tick(key, frac, f"{step}/{total} steps")
            self._log_line(f"step={step} loss={loss:.5f} psnr={psnr:.2f}")
        return cb

    def _latest_ckpt(self, ckpt_dir: Optional[str]) -> Optional[str]:
        if not ckpt_dir:
            return None
        cks = [c for c in Path(ckpt_dir).glob("ckpt_*_rank0.pt")
               if "pruned" not in c.name]
        if not cks:
            return None

        def _step(p: Path) -> int:
            try:
                return int(p.name.split("_")[1])
            except (IndexError, ValueError):
                return -1
        return str(max(cks, key=_step))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _Cancelled(Exception):
    pass


def _has(project_dir: Optional[str], pattern: str) -> bool:
    if not project_dir:
        return False
    return bool(list(Path(project_dir).glob(pattern)))


def _has_dir(path) -> bool:
    return Path(path).is_dir()


def _has_dir_bin(path: str) -> bool:
    return Path(path).is_dir() and bool(list(Path(path).glob("*.bin")))


def _priors_ready(data_dir: str) -> bool:
    p = Path(data_dir) / "priors"
    return (p / "align.json").is_file() and (p / "depth").is_dir()


def _prepared_ready(data_dir: str, factor: int, dense_init: bool) -> bool:
    """Late-artifact probe for a COMPLETE prepare_data (not just sparse/0, which is
    copied FIRST). Requires the factor-downsampled images dir to be populated (created
    late) and — if dense_init was on — its completion marker
    (<sparse>/_dense_init_work/dense_init_stats.json)."""
    dd = Path(data_dir)
    imgs = dd / f"images_{factor}"
    if not imgs.is_dir() or not any(imgs.iterdir()):
        return False
    if dense_init:
        marker = dd / "sparse" / "0" / "_dense_init_work" / "dense_init_stats.json"
        if not marker.is_file():
            return False
    return True


def _terminate_tree(proc: Optional[subprocess.Popen], timeout: float = 10.0) -> None:
    """Terminate a subprocess AND all its descendants (delivery spawns a nested
    self-refit trainer). Windows: taskkill /T /F kills the tree; elsewhere fall back to
    psutil if present, then terminate()->wait->kill escalation on the parent."""
    if proc is None or proc.poll() is not None:
        return
    pid = proc.pid
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                           capture_output=True, timeout=timeout)
        except Exception:
            pass
    else:
        try:
            import psutil  # optional
            parent = psutil.Process(pid)
            for ch in parent.children(recursive=True):
                try:
                    ch.terminate()
                except Exception:
                    pass
        except Exception:
            pass
    # Escalation on the parent handle regardless of platform
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
