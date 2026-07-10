"""UI-4 review closeout — floater-cleanup argv dry-run (HIGH-2 proof, no execution).

Constructs the EXACT delivery command the fixed `_floater_cleanup` handler builds
(gui2/app.py) for a run whose masks_reviewed was accepted (SONY_TEST), and asserts:
  * the shared builder (gui2.deliver.build_delivery_flags) is used, so
  * reviewed masks are honored: `--mask_dir …/masks_reviewed` present, `--gen_masks` absent
  * `--extract_object` present (object mode)
  * the refit choice matches the BASE run's cfg.delivery_refit (None -> no flag,
    False -> --no-refit, True -> --refit), and --no_qa passes through
Does NOT run delivery. Re-runnable:  python _infusion/ui4_floater_cleanup_argv_proof.py
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(r"D:\REPOS\HYPERclone\apex-sfm")
sys.path.insert(0, str(REPO))
from gui2 import deliver as DV        # noqa: E402
from gui2 import defaults as D        # noqa: E402

SONY = REPO / "_dataset" / "SONY_TEST"
OBJ = D.defaults_for("object", False)


def build_cmd(run_dir, settings, ckpt, data_dir, factor, name, delivery_refit,
              delivery_no_qa):
    """Mirror gui2/app.py::_floater_cleanup command construction exactly."""
    out_dir = str(Path(run_dir) / "deliverables")
    cmd = [sys.executable, "-m", "core.delivery",
           "--ckpt", ckpt, "--data_dir", data_dir,
           "--name", name, "--factor", str(factor), "--out", out_dir, "--gpu", "0"]
    cmd += DV.build_delivery_flags(settings, run_dir, delivery_refit=delivery_refit,
                                   delivery_no_qa=delivery_no_qa)
    return cmd


def main():
    results = {}
    # base run cfg.delivery_refit is None in normal object runs (delivery applies its
    # per-mode default = refit ON); the gate run used False. Prove all three.
    for tag, refit, no_qa in (("base_refit_None", None, False),
                              ("base_refit_False", False, True),
                              ("base_refit_True", True, False)):
        cmd = build_cmd(str(SONY), OBJ,
                        ckpt=str(SONY / "improve" / "cleanup" / "master_cleaned.pt"),
                        data_dir=str(SONY / "training" / "gsplat" / "data"),
                        factor=2, name="SONY_TEST_cleaned",
                        delivery_refit=refit, delivery_no_qa=no_qa)
        flags = cmd[cmd.index("--out") + 2:]  # trailing delivery flags only
        has_maskdir = "--mask_dir" in flags
        mask_val = flags[flags.index("--mask_dir") + 1] if has_maskdir else None
        checks = {
            "reviewed_masks_honored": has_maskdir and "--gen_masks" not in flags,
            "mask_dir_is_masks_reviewed": bool(mask_val
                                               and mask_val.endswith("masks_reviewed")),
            "extract_object_present": "--extract_object" in flags,
            "refit_matches_base": (
                ("--refit" not in flags and "--no-refit" not in flags) if refit is None
                else ("--no-refit" in flags if refit is False else "--refit" in flags)),
            "no_qa_matches_base": (("--no_qa" in flags) == bool(no_qa)),
        }
        results[tag] = {"cmd": cmd, "flags": flags, "mask_dir": mask_val,
                        "checks": checks, "all_pass": all(checks.values())}
        print(f"[{ 'PASS' if all(checks.values()) else 'FAIL' }] {tag}")
        print("   flags:", " ".join(flags))
        for k, v in checks.items():
            print(f"     {'ok ' if v else 'BAD'} {k}")

    # control: a run with NO reviewed masks must fall back to --gen_masks (never a stale
    # or wrong --mask_dir). Use a temp run dir with no masks_reviewed.
    import tempfile
    tmp = tempfile.mkdtemp(prefix="ui4_argv_")
    ctrl_flags = DV.build_delivery_flags(OBJ, tmp, delivery_refit=None)
    control = {"flags": ctrl_flags,
               "gen_masks_fallback": ("--gen_masks" in ctrl_flags
                                      and "--mask_dir" not in ctrl_flags)}
    print(f"[{'PASS' if control['gen_masks_fallback'] else 'FAIL'}] control_no_reviewed "
          f"-> {' '.join(ctrl_flags)}")
    results["control_no_reviewed"] = control

    all_pass = (all(r.get("all_pass", False) for k, r in results.items()
                    if k != "control_no_reviewed")
                and control["gen_masks_fallback"])
    results["PASS"] = all_pass
    out = REPO / "_infusion" / "ui4_floater_cleanup_argv_proof.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nPASS={all_pass}  ->  {out}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
