"""WP-G1 stage-1: render a trained 3DGS-DR model's held-out (every-8) test views
and save them as <image_name>.png keyed by name, plus a manifest. Runs against the
3DGS-DR codebase (deferred-reflection c7 rasterizer + cubemap env map).

Must be launched with the 3DGS-DR repo on sys.path (done below). model_path (-m)
must contain the trained point_cloud + cfg_args (written by their train.py).
GT is each test camera's original_image (= our exported undistorted frame), but we
re-evaluate metrics in stage-2 against the canonical load_image GT for a clean A/B.

Usage (any cwd):
  python _infusion/wpg1_render_dr.py -m <out_dir> --out_dir <render_dir> [--iteration N]
"""
import os
import sys
import json

_DR = r"D:\REPOS\HYPERclone\3DGS-DR"
sys.path.insert(0, _DR)

import torch  # noqa: E402
import torchvision  # noqa: E402
from argparse import ArgumentParser  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from gaussian_renderer import render, render_env_map  # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--out_dir", required=True)
    args = get_combined_args(parser)
    safe_state(True)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")  # black bg

        # env-map preview (sanity that reflections were learned)
        try:
            lt = render_env_map(gaussians)
            torchvision.utils.save_image(lt["env_cood1"], os.path.join(args.out_dir, "_envmap1.png"))
            torchvision.utils.save_image(lt["env_cood2"], os.path.join(args.out_dir, "_envmap2.png"))
        except Exception as e:
            print("env-map preview failed:", e)

        manifest = []
        test_cams = scene.getTestCameras()
        n_refl = int((gaussians.get_refl > 0.1).sum().item())
        for view in test_cams:
            view.refl_mask = None
            pkg = render(view, gaussians, pipe, bg)
            r = pkg["render"].clamp(0, 1)
            name = view.image_name + ".png"
            torchvision.utils.save_image(r, os.path.join(args.out_dir, name))
            # also dump refl_strength + normal maps for the first few (visual)
            manifest.append({"image_name": view.image_name})
        json.dump({"iteration": scene.loaded_iter,
                   "n_gaussians": int(gaussians.get_xyz.shape[0]),
                   "n_refl_gt_0.1": n_refl,
                   "n_test": len(test_cams),
                   "views": manifest},
                  open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=2)
    print(f"WPG1 render: {len(test_cams)} test views -> {args.out_dir} "
          f"(iter {scene.loaded_iter}, {gaussians.get_xyz.shape[0]} gauss, {n_refl} refl>0.1)")


if __name__ == "__main__":
    main()
