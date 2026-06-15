"""Difix3D+ inference runner — executes INSIDE the isolated 'difix' conda env.

Standalone (no apex-sfm imports): only stdlib + PIL + the Difix3D repo's
pipeline + torch/diffusers (provided by the difix env). Invoked as a subprocess
by core/enhancement/difix_backend.DifixBackend so the pinned diffusers/transformers
stay isolated from the main env. Native — no Docker/WSL.

Usage (run by the backend):
    conda run -n difix python difix_runner.py --input <dir> --output <dir> \
        --repo <Difix3D path> [--model nvidia/difix] [--timestep 199]
"""
import argparse
import os
import sys
from glob import glob

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--repo", required=True, help="Path to the nv-tlabs/Difix3D clone")
    ap.add_argument("--model", default="nvidia/difix")
    ap.add_argument("--prompt", default="remove degradation")
    ap.add_argument("--timestep", type=int, default=199)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, "src"))
    from pipeline_difix import DifixPipeline  # from the Difix3D repo

    os.makedirs(args.output, exist_ok=True)
    pipe = DifixPipeline.from_pretrained(args.model, trust_remote_code=True).to("cuda")

    files = sorted(
        glob(os.path.join(args.input, "*.png"))
        + glob(os.path.join(args.input, "*.jpg"))
        + glob(os.path.join(args.input, "*.jpeg"))
    )
    print(f"[difix_runner] {len(files)} images", flush=True)
    for f in files:
        img = Image.open(f).convert("RGB")
        res = pipe(args.prompt, image=img, num_inference_steps=1,
                   timesteps=[args.timestep], guidance_scale=0.0).images[0]
        if res.size != img.size:
            res = res.resize(img.size, Image.BICUBIC)
        res.save(os.path.join(args.output,
                              os.path.splitext(os.path.basename(f))[0] + ".png"))
    print("DIFIX_RUNNER_DONE", flush=True)


if __name__ == "__main__":
    main()
