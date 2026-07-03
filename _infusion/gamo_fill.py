"""GaMO-pattern gated generative completion — per-view fill runner (WP-C1 Part 2).

Runs in the WSL artifixer-venv (diffusers 0.37.1). Backbone v1 = SDXL-Inpaint
(ungated FLUX.1-Fill was NOT provisioned on this machine — no weights/env; the
conda plumbing in core/enhancement/inpaint_backend.py points at a migrated-away
box. SDXL-Inpaint is mask-native, ungated, DDPM -> a clean scheduler.add_noise
path for the GaMO latent injection). The backbone-agnostic gate mechanism lives in
core/enhancement/latent_gate.py.

For each hole-crossing view it consumes the holed render + OGGSplat alpha and:
  GATED   = latent-gate injection (noised render latent composited at OBSERVED
            latents at 30/50/70% of denoising, progressive dilation, widest first)
            + feathered pixel paste-back (INSET 0.6, FEATHER 0.08) -> observed
            pixels byte-identical to the render (hard gate integrity).
  UNGATED = the SAME generator/inference, NO latent injection, NO paste-back -> the
            raw VAE-decoded frame (observed pixels drift via the VAE round-trip +
            free generation). Isolates exactly what the gate buys.

RePaint 3-iter resample: v1 approximates it with the SDXL iterative refinement +
the trigger-point injection + feather (a callback cannot re-run a sampler step);
noted as the one deviation from the full mechanism.

Usage (from WSL):
  ~/artifixer-venv/bin/python gamo_fill.py --arc <dir> --out <dir> \
     --mode gated|ungated [--steps 30] [--res 1024] [--seed 0]
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

# repo root on path -> import the torch-only gate mechanism (no diffusers there)
REPO = "/mnt/d/REPOS/HYPERclone/apex-sfm"
sys.path.insert(0, REPO)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "latent_gate", os.path.join(REPO, "core/enhancement/latent_gate.py"))
lg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lg)

PROMPT = ("the continuation of the scene surface, same material, texture, colour "
          "and lighting, photorealistic, seamless, sharp focus")
NEG = "blurry, distorted, artifacts, seam, watermark, text"


def load_pipe(dtype):
    from diffusers import StableDiffusionXLInpaintPipeline, DDPMScheduler
    mid = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(mid, torch_dtype=dtype,
                                                            variant=None)
    # DDPM scheduler -> clean add_noise for the GaMO known-region injection
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    # NOTE: do NOT globally cast the VAE — the SDXL inpaint pipeline auto-upcasts the
    # VAE to fp32 for its internal encode/decode (config.force_upcast) and casts back;
    # a permanent fp32 VAE would mismatch the fp16 activations the pipeline feeds it.
    # Our manual render-encode upcasts locally in encode_render().
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe


def encode_render(pipe, rgb_uint8, W, H):
    """VAE-encode a render (resized to WxH) -> latent (1,4,H/8,W/8) in pipe dtype.

    Locally upcasts the VAE to fp32 for the encode (SDXL fp16 VAE encode NaNs),
    then restores the pipe dtype -- mirrors the pipeline's own force_upcast dance."""
    img = cv2.resize(rgb_uint8, (W, H), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 127.5 - 1.0   # [-1,1]
    t = t.to("cuda", torch.float32)
    pipe_dtype = pipe.unet.dtype
    with torch.no_grad():
        pipe.vae.to(torch.float32)
        lat = pipe.vae.encode(t).latent_dist.mean * pipe.vae.config.scaling_factor
        pipe.vae.to(pipe_dtype)
    return lat.to(pipe_dtype)


def fill_one(pipe, render_bgr, mask_u8, alpha, mode, steps, res, seed):
    H0, W0 = render_bgr.shape[:2]
    render_rgb = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB)
    # SDXL working resolution (long edge = res, /8)
    sc = res / max(H0, W0)
    Wg = int(round(W0 * sc / 8)) * 8
    Hg = int(round(H0 * sc / 8)) * 8
    init = Image.fromarray(cv2.resize(render_rgb, (Wg, Hg), interpolation=cv2.INTER_AREA))
    mask_g = cv2.resize(mask_u8, (Wg, Hg), interpolation=cv2.INTER_NEAREST)
    maskimg = Image.fromarray(mask_g)                        # white=void=inpaint
    lat_h, lat_w = Hg // 8, Wg // 8

    g = torch.Generator("cuda").manual_seed(seed)
    callback = None
    if mode == "gated":
        # NOTE — two coexisting mask definitions in this pipeline (both intentional):
        #   (1) the SDXL pipeline INPAINT mask (`mask_u8`, white=void) = the OGGSplat
        #       unobserved mask (alpha<0.3 + morphology, exported by hole_export.py) —
        #       drives which latents the inpaint UNet is allowed to synthesise, and the
        #       feathered paste-back region below.
        #   (2) the LATENT-GATE mask (built here from raw alpha<0.6 per GaMO's spec) —
        #       drives the known-region latent re-injection. alpha<0.6 is a CONSERVATIVE
        #       SUPERSET of alpha<0.3 (more pixels flagged "generate"), i.e. the gate
        #       protects a slightly SMALLER observed set than the inpaint mask carves —
        #       the safe direction (never injects known content over a void pixel). The
        #       hard observed-pixel guarantee is enforced by the OGGSplat-mask paste-back.
        alpha_t = torch.from_numpy(cv2.resize(alpha, (Wg, Hg), interpolation=cv2.INTER_AREA))
        m = lg.build_latent_mask(alpha_t, lat_h, lat_w, alpha_thresh=0.6, device="cuda")
        pmasks = lg.progressive_masks(m, k=5, iters=(2, 1, 0))     # widest first
        x0 = encode_render(pipe, cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB), Wg, Hg)
        noise = torch.randn(x0.shape, generator=torch.Generator("cuda").manual_seed(seed + 7),
                            device="cuda", dtype=torch.float32)
        trig = lg.trigger_steps(steps, fracs=(0.30, 0.50, 0.70))

        def _cb(pipe_, i, t, cbk):
            if i in trig:
                lat = cbk["latents"]
                mm = pmasks[trig[i]].to(lat.dtype)
                # `lat` is the scheduler.step OUTPUT — the sample at the NEXT timestep
                # timesteps[i+1], NOT at `t` (=timesteps[i], the step just completed).
                # Noise the known latent to timesteps[i+1] so it matches `lat`'s noise
                # level (mirrors the installed SDXL-inpaint pipeline's own known-region
                # injection: `noise_timestep = timesteps[i+1]`). Using `t` here is an
                # off-by-one that composites a slightly-too-noisy known latent.
                ts = pipe_.scheduler.timesteps
                nt = ts[i + 1] if (i + 1) < len(ts) else t
                xk = pipe_.scheduler.add_noise(x0.to(lat.dtype), noise.to(lat.dtype),
                                               nt.reshape(1))
                cbk["latents"] = lg.composite(lat, xk, mm)
            return cbk
        callback = _cb

    kw = dict(prompt=PROMPT, negative_prompt=NEG, image=init, mask_image=maskimg,
              height=Hg, width=Wg, num_inference_steps=steps, guidance_scale=7.5,
              strength=0.999, generator=g)
    if callback is not None:
        kw["callback_on_step_end"] = callback
        kw["callback_on_step_end_tensor_inputs"] = ["latents"]
    with torch.inference_mode():
        out = pipe(**kw).images[0]
    gen = cv2.resize(np.asarray(out), (W0, H0), interpolation=cv2.INTER_CUBIC)   # RGB full-res

    if mode == "gated":
        unobs = mask_u8 > 127
        comp = lg.feather_paste_back(render_rgb, gen, unobs, inset=0.6, feather=0.08)
        return cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(gen, cv2.COLOR_RGB2BGR)                                  # ungated: raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc", required=True, help="dir with vXXXX_render.png / _mask.png / _alpha.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["gated", "ungated"], default="gated")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = json.load(open(os.path.join(args.arc, "meta.json")))
    dtype = torch.float32 if args.fp32 else torch.float16
    pipe = load_pipe(dtype)
    print(f"[gamo] SDXL-Inpaint loaded ({dtype}); mode={args.mode} steps={args.steps} res={args.res}")

    stems = sorted(meta.keys())
    for k, stem in enumerate(stems):
        render = cv2.imread(os.path.join(args.arc, f"{stem}_render.png"))
        mask = cv2.imread(os.path.join(args.arc, f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        alpha = np.load(os.path.join(args.arc, f"{stem}_alpha.npy"))
        fill = fill_one(pipe, render, mask, alpha, args.mode, args.steps, args.res, args.seed)
        cv2.imwrite(os.path.join(args.out, f"{stem}_fill.png"), fill)
        if k % 5 == 0 or k == len(stems) - 1:
            print(f"  [{k+1}/{len(stems)}] {stem} filled", flush=True)
    print(f"[gamo] {len(stems)} fills -> {args.out}")


if __name__ == "__main__":
    main()
