"""Subprocess runner: masked image inpainting for the Type-3 (unseen-region) path.

Runs in an isolated env with modern diffusers (the `zimage` conda env). Fills the
WHITE region of the mask with plausible content consistent with the surroundings,
then composites the original UNMASKED pixels back over the result so ONLY the hole
changes (true inpainting; diffusion VAE round-trip can otherwise drift untouched
pixels — see arXiv:2512.05198).

Default model: Z-Image-Turbo (Tongyi-MAI, Apache-2.0, ~6B, fits 24GB). The model id
is a CLI arg so FLUX.1-Fill / Qwen-Image-Inpaint can be dropped in later.

Usage:
  python inpaint_runner.py --image rgb.png --mask mask.png --out out.png \
      --prompt "..." [--model Tongyi-MAI/Z-Image-Turbo] [--steps 8] [--guidance 1.0]
Mask convention: WHITE (255) = inpaint, BLACK (0) = preserve.
"""
import argparse

import numpy as np
import torch
from PIL import Image


def _load_flux_fp8(model_id, dtype):
    """FLUX.1-Fill at FP8 (optimum-quanto) + CPU offload -> fits the 4090's 24GB.
    12B transformer + T5-XXL quantized to qfloat8; offload keeps only the active
    module resident. Full BF16 would need ~96GB (the Pro 6000)."""
    from optimum.quanto import freeze, qfloat8, quantize
    from diffusers import FluxFillPipeline, FluxTransformer2DModel
    from transformers import T5EncoderModel
    tr = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=dtype)
    quantize(tr, weights=qfloat8); freeze(tr)
    te2 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype)
    quantize(te2, weights=qfloat8); freeze(te2)
    pipe = FluxFillPipeline.from_pretrained(model_id, transformer=tr, text_encoder_2=te2, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()         # offload manages devices; do NOT .to(cuda)
    return pipe


def load_pipe(model_id, dtype):
    """Load an inpainting pipeline. FLUX.1-Fill gets the FP8 path; otherwise iterate
    the known diffusers inpaint classes (Z-Image / Qwen / SDXL / Auto)."""
    if "flux" in model_id.lower():
        return _load_flux_fp8(model_id, dtype), "FluxFillPipeline(fp8)"
    import diffusers
    for cls_name in ("ZImageInpaintPipeline", "QwenImageInpaintPipeline",
                     "AutoPipelineForInpainting"):
        cls = getattr(diffusers, cls_name, None)
        if cls is None:
            continue
        try:
            pipe = cls.from_pretrained(model_id, torch_dtype=dtype)
            return pipe.to("cuda"), cls_name
        except Exception:
            continue
    raise RuntimeError(f"No diffusers inpaint pipeline could load model '{model_id}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="the continuation of the object surface, "
                    "same material, texture and lighting, photorealistic, seamless")
    ap.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe, cls_name = load_pipe(args.model, dtype)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    print(f"loaded {cls_name} ({args.model})", flush=True)

    img = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L").resize(img.size)
    W, H = img.size
    # round to /16 for the VAE
    Wp, Hp = (W // 16) * 16, (H // 16) * 16
    img_p = img.resize((Wp, Hp)); mask_p = mask.resize((Wp, Hp))

    g = torch.Generator(device="cuda").manual_seed(args.seed)
    with torch.inference_mode():
        out = pipe(prompt=args.prompt, image=img_p, mask_image=mask_p,
                   num_inference_steps=args.steps, guidance_scale=args.guidance,
                   generator=g).images[0]
    out = out.resize((W, H))

    # composite: keep original pixels where mask is black; only fill the white hole
    o = np.asarray(out).astype(np.float32)
    base = np.asarray(img).astype(np.float32)
    m = (np.asarray(mask).astype(np.float32) / 255.0)[..., None]
    m = np.clip((m - 0.5) * 6 + 0.5, 0, 1)            # sharpen the mask edge a touch
    comp = (m * o + (1 - m) * base).astype(np.uint8)
    Image.fromarray(comp).save(args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
