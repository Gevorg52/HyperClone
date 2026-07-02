import sys
sys.argv = ["x"]
import eval_compare as E

base = r"D:\REPOS\HYPERclone\apex-sfm\_dataset\GPU_V4\apex_images\training"
data_dir = base + r"\gsplat_prod\data"
ckpts = {
    "factor2": base + r"\gsplat_prod\output\ckpts\ckpt_29999_rank0.pt",
    "native":  base + r"\gsplat_native\output\ckpts\ckpt_29999_rank0.pt",
}

for F in (1, 2):
    E.FACTOR = F
    data = E.load_colmap_data(data_dir, factor=F, test_every=8, normalize=True)
    print(f"\n=== Eval @ factor {F} ({'24MP' if F==1 else '6MP'}) on {len(data['test_indices'])} held-out views ===")
    print(f"{'model':10s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s}")
    for label, ck in ckpts.items():
        s = E.load_splats(ck)
        p, q, l, d = E.evaluate(s, data)
        print(f"{label:10s} {p:7.3f} {q:7.4f} {l:7.4f}")
