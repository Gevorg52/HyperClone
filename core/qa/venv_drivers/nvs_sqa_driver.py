"""NVS-SQA driver — RUNS INSIDE the qa venv.

Loads the NVS-SQA sequence encoder (18 MB in-repo ckpt), extracts one feature
vector for a folder of SORTED frames, and applies the linear head to get a JOD
score (higher = better; WITHIN-scene deltas only). The original head is a pickled
sklearn Ridge (sklearn<1.7 pickle); we extracted its coef_/intercept_ to a plain
`linear_head.npz` ONCE (see THIRD_PARTY.md) so runtime needs no sklearn and is
fully deterministic: jod = feat @ coef + intercept.

Protocol: `python nvs_sqa_driver.py <input.json> <output.json>`, input.json =
  {"repo": ".../NVS-SQA", "model_folder": ".../checkpoints/v0.0.1/models",
   "linear_head": ".../linear_head.npz", "frames_dir": ".../sorted_frames",
   "epo_offset": 100}
output.json = {"jod": <float>, "feat_dim": 384, "n_frames": N}
"""
import glob
import json
import os
import sys

import numpy as np
import torch


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    cfg = json.load(open(in_path))
    repo = cfg["repo"]
    model_folder = cfg["model_folder"]
    head_path = cfg["linear_head"]
    frames_dir = cfg["frames_dir"]
    epo = int(cfg.get("epo_offset", 100))

    os.chdir(repo)                       # repo uses relative `from models import ...`
    sys.path.insert(0, repo)
    # Import the encoder module DIRECTLY (not utils.train_utils): train_utils pulls
    # in contrast_learning -> qa_methods -> skvideo (scikit-video), which is a
    # training-only dep that breaks under numpy 2. sequence_models + encoders only
    # need torch/torchvision/torchsummary/PIL/cv2. We replicate load_model's
    # essential encoder path here.
    import glob                          # noqa: E402
    from models import sequence_models   # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfgs = sorted(glob.glob(os.path.join(model_folder, "SeqModel*.json")))
    if not cfgs:
        raise FileNotFoundError(f"no SeqModel*.json config in {model_folder}")
    encoder = sequence_models.EfficientSequenceModel.load_from_config(cfgs[0]).to(device)
    encoder_path = os.path.join(model_folder, f"{encoder.model_name}_{epo}.pth")
    encoder.load_state_dict(torch.load(encoder_path, map_location=device,
                                       weights_only=False))
    encoder.eval()

    view_paths = sorted(p for p in glob.glob(os.path.join(frames_dir, "*.*"))
                        if p.rsplit(".", 1)[-1].lower() in ("png", "jpg", "jpeg"))
    if len(view_paths) < 2:
        raise ValueError(f"NVS-SQA needs >=2 sorted frames, got {len(view_paths)}")

    with torch.no_grad():
        feat = encoder.predict_features_from_view_paths(view_paths, device)
    feat = feat.detach().cpu().numpy().ravel().astype(np.float64)

    head = np.load(head_path)
    coef = head["coef"].astype(np.float64).ravel()
    intercept = float(head["intercept"])
    if coef.shape[0] != feat.shape[0]:
        raise ValueError(f"feat dim {feat.shape[0]} != head dim {coef.shape[0]}")
    jod = float(feat @ coef + intercept)

    out = {"jod": jod, "feat_dim": int(feat.shape[0]), "n_frames": len(view_paths)}
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[nvs_sqa_driver] JOD={jod:.4f} over {len(view_paths)} frames")


if __name__ == "__main__":
    main()
