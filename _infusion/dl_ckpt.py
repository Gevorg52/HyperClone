"""Download the InFusion depth-inpainting diffusers checkpoint (safetensors only)."""
from huggingface_hub import snapshot_download

p = snapshot_download(
    repo_id="Johanan0528/Infusion",
    local_dir=r"F:\HYPERclone\infusion\checkpoints\Infusion",
    allow_patterns=["*.safetensors", "*.json", "*.txt"],
    ignore_patterns=["*.bin"],
)
print("DONE", p)
