import os
f = os.path.expanduser("~/artifixer/model_training/net/transformer.py")
src = open(f).read()
old = "flex_attention = torch.compile(_flex_attention_with_options, dynamic=False)"
new = (
    "import torch._dynamo as _apex_dynamo\n"
    "_apex_dynamo.config.recompile_limit = 64  # PATCH(apex/sm_120): avoid eager-flex fallback after 8 shapes\n"
    "flex_attention = torch.compile(_flex_attention_with_options, dynamic=True)  # PATCH(apex/sm_120): dynamic shapes -> no per-frame recompile (pytorch#141486)"
)
if "PATCH(apex/sm_120): dynamic shapes" in src:
    print("ALREADY_PATCHED")
elif old in src:
    open(f + ".dyn_orig", "w").write(src)
    open(f, "w").write(src.replace(old, new, 1))
    print("PATCHED dynamic=True + recompile_limit=64")
else:
    print("TARGET_NOT_FOUND -- line changed?")
