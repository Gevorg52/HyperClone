#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
python3 - <<'PY'
import cudnn
doc = cudnn.pygraph.sdpa.__doc__ or ""
# print only lines mentioning masking-relevant params
import re
print("=== sdpa docstring (mask-relevant lines) ===")
for ln in doc.splitlines():
    if re.search(r"mask|band|window|sink|causal|bias|diag|block|left|right|score|ragged|seq_len", ln, re.I):
        print(ln.strip()[:160])
print("\n=== does Tensor/graph expose diagonal-band or block-mask setters? ===")
for obj,label in [(cudnn.pygraph,"pygraph")]:
    ms=[m for m in dir(obj) if re.search(r"band|window|block|mask|sink", m, re.I)]
    print(label, ms)
PY
