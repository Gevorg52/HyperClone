import os, sys, shutil
f = os.path.expanduser("~/artifixer/model_training/net/transformer.py")
src = open(f).read()
old = (
'        except Exception as e:\n'
'            print(f"WARNING: FA4 activation failed on sm_{major}0: {e}")\n'
'\n'
'        # Use FA4\'s CuTeDSL kernels for flex_attention (requires flash_attn.cute).\n'
'        _FLEX_KERNEL_OPTIONS["BACKEND"] = "FLASH"\n'
'        print("flex_attention will use FA4 CuTeDSL backend")\n'
)
new = (
'            # Use FA4\'s CuTeDSL kernels for flex_attention (requires flash_attn.cute).\n'
'            # PATCH (apex/sm_120): only enable FLASH flex backend if FA4 actually activated;\n'
'            # otherwise stay on Triton flex (verified working on sm_120). Original set this\n'
'            # unconditionally, breaking flex_attention when flash_attn is absent.\n'
'            _FLEX_KERNEL_OPTIONS["BACKEND"] = "FLASH"\n'
'            print("flex_attention will use FA4 CuTeDSL backend")\n'
'        except Exception as e:\n'
'            print(f"WARNING: FA4 activation failed on sm_{major}0: {e}; flex_attention stays on Triton")\n'
)
if new.split("PATCH")[0] in src and old not in src:
    print("ALREADY_PATCHED"); sys.exit(0)
if old not in src:
    print("PATCH_TARGET_NOT_FOUND"); sys.exit(2)
shutil.copy2(f, f + ".orig")
open(f, "w").write(src.replace(old, new, 1))
print("PATCHED OK -> backup at", f + ".orig")
