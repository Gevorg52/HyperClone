import pycolmap
rec = pycolmap.Reconstruction(r"D:\REPOS\HYPERclone\apex-sfm\_dataset\Laptop\train\data\sparse\0")
names = sorted(img.name for img in rec.images.values())
train = [n for i, n in enumerate(names) if i % 8 != 0]   # held-out = idx%8==0 (matches trainer)
out = r"D:\REPOS\HYPERclone\artifixer-checkpoints\laptop_train.txt"
open(out, "w", newline="\n").write("\n".join(train) + "\n")
print(f"{len(names)} registered -> {len(train)} anchors, {len(names)-len(train)} held-out")
print("held-out sample:", [n for i, n in enumerate(names) if i % 8 == 0][:4])
print("wrote", out)
