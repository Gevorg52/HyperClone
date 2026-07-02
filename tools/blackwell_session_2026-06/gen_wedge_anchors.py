import pycolmap
rec = pycolmap.Reconstruction(r"D:\REPOS\HYPERclone\apex-sfm\_dataset\Laptop\train\data\sparse\0")
names = sorted(img.name for img in rec.images.values())
N = len(names)
W0, W1 = 130, 170          # contiguous held-out wedge: sorted-idx [130,170) = 40 views
anchors = [n for i, n in enumerate(names) if not (W0 <= i < W1)]
wedge   = [n for i, n in enumerate(names) if W0 <= i < W1]
out = r"D:\REPOS\HYPERclone\artifixer-checkpoints\laptop_wedge_anchors.txt"
open(out, "w", newline="\n").write("\n".join(anchors) + "\n")
print(f"{N} registered -> {len(anchors)} anchors, {len(wedge)} held-out WEDGE (contiguous)")
print("wedge first/last:", wedge[0], "..", wedge[-1])
print("wedge sorted-idx range:", W0, "..", W1-1)
print("wrote", out)
