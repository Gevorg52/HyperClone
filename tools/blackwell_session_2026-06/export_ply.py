import torch, numpy as np, sys
from plyfile import PlyData, PlyElement
ck_path, out_path = sys.argv[1], sys.argv[2]
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
def g(k): return ck[k].detach().float().cpu().numpy()
pos   = g("positions")
N     = pos.shape[0]
nrm   = np.zeros((N,3), np.float32); nrm[:,2]=1.0
dc    = g("features_albedo")
nspec = (int(ck["max_n_features"])+1)**2 - 1            # 15
spec  = g("features_specular").reshape(N,nspec,3).transpose(0,2,1).reshape(N,nspec*3)  # channel-major (3DGS PLY order)
opac  = g("density")
scale = g("scale")
rot   = g("rotation")
cols  = ["x","y","z","nx","ny","nz"]+[f"f_dc_{i}" for i in range(3)]+[f"f_rest_{i}" for i in range(nspec*3)]+["opacity"]+[f"scale_{i}" for i in range(3)]+[f"rot_{i}" for i in range(4)]
attrs = np.concatenate([pos,nrm,dc,spec,opac,scale,rot],axis=1).astype(np.float32)
el = np.empty(N, dtype=[(c,"f4") for c in cols])
for i,c in enumerate(cols): el[c] = attrs[:,i]
PlyData([PlyElement.describe(el,"vertex")], text=False).write(out_path)
print("wrote", out_path, "|", N, "gaussians,", len(cols), "attrs")
