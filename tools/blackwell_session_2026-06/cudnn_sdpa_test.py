import math, torch, cudnn
dev="cuda"; dt=torch.bfloat16
b,h,s,d = 1,8,512,128            # head_dim=128 like Wan
q=torch.randn(b,h,s,d,device=dev,dtype=dt)
k=torch.randn(b,h,s,d,device=dev,dtype=dt)
v=torch.randn(b,h,s,d,device=dev,dtype=dt)
scale=1.0/math.sqrt(d)

def run_cudnn(causal=True, left_bound=None):
    g=cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16,
                    intermediate_data_type=cudnn.data_type.FLOAT,
                    compute_data_type=cudnn.data_type.FLOAT)
    qc=g.tensor_like(q); kc=g.tensor_like(k); vc=g.tensor_like(v)
    kw=dict(name="sdpa", q=qc, k=kc, v=vc, is_inference=True, attn_scale=scale)
    if causal: kw["diagonal_alignment"]=cudnn.diagonal_alignment.BOTTOM_RIGHT; kw["diagonal_band_right_bound"]=0
    if left_bound is not None: kw["diagonal_band_left_bound"]=left_bound
    o,_=g.sdpa(**kw)
    o.set_output(True).set_dim([b,h,s,d]).set_stride([h*s*d, s*d, d, 1])
    g.validate(); g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A]); g.check_support(); g.build_plans()
    ws=torch.empty(g.get_workspace_size(), device=dev, dtype=torch.uint8)
    ot=torch.empty(b,h,s,d, device=dev, dtype=dt)
    g.execute({qc:q, kc:k, vc:v, o:ot}, ws)
    torch.cuda.synchronize(); return ot

# 1) full causal vs torch reference
oc=run_cudnn(causal=True)
ref=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
print("CAUSAL  max|diff| =", (oc.float()-ref.float()).abs().max().item(), " finite", bool(torch.isfinite(oc).all()))

# 2) sliding-window (window=64) vs a reference computed with an explicit band mask
W=64
ow=run_cudnn(causal=True, left_bound=W)
qi=torch.arange(s,device=dev)[:,None]; ki=torch.arange(s,device=dev)[None,:]
band=(ki<=qi) & (ki>qi-W)          # causal AND within last W (BOTTOM_RIGHT, sq==skv)
amask=torch.where(band, 0.0, float("-inf")).to(dt)
refw=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=amask)
print("WINDOW  max|diff| =", (ow.float()-refw.float()).abs().max().item(), " finite", bool(torch.isfinite(ow).all()))
print("CUDNN_SDPA_SM120_OK")
