import os
from PIL import Image, ImageDraw, ImageFont
REVIEW = r"D:\REPOS\HYPERclone\artifixer-checkpoints\REVIEW"
os.makedirs(REVIEW, exist_ok=True)
cases = {
    "easy":  (r"D:\REPOS\HYPERclone\artifixer-checkpoints\laptop720_bidir_sample", ["00000","00120","00240"]),
    "wedge": (r"D:\REPOS\HYPERclone\artifixer-checkpoints\laptop720_wedge_sample", ["00135","00150","00165"]),
}
panels = [("rendered","base 3DGRUT"), ("artifixer3d","ArtiFixer3D"), ("gt","GROUND TRUTH")]
try: font = ImageFont.truetype("arial.ttf", 28)
except: font = ImageFont.load_default()
for case,(base,frames) in cases.items():
    for fr in frames:
        imgs=[]
        for sub,_ in panels:
            p=os.path.join(base,sub,fr+".png")
            imgs.append(Image.open(p).convert("RGB") if os.path.exists(p) else None)
        imgs=[i for i in imgs if i is not None]
        if not imgs: continue
        w,h=imgs[0].size; pad=8; lblh=40
        strip=Image.new("RGB",(w*len(imgs)+pad*(len(imgs)-1), h+lblh),(20,20,20))
        d=ImageDraw.Draw(strip)
        for i,(im,(sub,lbl)) in enumerate(zip(imgs,panels)):
            x=i*(w+pad); strip.paste(im,(x,lblh))
            d.text((x+10,6),lbl,fill=(255,255,255),font=font)
        out=os.path.join(REVIEW,f"{case}_{fr}_compare.png")
        strip.save(out); print("wrote",out)
print("done")
