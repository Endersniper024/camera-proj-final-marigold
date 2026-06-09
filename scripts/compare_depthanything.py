import os, glob, time
import numpy as np
import torch
import matplotlib
from PIL import Image
from transformers import pipeline

SRC = r"D:\PKU_CS\course_projects\camera\final\my_data_prep"
DST = r"D:\PKU_CS\course_projects\camera\final\results\depthanything_mydata"
os.makedirs(DST, exist_ok=True)

device = 0 if torch.cuda.is_available() else -1
model = "depth-anything/Depth-Anything-V2-Base-hf"
print(f"loading {model} ...")
pipe = pipeline("depth-estimation", model=model, device=device)

cm = matplotlib.colormaps["Spectral"]
files = sorted(glob.glob(os.path.join(SRC, "*.png")))
times = []
for f in files:
    im = Image.open(f).convert("RGB")
    t0 = time.time()
    out = pipe(im)
    times.append(time.time() - t0)
    pred = out["predicted_depth"].squeeze().detach().cpu().numpy().astype(np.float32)

    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    near_small = 1.0 - pred
    colored = (cm(near_small)[..., :3] * 255).astype(np.uint8)
    base = os.path.splitext(os.path.basename(f))[0]
    Image.fromarray(colored).resize(im.size, Image.BILINEAR).save(
        os.path.join(DST, f"{base}_dav2.png"))
    print(f"{base}  {times[-1]*1000:.0f} ms")

print(f"=== DAv2 done: {len(files)} imgs, avg {sum(times)/len(times)*1000:.0f} ms/img ===")
