import os
from PIL import Image, ImageOps

SRC = r"D:\PKU_CS\course_projects\camera\final\my_data"
DST = r"D:\PKU_CS\course_projects\camera\final\my_data_prep"
MAX_LONG = 1536

os.makedirs(DST, exist_ok=True)
EXT = {".jpg", ".jpeg", ".png"}
count = 0
for root, _, files in os.walk(SRC):
    cat = os.path.basename(root)
    if cat == os.path.basename(SRC):
        cat = "misc"

    tag = cat.split("_")[0] if cat[0:1].isdigit() else cat
    for fn in sorted(files):
        if os.path.splitext(fn)[1].lower() not in EXT:
            continue
        src = os.path.join(root, fn)
        try:
            im = Image.open(src)
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            w, h = im.size
            scale = MAX_LONG / max(w, h)
            if scale < 1.0:
                im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
            base = os.path.splitext(fn)[0]
            out = os.path.join(DST, f"{tag}__{base}.png")
            im.save(out)
            count += 1
            print(f"{cat}/{fn} {w}x{h} -> {im.size}  {os.path.basename(out)}")
        except Exception as e:
            print(f"FAIL {src}: {e}")
print(f"=== prepared {count} images into {DST} ===")
