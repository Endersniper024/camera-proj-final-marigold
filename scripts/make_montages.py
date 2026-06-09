import os, glob, csv
from PIL import Image, ImageDraw, ImageFont

ROOT = r"D:\PKU_CS\course_projects\camera\final"
PREP = os.path.join(ROOT, "other_data_prep")
OUT_BASE = os.path.join(ROOT, "results", "other_data", "other_data_out")
DEPTH = os.path.join(OUT_BASE, "depth", "depth_colored")
NORM = os.path.join(OUT_BASE, "normals", "normals_vis")
IID = os.path.join(OUT_BASE, "iid_appearance", "iid_appearance_vis")
MANIFEST = os.path.join(ROOT, "other_data_manifest.csv")
MONT = os.path.join(ROOT, "results", "other_data", "montages")
os.makedirs(MONT, exist_ok=True)

PANEL_H = 360
PAD = 8
LABEL_H = 22
BG = (255, 255, 255)

def font(sz=16):
    try:
        return ImageFont.truetype("arial.ttf", sz)
    except Exception:
        return ImageFont.load_default()

def load_meta():
    m = {}
    with open(MANIFEST, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            base = os.path.splitext(r["prep"])[0]
            m[base] = r
    return m

def fit(im, h=PANEL_H):
    w = round(im.width * h / im.height)
    return im.resize((w, h), Image.LANCZOS)

def labeled(im, text):
    im = fit(im)
    canvas = Image.new("RGB", (im.width, im.height + LABEL_H), BG)
    canvas.paste(im, (0, LABEL_H))
    d = ImageDraw.Draw(canvas)
    d.text((4, 3), text, fill=(0, 0, 0), font=font(15))
    return canvas

def montage_for(base):
    panels = []
    rgb = os.path.join(PREP, base + ".png")
    if os.path.exists(rgb):
        panels.append(labeled(Image.open(rgb).convert("RGB"), "RGB"))
    for path, lab in [
        (os.path.join(DEPTH, base + "_depth_colored.png"), "Depth"),
        (os.path.join(NORM, base + "_normals.png"), "Normals"),
        (os.path.join(IID, base + "_albedo.png"), "Albedo"),
        (os.path.join(IID, base + "_material.png"), "Material"),
    ]:
        if os.path.exists(path):
            panels.append(labeled(Image.open(path).convert("RGB"), lab))
    if not panels:
        return None
    W = sum(p.width for p in panels) + PAD * (len(panels) + 1)
    H = max(p.height for p in panels) + 2 * PAD
    canvas = Image.new("RGB", (W, H), BG)
    x = PAD
    for p in panels:
        canvas.paste(p, (x, PAD))
        x += p.width + PAD
    return canvas

def main():
    meta = load_meta()
    bases = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(PREP, "*.png")))
    cat_sheets = {}
    for base in bases:
        mont = montage_for(base)
        if mont is None:
            print("skip", base)
            continue
        r = meta.get(base, {})
        cat = r.get("category", "misc")
        diff = r.get("difficulty", "")

        titled = Image.new("RGB", (mont.width, mont.height + 26), BG)
        titled.paste(mont, (0, 26))
        d = ImageDraw.Draw(titled)
        d.text((6, 5), f"{base}   [{cat} / {diff}]   {r.get('phenomenon','')}", fill=(20, 20, 120), font=font(16))
        out = os.path.join(MONT, base + "_montage.png")
        titled.save(out)
        cat_sheets.setdefault(cat, []).append(titled)
        print("wrote", os.path.basename(out))

    sheetdir = os.path.join(MONT, "_by_category")
    os.makedirs(sheetdir, exist_ok=True)
    for cat, rows in cat_sheets.items():
        W = max(r.width for r in rows)
        H = sum(r.height for r in rows) + PAD * (len(rows) + 1)
        sheet = Image.new("RGB", (W, H), BG)
        y = PAD
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height + PAD
        sheet.save(os.path.join(sheetdir, cat + ".png"))
        print("category sheet:", cat, len(rows))
    print(f"\n=== montages in {MONT} ===")

if __name__ == "__main__":
    main()
