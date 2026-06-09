import os
import csv
import shutil
from PIL import Image, ImageOps

ROOT = r"D:\PKU_CS\course_projects\camera\final"
SRC = os.path.join(ROOT, "other_data")
CLS = os.path.join(ROOT, "other_data_classified")
PREP = os.path.join(ROOT, "other_data_prep")
MANIFEST = os.path.join(ROOT, "other_data_manifest.csv")
MAX_LONG = 1536

CATS = {
    1: "01_indoor_museum_glass",
    2: "02_outdoor_street",
    3: "03_outdoor_water_landscape",
    4: "04_large_depth_aerial",
    5: "05_lowlight_night",
    6: "06_reflective_refractive",
    7: "07_forced_perspective_illusion",
    8: "08_synthetic_illusion",
}

MAP = {

    "web001.jpg": (6, "hard", "mirror+perspective", "室内房间，墙上两面镜子/窗，棋盘地板强透视"),
    "web002.jpg": (7, "hard", "forced_perspective", "玻利维亚盐湖，人站在水瓶上的强迫透视；含透明瓶"),
    "web003.jpg": (6, "hard", "refraction+dof", "城堡前石墙上的玻璃水晶球，折射+背景虚化"),
    "web004.jpg": (8, "hard", "synthetic_shadow_illusion", "Adelson 棋盘阴影错觉（合成渲染），本征/反照率相关"),
    "web005.jpg": (7, "hard", "forced_perspective", "手'捏'埃菲尔铁塔的强迫透视，HDR 天空"),
    "web006.jpg": (8, "hard", "moire+glass", "公园里玻璃后的莫尔条纹光栅板，混叠纹理"),
    "web007.jpg": (6, "hard", "convex_mirror", "树篱中的橙色凸面交通镜"),

    "IMG_0124.jpg": (1, "normal", "glass_case_lowtexture", "博物馆玻璃展柜内北大校徽，低纹理灰布"),
    "IMG_0591.jpg": (1, "normal", "glass_case", "博物馆展柜内金色盆景文物"),
    "IMG_0838.jpg": (1, "normal", "glass_case_reflection", "博物馆玻璃柜内玉琮，含玻璃反光"),
    "IMG_2358.jpg": (2, "normal", "street", "空旷十字路口，蓝天"),
    "IMG_2366.jpg": (2, "normal", "street", "带红绿灯的街道，行道树"),
    "IMG_2367.jpg": (2, "normal", "street_people_shadow", "校园道路，行人/自行车，强阴影"),
    "IMG_0138.jpg": (3, "normal", "water_dusk", "湖边成排绿船，垂柳，黄昏"),
    "IMG_1471.jpg": (3, "normal", "water_garden", "古典园林亭台+荷塘+远山，蓝天"),
    "IMG_2369.jpg": (3, "normal", "water_landmark", "未名湖+博雅塔+石舫"),
    "IMG_2370.jpg": (3, "hard", "water+forced_perspective", "同未名湖场景，前景手做强迫透视伸向博雅塔"),
    "IMG_0734.jpg": (4, "hard", "aerial_sky", "飞机舷窗外机翼与云层，超大纵深+天空"),
    "IMG_0810.jpg": (4, "normal", "cityscape_haze", "太平山俯瞰香港天际线，阴天有霾"),
    "IMG_1555.jpg": (5, "hard", "night_snow", "夜间雪后校园，行人，行道树"),
}

EXT = {".jpg", ".jpeg", ".png"}

def find_src(basename):
    for root, _, files in os.walk(SRC):
        if basename in files:
            return os.path.join(root, basename)
    return None

def main():
    os.makedirs(PREP, exist_ok=True)
    for c in CATS.values():
        os.makedirs(os.path.join(CLS, c), exist_ok=True)

    rows = []
    whdr_set = {"web001.jpg", "web002.jpg", "web003.jpg", "web004.jpg",
                "web005.jpg", "web006.jpg", "web007.jpg"}
    for fn, (cid, diff, phen, notes) in sorted(MAP.items(), key=lambda kv: (MAP[kv[0]][0], kv[0])):
        src = find_src(fn)
        if not src:
            print(f"MISSING {fn}")
            continue
        cat = CATS[cid]

        shutil.copy2(src, os.path.join(CLS, cat, fn))

        im = Image.open(src)
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        scale = MAX_LONG / max(w, h)
        if scale < 1.0:
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        base = os.path.splitext(fn)[0]
        out = os.path.join(PREP, f"{cid:02d}__{base}.png")
        im.save(out)
        rows.append({
            "file": fn, "category": cat, "difficulty": diff, "phenomenon": phen,
            "has_whdr_pairs": "yes" if fn in whdr_set else "no",
            "orig_w": w, "orig_h": h, "prep": os.path.basename(out), "notes": notes,
        })
        print(f"[{cat}] {fn} {w}x{h} diff={diff} -> {os.path.basename(out)}")

    with open(MANIFEST, "w", newline="", encoding="utf-8-sig") as f:
        wri = csv.DictWriter(f, fieldnames=[
            "file", "category", "difficulty", "phenomenon",
            "has_whdr_pairs", "orig_w", "orig_h", "prep", "notes"])
        wri.writeheader()
        wri.writerows(rows)

    from collections import Counter
    cc = Counter(r["category"] for r in rows)
    print("\n=== summary ===")
    for c in CATS.values():
        print(f"  {c}: {cc.get(c, 0)}")
    print(f"  total: {len(rows)}  (manifest: {MANIFEST})")

if __name__ == "__main__":
    main()
