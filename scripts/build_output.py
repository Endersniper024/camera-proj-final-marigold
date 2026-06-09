import os, csv, json, shutil, glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

ROOT = r"D:\PKU_CS\course_projects\camera\final"
OUT = os.path.join(ROOT, "output")
MYDATA_NPY = os.path.join(ROOT, "results", "depth_mydata", "depth_npy")
MYDATA_COL = os.path.join(ROOT, "results", "depth_mydata", "depth_colored")
OD_NPY = os.path.join(ROOT, "_tmp_otherdepth", "depth_npy")
OD_COL = os.path.join(ROOT, "results", "other_data", "other_data_out", "depth", "depth_colored")
DIW_ANN = os.path.join(ROOT, "_tmp_diwann", "DIW_Annotations", "DIW_test.csv")
DIW_PER = os.path.join(ROOT, "results", "diw_eval", "diw_full_per_sample.csv")

CKPT = "prs-eth/marigold-depth-v1-1"
COMMON = dict(model="marigold_depth_v1_1", checkpoint=CKPT, denoise_steps=1,
              ensemble_size=10, processing_res=0, device="cuda",
              cuda_device_name="NVIDIA GeForce RTX 5090 D", npy_dtype="float32",
              depth_convention="affine_invariant_[0,1]_larger_is_farther",
              visualization="Spectral (native Marigold)")

def recolor(npy_path, out_png):
    d = np.load(npy_path).astype(np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    n = np.clip((d - lo) / max(1e-6, hi - lo), 0, 1)
    near_bright = 1.0 - n
    rgb = (cm.get_cmap("Spectral_r")(near_bright)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(out_png)

def build_set(name, items, npy_dir, col_dir, npy_suffix="_depth", col_suffix="_depth_colored"):
    dst = os.path.join(OUT, name)
    rec = os.path.join(dst, "recolor")
    os.makedirs(rec, exist_ok=True)
    rows = []
    for i, (src, dstb) in enumerate(sorted(items), 1):
        npy_src = os.path.join(npy_dir, src + npy_suffix + ".npy")
        col_src = os.path.join(col_dir, src + col_suffix + ".png")
        if not os.path.exists(npy_src):
            print("  MISSING npy", npy_src); continue
        npy_dst = os.path.join(dst, dstb + ".npy")
        png_dst = os.path.join(dst, dstb + ".png")
        shutil.copy2(npy_src, npy_dst)
        if os.path.exists(col_src):
            shutil.copy2(col_src, png_dst)
        recolor(npy_dst, os.path.join(rec, dstb + ".png"))
        d = np.load(npy_dst)
        h, w = d.shape[:2]
        rows.append(dict(index=i, image=dstb, width=w, height=h,
                         depth_min=round(float(d.min()), 6), depth_max=round(float(d.max()), 6),
                         checkpoint=CKPT, denoise_steps=1, ensemble_size=10, processing_res="native",
                         npy_dtype="float32",
                         output_png=os.path.join(name, dstb + ".png").replace("\\", "/"),
                         output_npy=os.path.join(name, dstb + ".npy").replace("\\", "/")))

    summ = dict(image_count=len(rows), **COMMON,
                note="per-image runtime not logged (batch run); see report for speed (5090 ~0.12-0.23 s/img).")
    with open(os.path.join(dst, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)

    cols = ["index", "image", "width", "height", "depth_min", "depth_max", "checkpoint",
            "denoise_steps", "ensemble_size", "processing_res", "npy_dtype", "elapsed_ms",
            "output_png", "output_npy"]
    with open(os.path.join(dst, "runtime.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r = dict(r); r["elapsed_ms"] = ""
            w.writerow(r)

    with open(os.path.join(rec, "recolor_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(source_dir=name, style="marigold-recolor", colormap="Spectral_r",
                       visualization="near_bright_far_dark", normalize="per-image 2-98 pct",
                       written=len(rows)), f, indent=2)
    print(f"[{name}] {len(rows)} images")

def parse_diw_ann(path):
    with open(path, errors="replace") as f:
        lines = [l.rstrip("\n").rstrip("\r") for l in f]
    ann = {}
    i = 0
    while i < len(lines) - 1:
        p = lines[i].strip()
        if not p:
            i += 1; continue
        flds = [x for x in lines[i + 1].replace("\t", ",").split(",") if x.strip() != ""]
        if len(flds) >= 7:
            ann[p] = dict(yA=int(float(flds[0])), xA=int(float(flds[1])),
                          yB=int(float(flds[2])), xB=int(float(flds[3])),
                          rel=flds[4][0])
        i += 2
    return ann

def build_diw():
    dst = os.path.join(OUT, "diw", "eval_full")
    os.makedirs(dst, exist_ok=True)
    ann = parse_diw_ann(DIW_ANN)
    rows = []
    correct = total = 0
    with open(DIW_PER) as f:
        for k, r in enumerate(csv.DictReader(f), 1):
            path = r["path"]
            a = ann.get(path) or ann.get(path.lstrip("./")) or ann.get("./" + path)
            if a is None:
                continue
            zA, zB = float(r["zA"]), float(r["zB"])
            rel = a["rel"]
            closer = "B" if rel == ">" else "A"
            prediction = "A" if zA < zB else "B"
            ok = int(r["correct"])
            correct += ok; total += 1
            denom = abs(zA) + abs(zB) + 1e-9
            rows.append(dict(model="marigold_depth_v1_1",
                             image=path.lstrip("./"),
                             pair_id=f"diw_{k:06d}",
                             ax=a["xA"] - 1, ay=a["yA"] - 1, bx=a["xB"] - 1, by=a["yB"] - 1,
                             closer=closer, selected_direction="larger_farther",
                             prediction=prediction, correct=ok,
                             depth_a=f"{zA:.8f}", depth_b=f"{zB:.8f}",
                             relative_margin=f"{abs(zA - zB) / denom:.8f}",
                             pred_larger_closer=("A" if zA > zB else "B"),
                             pred_larger_farther=("A" if zA < zB else "B"),
                             depth_file="(not saved per DIW image)",
                             notes=f"diw_relation={rel}"))
    whdr = 1 - correct / max(1, total)
    pcols = ["model", "image", "pair_id", "ax", "ay", "bx", "by", "closer", "selected_direction",
             "prediction", "correct", "depth_a", "depth_b", "relative_margin",
             "pred_larger_closer", "pred_larger_farther", "depth_file", "notes"]
    with open(os.path.join(dst, "pair_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pcols); w.writeheader(); w.writerows(rows)
    model_row = dict(model="marigold_depth_v1_1", outdir="output/diw/eval_full",
                     selected_direction="larger_farther", total_pairs=total, correct=correct,
                     ties=0, accuracy=f"{correct/total:.4f}", whdr=f"{whdr:.4f}",
                     non_tie_accuracy=f"{correct/total:.4f}", mean_relative_margin="",
                     runtime_image_count=73983, runtime_mean_ms=123.0,
                     runtime_median_ms="", runtime_p95_ms="", runtime_min_ms="", runtime_max_ms="")
    with open(os.path.join(dst, "model_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(model_row.keys())); w.writeheader(); w.writerow(model_row)
    summ = dict(annotation="DIW_Annotations/DIW_test.csv",
                images_root="diw/DIW_test", sample_radius=0, tie_threshold=0.0,
                models=[dict(model="marigold_depth_v1_1", outdir="output/diw/eval_full",
                             selected_direction="larger_farther", total_pairs=total, correct=correct,
                             ties=0, accuracy=f"{correct/total:.4f}", whdr=f"{whdr:.4f}",
                             non_tie_accuracy=f"{correct/total:.4f}",
                             denoise_steps=1, ensemble_size=1, processing_res=768,
                             runtime_image_count=73983, runtime_mean_ms=123.0)])
    with open(os.path.join(dst, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)
    print(f"[diw/eval_full] pairs={total} correct={correct} WHDR={whdr:.4f}")

def main():
    os.makedirs(OUT, exist_ok=True)

    other = [(os.path.splitext(os.path.basename(p))[0].replace("_depth", ""),
              os.path.splitext(os.path.basename(p))[0].replace("_depth", ""))
             for p in glob.glob(os.path.join(MYDATA_NPY, "*_depth.npy"))]
    build_set("marigold-other", other, MYDATA_NPY, MYDATA_COL)

    od = [os.path.splitext(os.path.basename(p))[0].replace("_depth", "") for p in glob.glob(os.path.join(OD_NPY, "*_depth.npy"))]
    phone = [(b, b.split("__", 1)[1]) for b in od if "IMG_" in b]
    web = [(b, b.split("__", 1)[1]) for b in od if "web" in b.lower()]
    build_set("marigold-phone_photo", phone, OD_NPY, OD_COL)
    build_set("marigold-web", web, OD_NPY, OD_COL)

    build_diw()
    print("\n=== output/ built ===")

if __name__ == "__main__":
    main()
