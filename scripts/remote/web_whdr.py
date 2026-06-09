import os, csv, glob, argparse
import numpy as np

def load_manifest_dims(manifest):
    dims = {}
    with open(manifest, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            dims[r["file"]] = (int(r["orig_w"]), int(r["orig_h"]))
    return dims

def find_depth_npy(npy_dir, base):

    hits = glob.glob(os.path.join(npy_dir, f"*__{base}_depth.npy"))
    return hits[0] if hits else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/root/autodl-tmp/other_data/web/depth_pairs.csv")
    ap.add_argument("--manifest", default="/root/autodl-tmp/other_data_manifest.csv")
    ap.add_argument("--npy_dir", default="/root/autodl-tmp/other_data_out/depth/depth_npy")
    ap.add_argument("--out_csv", default="/root/autodl-tmp/web_whdr_per_pair.csv")
    args = ap.parse_args()

    dims = load_manifest_dims(args.manifest)
    rows = []
    with open(args.pairs, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    depth_cache = {}
    out = []
    per_img = {}
    total = correct = 0
    for r in rows:
        img = r["image"]
        base = os.path.splitext(os.path.basename(img))[0]
        fn = os.path.basename(img)
        if base not in depth_cache:
            npy = find_depth_npy(args.npy_dir, base)
            depth_cache[base] = np.load(npy) if npy else None
        d = depth_cache[base]
        if d is None or fn not in dims:
            print(f"SKIP {img} (no depth or dims)")
            continue
        ow, oh = dims[fn]
        dh, dw = d.shape[:2]
        ax, ay, bx, by = int(r["ax"]), int(r["ay"]), int(r["bx"]), int(r["by"])
        def samp(x, y):
            xi = min(dw - 1, max(0, int(round(x * dw / ow))))
            yi = min(dh - 1, max(0, int(round(y * dh / oh))))
            return float(d[yi, xi])
        zA, zB = samp(ax, ay), samp(bx, by)
        closer = r["closer"].strip().upper()

        pred_closer = "A" if zA < zB else "B"
        ok = 1 if pred_closer == closer else 0
        total += 1
        correct += ok
        per_img.setdefault(base, [0, 0])
        per_img[base][0] += ok
        per_img[base][1] += 1
        out.append((img, r["pair_id"], closer, pred_closer, ok, round(zA, 4), round(zB, 4)))

    whdr = 1.0 - correct / max(1, total)
    print("=== per image ===")
    for base in sorted(per_img):
        c, n = per_img[base]
        print(f"  {base}: {c}/{n} correct, WHDR={1-c/n:.3f}")
    print(f"\n=== WEB WHDR (hard cases) ===")
    print(f"pairs={total} correct={correct} WHDR={whdr:.4f} ({whdr*100:.2f}%)")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "pair_id", "gt_closer", "pred_closer", "correct", "zA", "zB"])
        w.writerows(out)
    print(f"wrote {args.out_csv}")

if __name__ == "__main__":
    main()
