import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

for cand in ("/root/autodl-tmp/Marigold", os.path.join(os.path.dirname(__file__), "Marigold")):
    if os.path.isdir(os.path.join(cand, "marigold")):
        sys.path.insert(0, cand)
        break

import argparse, csv, random, time
import numpy as np
from PIL import Image
import torch
from marigold import MarigoldDepthPipeline

def parse_diw(annot_path):
    with open(annot_path, "r", errors="replace") as f:
        lines = [ln.rstrip("\n").rstrip("\r") for ln in f]

    items = []
    i = 0
    n = len(lines)
    while i < n:
        path = lines[i].strip()
        if path == "":
            i += 1
            continue

        j = i + 1
        while j < n and lines[j].strip() == "":
            j += 1
        if j >= n:
            break
        fields = lines[j].replace("\t", ",").split(",")
        fields = [x for x in (s.strip() for s in fields) if x != ""]
        if len(fields) < 7 or ("/" not in path and "\\" not in path and not path.lower().endswith((".jpg", ".png", ".jpeg", ".thumb"))):

            i += 1
            continue
        try:
            yA, xA, yB, xB = int(float(fields[0])), int(float(fields[1])), int(float(fields[2])), int(float(fields[3]))
            rel = 1 if fields[4][0] == ">" else (-1 if fields[4][0] == "<" else 0)
            W, H = int(float(fields[5])), int(float(fields[6]))
        except Exception:
            i += 1
            continue
        if rel != 0:
            items.append(dict(path=path, yA=yA, xA=xA, yB=yB, xB=xB, rel=rel, W=W, H=H))
        i = j + 1
    return items

def resolve_image(diw_root, rel_path):
    cands = [os.path.join(diw_root, rel_path),
             os.path.join(diw_root, rel_path.lstrip("./")),
             os.path.join(diw_root, os.path.basename(rel_path))]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diw_root", required=True)
    ap.add_argument("--annot", required=True)
    ap.add_argument("--checkpoint", default="prs-eth/marigold-depth-v1-1")
    ap.add_argument("--denoise_steps", type=int, default=1)
    ap.add_argument("--ensemble_size", type=int, default=1)
    ap.add_argument("--processing_res", type=int, default=768)
    ap.add_argument("--out_csv", default="diw_per_sample.csv")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--shuffle_seed", type=int, default=-1, help=">=0 to shuffle before limiting")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()

    items = parse_diw(args.annot)
    print(f"parsed {len(items)} annotated pairs from {args.annot}", flush=True)
    if args.shuffle_seed >= 0:
        random.Random(args.shuffle_seed).shuffle(items)
    if args.limit and args.limit < len(items):
        items = items[: args.limit]
        print(f"limited to {len(items)} pairs", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = MarigoldDepthPipeline.from_pretrained(args.checkpoint, torch_dtype=torch.float16, variant="fp16")
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe = pipe.to(device)
    print(f"loaded {args.checkpoint} on {device}; scale_inv={pipe.scale_invariant} shift_inv={pipe.shift_invariant}", flush=True)

    rows = []
    correct = 0
    evaluated = 0
    missing = 0
    t0 = time.time()
    with torch.no_grad():
        for k, it in enumerate(items):
            img_path = resolve_image(args.diw_root, it["path"])
            if img_path is None:
                missing += 1
                continue
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                missing += 1
                continue
            gen = torch.Generator(device=device).manual_seed(args.seed)
            out = pipe(img, denoising_steps=args.denoise_steps, ensemble_size=args.ensemble_size,
                       processing_res=args.processing_res, match_input_res=True,
                       batch_size=0, show_progress_bar=False, generator=gen)
            d = out.depth_np
            Hp, Wp = d.shape[:2]
            W = it["W"] if it["W"] > 1 else img.width
            H = it["H"] if it["H"] > 1 else img.height
            def idx(x, y):
                xi = min(Wp - 1, max(0, int(round((x - 1) * (Wp - 1) / max(1, W - 1)))))
                yi = min(Hp - 1, max(0, int(round((y - 1) * (Hp - 1) / max(1, H - 1)))))
                return yi, xi
            yA, xA = idx(it["xA"], it["yA"])
            yB, xB = idx(it["xB"], it["yB"])
            zA, zB = float(d[yA, xA]), float(d[yB, xB])
            classify = 1 if zA > zB else (-1 if zA < zB else 0)
            ok = 1 if (classify * it["rel"] > 0) else 0
            correct += ok
            evaluated += 1
            rows.append((it["path"], ok, zA, zB, it["rel"]))
            if evaluated % args.log_every == 0:
                whdr = 1.0 - correct / evaluated
                dt = time.time() - t0
                print(f"[{evaluated}/{len(items)}] WHDR={whdr:.4f} "
                      f"({dt/evaluated:.3f}s/img, missing={missing})", flush=True)

    whdr = 1.0 - correct / max(1, evaluated)
    print(f"\n=== DIW RESULT ===", flush=True)
    print(f"checkpoint={args.checkpoint} steps={args.denoise_steps} ens={args.ensemble_size} res={args.processing_res}", flush=True)
    print(f"evaluated={evaluated} missing={missing} correct={correct}", flush=True)
    print(f"WHDR={whdr:.4f}  ({whdr*100:.2f}%)", flush=True)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "correct", "zA", "zB", "rel"])
        w.writerows(rows)

    with open(args.out_csv + ".summary.txt", "w") as f:
        f.write(f"checkpoint={args.checkpoint} steps={args.denoise_steps} ens={args.ensemble_size} res={args.processing_res}\n")
        f.write(f"evaluated={evaluated} missing={missing} correct={correct}\n")
        f.write(f"WHDR={whdr:.6f} ({whdr*100:.2f}%)\n")
    print(f"wrote {args.out_csv} and summary", flush=True)

if __name__ == "__main__":
    main()
