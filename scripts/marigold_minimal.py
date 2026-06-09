import argparse
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import matplotlib

SCALE = 0.18215

def load_components(ckpt, dtype, device):

    variant = "fp16" if dtype == torch.float16 else None
    vae = AutoencoderKL.from_pretrained(ckpt, subfolder="vae", torch_dtype=dtype, variant=variant)
    unet = UNet2DConditionModel.from_pretrained(ckpt, subfolder="unet", torch_dtype=dtype, variant=variant)
    text_encoder = CLIPTextModel.from_pretrained(ckpt, subfolder="text_encoder", torch_dtype=dtype, variant=variant)
    tokenizer = CLIPTokenizer.from_pretrained(ckpt, subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained(ckpt, subfolder="scheduler")
    vae.to(device).eval(); unet.to(device).eval(); text_encoder.to(device).eval()
    return vae, unet, text_encoder, tokenizer, scheduler

@torch.no_grad()
def empty_text_embedding(tokenizer, text_encoder, dtype, device):
    tok = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt")
    emb = text_encoder(tok.input_ids.to(device))[0].to(dtype)
    return emb

def preprocess(img: Image.Image, proc_res=768):
    img = img.convert("RGB")
    w, h = img.size
    scale = proc_res / max(w, h)
    nw, nh = round(w * scale), round(h * scale)
    nw -= nw % 8; nh -= nh % 8
    img_r = img.resize((nw, nh), Image.BILINEAR)
    arr = np.asarray(img_r).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)[None]
    return t * 2.0 - 1.0, (h, w)

@torch.no_grad()
def encode_rgb(vae, rgb, dtype):
    z = vae.encode(rgb.to(dtype)).latent_dist.mode()
    return z * SCALE

@torch.no_grad()
def decode_depth(vae, z_d):
    img = vae.decode(z_d / SCALE).sample
    depth = img.mean(dim=1, keepdim=True)
    depth = torch.clamp((depth + 1.0) / 2.0, 0.0, 1.0)
    return depth

@torch.no_grad()
def single_inference(vae, unet, scheduler, z_x, text_emb, steps, generator, device, dtype):
    scheduler.set_timesteps(steps, device=device)
    z_d = torch.randn(z_x.shape, generator=generator, device=device, dtype=dtype)
    z_d = z_d * scheduler.init_noise_sigma
    emb = text_emb.repeat(z_x.shape[0], 1, 1)
    for t in scheduler.timesteps:
        model_in = torch.cat([z_x, z_d], dim=1)
        noise_pred = unet(model_in, t, encoder_hidden_states=emb).sample
        z_d = scheduler.step(noise_pred, t, z_d, generator=generator).prev_sample
    return decode_depth(vae, z_d)

def align_scale_shift(pred, ref):
    p = pred.flatten().astype(np.float64)
    r = ref.flatten().astype(np.float64)
    A = np.stack([p, np.ones_like(p)], axis=1)
    sol, *_ = np.linalg.lstsq(A, r, rcond=None)
    a, b = sol
    return a * pred + b

def ensemble(preds):
    ref = preds[0]
    aligned = [preds[0]] + [align_scale_shift(p, ref) for p in preds[1:]]
    stacked = np.stack(aligned, axis=0)
    med = np.median(stacked, axis=0)
    med = (med - med.min()) / (med.max() - med.min() + 1e-8)
    return med

def colorize(depth, cmap="Spectral"):

    cm = matplotlib.colormaps[cmap]
    colored = cm(depth)[..., :3]
    return (colored * 255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--checkpoint", default="prs-eth/marigold-depth-v1-1")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--ensemble", type=int, default=10)
    ap.add_argument("--proc_res", type=int, default=768)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"loading {args.checkpoint} ...")
    vae, unet, text_encoder, tokenizer, scheduler = load_components(args.checkpoint, dtype, device)
    text_emb = empty_text_embedding(tokenizer, text_encoder, dtype, device)

    img = Image.open(args.input)
    rgb, (H0, W0) = preprocess(img, args.proc_res)
    rgb = rgb.to(device)
    z_x = encode_rgb(vae, rgb, dtype)

    preds = []
    for i in tqdm(range(args.ensemble), desc="ensemble"):
        g = torch.Generator(device=device).manual_seed(args.seed + i)
        d = single_inference(vae, unet, scheduler, z_x, text_emb, args.steps, g, device, dtype)
        preds.append(d.float().cpu().numpy()[0, 0])
    depth = ensemble(preds) if len(preds) > 1 else (
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))(preds[0])

    depth_img = Image.fromarray((depth * 65535).astype(np.uint16)).resize((W0, H0), Image.BILINEAR)
    depth = np.asarray(depth_img).astype(np.float32) / 65535.0

    base = os.path.splitext(os.path.basename(args.input))[0]
    np.save(os.path.join(args.output_dir, f"{base}_depth.npy"), depth)
    Image.fromarray(colorize(depth)).save(os.path.join(args.output_dir, f"{base}_depth_colored.png"))
    print(f"saved to {args.output_dir}")

if __name__ == "__main__":
    main()
