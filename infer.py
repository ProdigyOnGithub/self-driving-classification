"""Run a trained decider on images and measure speed. Works on laptop, Orin (cuda) and RPi (cpu).

    HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 python infer.py --run runs/lora384 data/STOP
    HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 python infer.py --run runs/lora384 data/MOVE --device cpu --limit 50

Prints MOVE/STOP per image plus latency (encoder, SLM, total). The prompt prefix is computed once and
cached, and LoRA is merged into the weights, so each frame only runs the image + short question.
"""
import argparse
import os
import statistics
import time

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

import vlm

IMG_EXT = (".jpg", ".jpeg", ".png")


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("paths", nargs="+", help="images or folders")
    a.add_argument("--run", required=True, help="folder with best.pt")
    a.add_argument("--device", default=None)
    a.add_argument("--fp16", action="store_true", help="half precision (cuda/mps)")
    a.add_argument("--threshold", type=float, default=None, help="override the saved safety threshold")
    a.add_argument("--limit", type=int, default=0)
    args = a.parse_args()

    device = vlm.pick_device(args.device)
    dtype = torch.float16 if args.fp16 and device != "cpu" else torch.float32
    ck = torch.load(os.path.join(args.run, "best.pt"), map_location="cpu")
    cfg = ck["args"]
    thr = ck["threshold"] if args.threshold is None else args.threshold

    proc = AutoImageProcessor.from_pretrained(cfg["encoder"])
    enc = SiglipVisionModel.from_pretrained(cfg["encoder"], dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(cfg["slm"])
    slm = AutoModelForCausalLM.from_pretrained(cfg["slm"], dtype=torch.float32)
    if cfg["mode"] == "lora":
        vlm.add_lora(slm, cfg["lora_r"], cfg["lora_alpha"])
    slm.load_state_dict(ck["slm_trainable"], strict=False)
    vlm.merge_lora(slm)
    projector = vlm.Projector(ck["vis_dim"], slm.config.hidden_size)
    projector.load_state_dict(ck["projector"])
    slm.to(device, dtype).eval()
    model = vlm.Decider(slm, tok, projector.to(device, dtype)).eval()
    cache, plen = model.prefix_cache()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.lower().endswith(IMG_EXT)]
        else:
            files.append(p)
    files = files[:args.limit] if args.limit else files
    print(f"device={device} dtype={dtype} threshold={thr:+.3f} images={len(files)}  (run {args.run}, "
          f"epoch {ck['epoch']}, {cfg['encoder'].split('/')[-1]}, mode={cfg['mode']})")

    t_enc, t_slm, counts = [], [], {"MOVE": 0, "STOP": 0}
    for i, f in enumerate(files):
        img = Image.open(f).convert("RGB")
        sync(device); t0 = time.perf_counter()
        feats = vlm.encode_images(enc, proc, [img], cfg["layer"], cfg["pool"])
        sync(device); t1 = time.perf_counter()
        score = vlm.stop_score(model.forward_cached(feats.to(dtype), cache, plen)).item()
        sync(device); t2 = time.perf_counter()
        decision = "STOP" if score > thr else "MOVE"
        counts[decision] += 1
        if i >= 2:                                               # skip warm-up frames in timing
            t_enc.append((t1 - t0) * 1e3); t_slm.append((t2 - t1) * 1e3)
        print(f"{decision}  score={score:+.3f}  {os.path.basename(f)}")

    if t_enc:
        tot = sorted(e + s for e, s in zip(t_enc, t_slm))
        p95 = tot[min(len(tot) - 1, int(0.95 * len(tot)))]
        print(f"\n{counts}  |  latency ms: encoder {statistics.median(t_enc):.1f}, SLM {statistics.median(t_slm):.1f}, "
              f"total median {statistics.median(tot):.1f} / p95 {p95:.1f}  ->  {1000 / statistics.median(tot):.1f} FPS")


if __name__ == "__main__":
    main()
