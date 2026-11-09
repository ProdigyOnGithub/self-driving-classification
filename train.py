"""Train the MOVE/STOP decider: frozen SigLIP -> projector (+ LoRA in Qwen) -> Qwen answers MOVE or STOP.

Run on the GPU server (offline):
    HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 python train.py --name lora384

Quick local smoke test:
    HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 python train.py --name smoke --limit 40 --epochs 1 --n_aug 1
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

import vlm

IMG_EXT = (".jpg", ".jpeg", ".png")


def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("--name", default=time.strftime("run_%m%d_%H%M"))
    a.add_argument("--train_dir", default="data_bddoia", help="public data with MOVE/ and STOP/ folders")
    a.add_argument("--test_dir", default="data", help="BITS campus photos with MOVE/ and STOP/ (test only)")
    a.add_argument("--encoder", default="google/siglip-base-patch16-384")
    a.add_argument("--layer", type=int, default=-2, help="encoder layer to read (-1 last, -2 second to last)")
    a.add_argument("--pool", type=int, default=2, help="average-pool token grid (1 = off)")
    a.add_argument("--slm", default="Qwen/Qwen2.5-0.5B-Instruct")
    a.add_argument("--mode", choices=["frozen", "lora", "full"], default="lora",
                   help="frozen = projector only (old setup); lora = projector + LoRA; full = projector + whole SLM")
    a.add_argument("--lora_r", type=int, default=16)
    a.add_argument("--lora_alpha", type=int, default=32)
    a.add_argument("--epochs", type=int, default=8)
    a.add_argument("--batch", type=int, default=32)
    a.add_argument("--lr", type=float, default=3e-4, help="projector + LoRA learning rate")
    a.add_argument("--slm_lr", type=float, default=2e-5, help="SLM learning rate in --mode full")
    a.add_argument("--wd", type=float, default=0.01)
    a.add_argument("--warmup", type=float, default=0.05)
    a.add_argument("--stop_weight", type=float, default=1.5, help="loss weight for STOP (missing a STOP is worse)")
    a.add_argument("--label_smoothing", type=float, default=0.05, help="BDD labels are noisy")
    a.add_argument("--n_aug", type=int, default=2, help="extra augmented views cached per training image")
    a.add_argument("--target_recall", type=float, default=0.95, help="STOP recall the threshold is tuned for")
    a.add_argument("--limit", type=int, default=0, help="cap images per class (smoke tests)")
    a.add_argument("--cache_dir", default="cache")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--device", default=None)
    return a.parse_args()


# ---------------------------------------------------------------- data
def list_images(root):
    items, skipped = [], 0
    if not os.path.isdir(root):
        return items
    subdirs = {d.upper(): d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}
    for label, name in enumerate(vlm.LABELS):
        if name not in subdirs:
            continue
        folder = os.path.join(root, subdirs[name])
        for f in sorted(os.listdir(folder)):
            p = os.path.join(folder, f)
            if not f.lower().endswith(IMG_EXT):
                skipped += f.lower().endswith((".heic", ".heif"))
                continue
            try:
                Image.open(p).size
            except Exception:
                print("  unreadable, skipping", p)
                continue
            items.append((p, label))
    if skipped:
        print(f"  WARNING: {skipped} HEIC files in {root} ignored - convert them to JPG first")
    return items


def augment(img, rng):
    """Camera / weather degradations seen on campus: rain, blur, low light, JPEG, framing."""
    w, h = img.size
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    s = rng.uniform(0.85, 1.0)                                   # mild crop, keeps the path ahead
    cw, ch = int(w * s), int(h * s)
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    img = img.crop((x0, y0, x0 + cw, y0 + ch))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 1.3))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.35:                                      # rain streaks + haze
        w, h = img.size
        streaks = Image.new("L", img.size, 0)
        d = ImageDraw.Draw(streaks)
        slant = rng.uniform(-0.4, 0.4)
        for _ in range(rng.randint(150, 600)):
            x, y, ln = rng.randint(0, w), rng.randint(0, h), rng.randint(8, 30)
            d.line([(x, y), (x + slant * ln, y + ln)], fill=rng.randint(120, 220), width=1)
        streaks = streaks.filter(ImageFilter.GaussianBlur(0.8))
        img = Image.composite(Image.new("RGB", img.size, (200, 200, 205)), img, streaks.point(lambda v: v * 0.45))
        img = ImageEnhance.Contrast(img).enhance(0.8)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0)))
    if rng.random() < 0.3:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(15, 60))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


def load_image(path, view, seed):
    img = Image.open(path).convert("RGB")
    img.thumbnail((800, 800))                                    # speed; the encoder sees 384px anyway
    if view > 0:
        img = augment(img, random.Random(zlib.crc32(f"{path}|{view}|{seed}".encode())))
    return img


def extract(items, view, tag, args, enc, proc):
    """Encoder features for one view of a list of images, cached on disk keyed by content of the settings."""
    key = hashlib.sha1(json.dumps([args.encoder, args.layer, args.pool, view, args.seed,
                                   [p for p, _ in items]]).encode()).hexdigest()[:12]
    path = os.path.join(args.cache_dir, f"{tag}_v{view}_{key}.pt")
    if os.path.exists(path):
        return torch.load(path)
    os.makedirs(args.cache_dir, exist_ok=True)
    feats, bs = [], 64
    with ThreadPoolExecutor(8) as pool:
        for i in range(0, len(items), bs):
            imgs = list(pool.map(lambda it: load_image(it[0], view, args.seed), items[i:i + bs]))
            feats.append(vlm.encode_images(enc, proc, imgs, args.layer, args.pool).half().cpu())
            print(f"  [{tag} view {view}] encoded {min(i + bs, len(items))}/{len(items)}", end="\r")
    print()
    out = torch.cat(feats)
    torch.save(out, path)
    return out


# ---------------------------------------------------------------- metrics
def auroc(s, y):
    sp, sn = s[y == 1], s[y == 0]
    if len(sp) == 0 or len(sn) == 0:
        return float("nan")
    gt = (sp[:, None] > sn[None, :]).float().mean()
    eq = (sp[:, None] == sn[None, :]).float().mean()
    return (gt + 0.5 * eq).item()


def threshold_for_recall(s, y, target):
    sp = torch.sort(s[y == 1]).values
    if len(sp) == 0:
        return 0.0
    k = min(int(math.floor((1 - target) * len(sp))), len(sp) - 1)
    return sp[k].item() - 1e-6                                   # STOP if score > thr


def metrics(s, y, thr):
    pred = (s > thr).long()
    n_stop, n_move = int((y == 1).sum()), int((y == 0).sum())
    recall = pred[y == 1].float().mean().item() if n_stop else float("nan")
    false_stop = pred[y == 0].float().mean().item() if n_move else float("nan")
    return {"n": len(y), "n_STOP": n_stop, "n_MOVE": n_move, "threshold": round(thr, 4),
            "acc": round((pred == y).float().mean().item(), 4), "auroc": round(auroc(s, y), 4),
            "STOP_recall": round(recall, 4), "false_MOVE": round(1 - recall, 4) if n_stop else float("nan"),
            "false_STOP": round(false_stop, 4)}


def show(name, m):
    warn = "  <-- too few STOP images, numbers unreliable" if m["n_STOP"] < 30 else ""
    print(f"[{name:9s}] n={m['n']:4d} (STOP {m['n_STOP']}) thr={m['threshold']:+.3f} acc={m['acc']:.3f} "
          f"auroc={m['auroc']:.3f} STOP-recall={m['STOP_recall']:.3f} false-STOP={m['false_STOP']:.3f}{warn}")


# ---------------------------------------------------------------- main
def main():
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = vlm.pick_device(args.device)
    amp = device == "cuda" and torch.cuda.is_bf16_supported()
    out_dir = os.path.join("runs", args.name)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"), indent=2)
    print(f"device={device} bf16={amp} -> {out_dir}")

    # ---- data: BDD split 80/10/10, BITS = test only
    items = list_images(args.train_dir)
    random.Random(args.seed).shuffle(items)
    if args.limit:
        items = [it for c in (0, 1) for it in [x for x in items if x[1] == c][:args.limit]]
        random.Random(args.seed).shuffle(items)
    n = len(items)
    va_items, te_items, tr_items = items[:n // 10], items[n // 10:n // 5], items[n // 5:]
    bits_items = list_images(args.test_dir)
    print(f"BDD train {len(tr_items)} | val {len(va_items)} | test {len(te_items)} | BITS {len(bits_items)}")

    # ---- encoder features (frozen, cached)
    enc_dtype = torch.float16 if device == "cuda" else torch.float32
    proc = AutoImageProcessor.from_pretrained(args.encoder)
    enc = SiglipVisionModel.from_pretrained(args.encoder, dtype=enc_dtype).to(device).eval()
    Xtr = torch.stack([extract(tr_items, v, "train", args, enc, proc) for v in range(args.n_aug + 1)], 1)
    Xva = extract(va_items, 0, "val", args, enc, proc)
    Xte = extract(te_items, 0, "test", args, enc, proc)
    Xbits = extract(bits_items, 0, "bits", args, enc, proc) if bits_items else None
    ytr, yva, yte = (torch.tensor([l for _, l in it]) for it in (tr_items, va_items, te_items))
    ybits = torch.tensor([l for _, l in bits_items]) if bits_items else None
    vis_dim = Xtr.shape[-1]
    print(f"visual tokens per image: {Xtr.shape[2]} x {vis_dim}  (views per train image: {Xtr.shape[1]})")
    del enc
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- SLM + projector (+ LoRA)
    tok = AutoTokenizer.from_pretrained(args.slm)
    slm = AutoModelForCausalLM.from_pretrained(args.slm, dtype=torch.float32).to(device)
    for p in slm.parameters():
        p.requires_grad_(False)
    if args.mode == "lora":
        vlm.add_lora(slm, args.lora_r, args.lora_alpha)
    elif args.mode == "full":
        for p in slm.model.layers.parameters():
            p.requires_grad_(True)
        slm.model.norm.weight.requires_grad_(True)
        slm.gradient_checkpointing_enable()
    projector = vlm.Projector(vis_dim, slm.config.hidden_size).to(device)
    model = vlm.Decider(slm, tok, projector)

    slm_params = [p for p in slm.parameters() if p.requires_grad]
    groups = [{"params": projector.parameters(), "lr": args.lr}]
    if slm_params:
        groups.append({"params": slm_params, "lr": args.lr if args.mode == "lora" else args.slm_lr})
    n_train = sum(p.numel() for p in projector.parameters()) + sum(p.numel() for p in slm_params)
    print(f"mode={args.mode}  trainable params: {n_train / 1e6:.2f}M")

    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(Xtr) / args.batch)
    warm = max(1, int(args.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    class_w = torch.tensor([1.0, args.stop_weight], device=device)

    @torch.no_grad()
    def scores(X):
        model.eval()
        out = []
        for i in range(0, len(X), 64):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out.append(vlm.stop_score(model(X[i:i + 64].to(device).float())).float().cpu())
        model.train()
        return torch.cat(out)

    def save(path, thr, epoch):
        trainable = {n: p.detach().cpu() for n, p in slm.named_parameters() if p.requires_grad}
        torch.save({"args": vars(args), "projector": projector.state_dict(), "slm_trainable": trainable,
                    "threshold": thr, "epoch": epoch, "vis_dim": vis_dim}, path)

    # ---- train
    log = csv.writer(open(os.path.join(out_dir, "log.csv"), "w", newline="", buffering=1))  # line-buffered: live progress
    log.writerow(["epoch", "train_loss", "val_auroc", "val_acc@0", "val_STOP_recall@0", "seconds"])
    best_auc, step = -1.0, 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        t0, total, perm = time.time(), 0.0, torch.randperm(len(Xtr))
        for b, i in enumerate(range(0, len(perm), args.batch)):
            idx = perm[i:i + args.batch]
            view = torch.randint(0, Xtr.shape[1], (len(idx),))
            feats, y = Xtr[idx, view].to(device).float(), ytr[idx].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(feats)
            loss = F.cross_entropy(logits.float(), y, weight=class_w, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(projector.parameters()) + slm_params, 1.0)
            opt.step()
            sched.step()
            total += loss.item()
            step += 1
            if b % 50 == 0:
                print(f"epoch {epoch} step {b} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}")
        sv = scores(Xva)
        m = metrics(sv, yva, 0.0)
        dt = time.time() - t0
        print(f"== epoch {epoch}  loss {total / (b + 1):.4f}  ({dt:.0f}s)")
        show("val", m)
        log.writerow([epoch, round(total / (b + 1), 4), m["auroc"], m["acc"], m["STOP_recall"], round(dt)])
        if m["auroc"] > best_auc:
            best_auc = m["auroc"]
            save(os.path.join(out_dir, "best.pt"), threshold_for_recall(sv, yva, args.target_recall), epoch)
            print(f"   saved best (val auroc {best_auc:.4f})")

    # ---- final report with the best checkpoint
    ck = torch.load(os.path.join(out_dir, "best.pt"))
    projector.load_state_dict(ck["projector"])
    slm.load_state_dict(ck["slm_trainable"], strict=False)
    thr = ck["threshold"]
    report = {"best_epoch": ck["epoch"], "target_STOP_recall": args.target_recall, "threshold": thr}
    print(f"\nFinal (best epoch {ck['epoch']}); threshold tuned on val for STOP recall >= {args.target_recall}")
    sets = [("val", Xva, yva), ("BDD test", Xte, yte)] + ([("BITS", Xbits, ybits)] if Xbits is not None else [])
    for name, X, y in sets:
        s = scores(X)
        for tag, t in (("argmax", 0.0), ("safety", thr)):
            m = metrics(s, y, t)
            report[f"{name}|{tag}"] = m
            show(f"{name[:5]}/{tag[:4]}", m)
        if name == "BITS":                                       # per-image scores for error analysis
            with open(os.path.join(out_dir, "bits_scores.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path", "label", "stop_score", "pred_safety"])
                for (p, l), sc in zip(bits_items, s.tolist()):
                    w.writerow([p, vlm.LABELS[l], round(sc, 4), vlm.LABELS[int(sc > thr)]])
    json.dump(report, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    print(f"\nSaved {out_dir}/best.pt, metrics.json, log.csv")


if __name__ == "__main__":
    main()
