"""Train the upsampler on a patch bank.

    python -m pcseg.pu_train --train-bank X.npz --val-bank V.npz --out workspace/upsampler/name [--epochs 30 --batch 32 ...]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from pcseg.pu_core import PUNet, total_loss, patch_metrics, interp_patch, count_params


def load_bank(path, device):
    z = np.load(path)
    return torch.from_numpy(z["x"]).to(device), torch.from_numpy(z["y"]).to(device), int(z["up_ratio"])


def augment(x, y):
    """Same random z-rotation / flips / scale on input and target (patches are unit-sphere normalised)."""
    B = x.shape[0]
    a = torch.rand(B, device=x.device) * 2 * math.pi
    c, s = torch.cos(a), torch.sin(a)
    R = torch.zeros(B, 3, 3, device=x.device)
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1
    flip = (torch.rand(B, 3, device=x.device) < 0.5).float() * 2 - 1
    flip[:, 2] = 1
    sc = (torch.rand(B, 1, 1, device=x.device) * 0.4 + 0.8)
    f = lambda t: ((t * flip.unsqueeze(1)) @ R.transpose(1, 2)) * sc
    return f(x), f(y)


@torch.no_grad()
def evaluate(model, X, Y, bs=64, baseline=False, r=4):
    model.eval()
    acc = {}
    for i in range(0, len(X), bs):
        x, y = X[i:i + bs], Y[i:i + bs]
        p = interp_patch(x, r) if baseline else model(x)
        for k, v in patch_metrics(p, y).items():
            acc.setdefault(k, []).append(v)
    return {k: torch.cat(v).mean().item() for k, v in acc.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bank", required=True)
    ap.add_argument("--val-bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-rep", type=float, default=0.1)
    ap.add_argument("--w-uni", type=float, default=0.1)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="retrain even if <out>/log.json already exists")
    a = ap.parse_args(argv)
    if (Path(a.out) / "log.json").exists() and not a.force:
        print(f"already trained, skipping: {a.out}  (use --force to retrain)")
        return

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y, r = load_bank(a.train_bank, dev)
    Xv, Yv, rv = load_bank(a.val_bank, dev)
    assert r == rv
    model = PUNet(up_ratio=r, k=a.k, width=a.width).to(dev)
    print(f"device={dev} params={count_params(model):,} train patches={len(X)} val patches={len(Xv)} ratio=x{r}")
    base = evaluate(model, Xv, Yv, baseline=True, r=r)
    print("naive interpolation on val:", {k: round(v, 4) for k, v in base.items()})

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps_per_epoch = max(1, len(X) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log, best = [], float("inf")
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(X), device=dev)
        t0, run = time.time(), []
        for it in range(steps_per_epoch):
            idx = perm[it * a.batch:(it + 1) * a.batch]
            x, y = augment(X[idx], Y[idx])
            loss, parts = total_loss(model(x), y, w_rep=a.w_rep, w_uni=a.w_uni)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run.append(loss.item())
        val = evaluate(model, Xv, Yv, r=r)
        row = dict(epoch=ep + 1, train_loss=float(np.mean(run)), time=time.time() - t0, **{f"val_{k}": v for k, v in val.items()})
        log.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        torch.save(dict(state=model.state_dict(), up_ratio=r, k=a.k, width=a.width, epoch=ep + 1), out / "last.pt")
        if val["cd"] < best:
            best = val["cd"]
            torch.save(dict(state=model.state_dict(), up_ratio=r, k=a.k, width=a.width, epoch=ep + 1), out / "best.pt")
    (out / "log.json").write_text(json.dumps(dict(args=vars(a), baseline_val=base, log=log), indent=2))
    print("done. best val CD (x1e3):", round(best, 4), "| interpolation:", round(base["cd"], 4))


if __name__ == "__main__":
    main()
