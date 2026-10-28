"""Scene-level upsampler metrics (metres) and cost measurement."""
import time

import numpy as np
from scipy.spatial import cKDTree

from pcseg.common import load_room, room_dirs


def scene_metrics(up_xyz, dense_xyz, taus=(0.01, 0.02), n=200000, seed=0):
    """Distances between an upsampled scene and the dense ground-truth scene (both subsampled to n points).
    chamfer_cm : mean NN distance up->dense plus dense->up (cm, L1 flavour); hd_cm : 99th-percentile symmetric (robust Hausdorff);
    f@tau      : F-score at thresholds (metres); uniformity_cv : CV of mean 8-NN distance on the upsampled scene."""
    rng = np.random.default_rng(seed)
    u = up_xyz[rng.choice(len(up_xyz), min(n, len(up_xyz)), replace=False)]
    g = dense_xyz[rng.choice(len(dense_xyz), min(n, len(dense_xyz)), replace=False)]
    tu, tg = cKDTree(u), cKDTree(g)
    d_ug = tg.query(u, k=1, workers=-1)[0]
    d_gu = tu.query(g, k=1, workers=-1)[0]
    out = dict(chamfer_cm=float((d_ug.mean() + d_gu.mean()) * 100), hd99_cm=float(max(np.percentile(d_ug, 99), np.percentile(d_gu, 99)) * 100))
    for t in taus:
        p, r = (d_ug < t).mean(), (d_gu < t).mean()
        out[f"f@{int(t * 100)}cm"] = float(2 * p * r / max(p + r, 1e-9))
    k8 = tu.query(u[:min(len(u), 50000)], k=9, workers=-1)[0][:, 1:].mean(1)
    out["uniformity_cv"] = float(k8.std() / max(k8.mean(), 1e-9))
    return out


def average(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def measure_upsampler_cost(model, sparse_root, area, device="cuda", rooms=3, up_ratio=4, **kw):
    """Parameter count, wall time per room and peak GPU memory of the upsampler on a few rooms."""
    import torch
    from pcseg.pu_data import upsample_scene
    n_par = sum(p.numel() for p in model.parameters())
    times, pts = [], []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for rd in room_dirs(sparse_root, area)[:rooms]:
        d = load_room(rd, ("coord", "color"))
        t0 = time.time()
        upsample_scene(model, d["coord"], d["color"], up_ratio=up_ratio, device=device, verbose=False, **kw)
        times.append(time.time() - t0)
        pts.append(len(d["coord"]))
    mem = torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else float("nan")
    return dict(params=n_par, sec_per_room=float(np.mean(times)), sec_per_million_input_pts=float(np.sum(times) / (np.sum(pts) / 1e6)),
                peak_gpu_mb=float(mem))
