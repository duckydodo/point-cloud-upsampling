"""Patch bank construction (training pairs), scene-level upsampling inference, and non-learned baselines."""
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pcseg.common import load_room, room_dirs, save_room


# ----------------------------------------------------------------------------- patch bank
def _voxel_pick(coord, cell, rng, m):
    """Pick m point indices spread uniformly over space (one random point per occupied voxel, then sample voxels)."""
    q = np.floor((coord - coord.min(0)) / cell).astype(np.int64)
    dims = q.max(0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    perm = rng.permutation(len(coord))
    _, first = np.unique(key[perm], return_index=True)
    cand = perm[first]
    if len(cand) >= m:
        return rng.choice(cand, m, replace=False)
    return rng.choice(cand, m, replace=True)


def build_patch_bank(dense_root, sparse_root, areas, out_path, keep_ratio, up_ratio=4, n_in=256,
                     patches_per_room=100, cell=0.1, seed=0, max_rooms=None, skip_existing=True):
    """Create (sparse patch, dense patch) pairs.
    sparse patch : n_in nearest sparse points around a centre
    dense patch  : n_in/keep_ratio nearest dense points around the same centre; if up_ratio*n_in is smaller,
                   a random subset of that size is kept (e.g. x2 on a 25% scan -> GT at 50% density)
    Both are centred on the sparse patch centroid and divided by the sparse patch radius (unit sphere)."""
    if skip_existing and Path(out_path).exists():
        print(f"bank already exists, skipping: {out_path}")
        return out_path
    rng = np.random.default_rng(seed)
    n_gt = n_in * up_ratio
    ball = int(round(n_in / keep_ratio))
    assert n_gt <= ball, f"cannot upsample x{up_ratio} a {keep_ratio:.3f} scan (needs up_ratio <= {1 / keep_ratio:.1f})"
    X, Y, meta = [], [], []
    rooms = room_dirs(sparse_root, areas)
    if max_rooms:
        rooms = rooms[:max_rooms]
    for i, rd in enumerate(rooms):
        area, room = rd.parent.name, rd.name
        sp = load_room(rd, ("coord",))["coord"]
        de = load_room(Path(dense_root) / area / room, ("coord",))["coord"]
        if len(sp) < n_in or len(de) < ball:
            continue
        ctr = sp[_voxel_pick(sp, cell, rng, patches_per_room)]
        _, i_sp = cKDTree(sp).query(ctr, k=n_in, workers=-1)
        _, i_de = cKDTree(de).query(ctr, k=ball, workers=-1)
        P = sp[i_sp]                                           # (m, n_in, 3)
        G = de[i_de]                                           # (m, ball, 3)
        if n_gt < ball:
            sel = np.stack([rng.choice(ball, n_gt, replace=False) for _ in range(len(G))])
            G = np.take_along_axis(G, sel[..., None], axis=1)
        c = P.mean(1, keepdims=True)
        s = np.linalg.norm(P - c, axis=-1).max(1).reshape(-1, 1, 1).clip(1e-6)
        X.append(((P - c) / s).astype(np.float32))
        Y.append(((G - c) / s).astype(np.float32))
        meta += [f"{area}/{room}"] * len(P)
        print(f"[{i + 1}/{len(rooms)}] {area}/{room}: {len(P)} patches", flush=True)
    if not X:
        raise RuntimeError(f"no patches built from {sparse_root} {areas}: rooms too small or folders empty")
    X, Y = np.concatenate(X), np.concatenate(Y)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_path) + ".tmp.npz")             # write-then-rename: an interrupted run never leaves a half-written bank
    np.savez(tmp, x=X, y=Y, meta=np.array(meta), keep_ratio=keep_ratio, up_ratio=up_ratio, n_in=n_in)
    tmp.replace(out_path)
    print(f"saved {len(X)} patches to {out_path}  (x: {X.shape}, y: {Y.shape})")
    return out_path


# ----------------------------------------------------------------------------- scene inference
def fps_torch(xyz, m, seed=0):
    """Farthest point sampling. xyz (N,3) torch tensor on any device -> LongTensor (m,)."""
    import torch
    n = xyz.shape[0]
    m = min(m, n)
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.empty(m, dtype=torch.long, device=xyz.device)
    dist = torch.full((n,), float("inf"), device=xyz.device)
    far = int(torch.randint(0, n, (1,), generator=g))
    for i in range(m):
        idx[i] = far
        d = ((xyz - xyz[far]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = int(dist.argmax())
    return idx


def _colors_from_neighbours(tree, src_xyz, src_rgb, query, mode="nn", k=3):
    if mode == "nn":
        _, nn = tree.query(query, k=1, workers=-1)
        return src_rgb[nn]
    d, nn = tree.query(query, k=k, workers=-1)
    w = 1.0 / (d + 1e-6)
    w /= w.sum(1, keepdims=True)
    return (src_rgb[nn] * w[..., None]).sum(1)


def upsample_scene(model, coord, color, up_ratio=4, n_in=256, batch=64, device="cuda", rgb_mode="nn",
                   include_input=False, outlier_factor=6.0, fps_cover=2.0, rand_cover=2.0, seed=0, verbose=True):
    """Sparse scene (coord (N,3), color (N,3) in 0-255) -> upsampled (coord, color).
    Patches: FPS centres (guarantee spatial coverage) + random centres (guarantee density-proportional coverage),
    each patch = n_in nearest sparse points. Outputs of all patches are merged, filtered for outliers
    (distance to nearest input point > outlier_factor x median NN spacing) and randomly reduced to N*up_ratio points."""
    import torch
    rng = np.random.default_rng(seed)
    n = len(coord)
    tree = cKDTree(coord)
    m_f = int(math.ceil(fps_cover * n / n_in))
    m_r = int(math.ceil(rand_cover * n / n_in))
    xyz_t = torch.from_numpy(coord).to(device)
    fps_idx = fps_torch(xyz_t, m_f, seed).cpu().numpy()
    ctr = np.concatenate([fps_idx, rng.choice(n, m_r, replace=m_r > n)])
    _, nn = tree.query(coord[ctr], k=n_in, workers=-1)
    cover = np.unique(nn).size / n
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ctr), batch):
            P = torch.from_numpy(coord[nn[i:i + batch]])
            c = P.mean(1, keepdim=True)
            s = (P - c).norm(dim=-1).amax(1).view(-1, 1, 1).clamp_min(1e-6)
            Q = model(((P - c) / s).to(device)).float().cpu()
            outs.append((Q * s + c).numpy())
    up = np.concatenate(outs).reshape(-1, 3)
    target = n * (up_ratio - (1 if include_input else 0))
    if len(up) > 2 * target:                                  # cheap pre-reduction before the outlier query
        up = up[rng.choice(len(up), 2 * target, replace=False)]
    d1, _ = tree.query(up, k=1, workers=-1)
    sub = rng.choice(n, min(n, 20000), replace=False)
    spacing = np.median(tree.query(coord[sub], k=2, workers=-1)[0][:, 1])
    up = up[d1 <= outlier_factor * max(spacing, 1e-6)]
    if len(up) > target:
        up = up[rng.choice(len(up), target, replace=False)]
    col = _colors_from_neighbours(tree, coord, color, up, rgb_mode)
    if include_input:
        up, col = np.concatenate([coord, up]), np.concatenate([color, col])
    if verbose:
        print(f"  upsample: {n:,} -> {len(up):,} pts | patches {len(ctr):,} | input coverage {cover:.1%}", flush=True)
    return up.astype(np.float32), col.astype(np.float32), dict(coverage=cover, patches=len(ctr), spacing=float(spacing))


# ----------------------------------------------------------------------------- non-learned baseline
def interp_scene(coord, color, up_ratio=4):
    """Naive baseline: keep every point, add midpoints to its (up_ratio-1) nearest neighbours (colour = mean)."""
    tree = cKDTree(coord)
    _, nn = tree.query(coord, k=up_ratio, workers=-1)
    pts, cols = [coord], [color]
    for j in range(1, up_ratio):
        pts.append((coord + coord[nn[:, j]]) / 2.0)
        cols.append((color + color[nn[:, j]]) / 2.0)
    return np.concatenate(pts).astype(np.float32), np.concatenate(cols).astype(np.float32)


# ----------------------------------------------------------------------------- write a Pointcept-format dataset
def write_upsampled_dataset(dst_root, src_sparse_root, areas, make_fn, dense_root=None, label_areas=(), skip_existing=True, room_stride=1):
    """For every room: coord/color = make_fn(coord, color); segment from the dense room by 1-NN only for label_areas
    (training areas); other areas get segment=-1 (never used for scoring - scoring is done on the dense points)."""
    for rd in room_dirs(src_sparse_root, areas)[::room_stride]:
        area, room = rd.parent.name, rd.name
        out = Path(dst_root) / area / room
        if skip_existing and (out / "coord.npy").exists() and (out / "segment.npy").exists():
            continue
        d = load_room(rd, ("coord", "color"))
        print(f"{area}/{room}", flush=True)
        xyz, rgb = make_fn(d["coord"], d["color"])[:2]
        if area in label_areas:
            dn = load_room(Path(dense_root) / area / room, ("coord", "segment"))
            _, nn = cKDTree(dn["coord"]).query(xyz, k=1, workers=-1)
            seg = dn["segment"][nn]
        else:
            seg = -np.ones(len(xyz), dtype=np.int32)
        save_room(out, xyz, rgb, seg)
