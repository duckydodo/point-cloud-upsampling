"""Tiny synthetic S3DIS-like dataset (Pointcept layout) for smoke tests: planes with colours and labels."""
from pathlib import Path

import numpy as np


def make_room(rng, n=9000):
    # points on the 6 faces of a 5 x 4 x 3 m box + a few "furniture" blobs; label = face id (0..5) or 7+ for blobs
    face = rng.integers(0, 6, n)
    u, v = rng.random(n), rng.random(n)
    xyz = np.zeros((n, 3))
    dims = np.array([5.0, 4.0, 3.0])
    for f in range(6):
        m = face == f
        ax = f // 2
        other = [i for i in range(3) if i != ax]
        xyz[m, ax] = 0.0 if f % 2 == 0 else dims[ax]
        xyz[m, other[0]] = u[m] * dims[other[0]]
        xyz[m, other[1]] = v[m] * dims[other[1]]
    seg = face.astype(np.int32)
    blob = rng.random(n) < 0.15
    xyz[blob] = rng.normal([2.5, 2.0, 0.5], 0.3, (blob.sum(), 3))
    seg[blob] = 7
    color = np.clip(np.stack([face * 40 + 20, 255 - face * 30, rng.integers(0, 255, n)], 1) + rng.normal(0, 5, (n, 3)), 0, 255).astype(np.uint8)
    return xyz.astype(np.float32), color, seg


def make_dataset(root, rooms_per_area=2, n=9000, seed=0, keep=0.25):
    rng = np.random.default_rng(seed)
    root = Path(root)
    for sub in ("s3dis", "s3dis_sparse25"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for a in range(1, 7):
        for r in range(rooms_per_area):
            xyz, col, seg = make_room(rng, n)
            name = f"office_{r + 1}"
            d = root / "s3dis" / f"Area_{a}" / name
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "coord.npy", xyz); np.save(d / "color.npy", col); np.save(d / "segment.npy", seg)
            idx = np.sort(rng.choice(n, int(n * keep), replace=False))      # exact subset, like the real sparse sets
            s = root / "s3dis_sparse25" / f"Area_{a}" / name
            s.mkdir(parents=True, exist_ok=True)
            np.save(s / "coord.npy", xyz[idx]); np.save(s / "color.npy", col[idx]); np.save(s / "segment.npy", seg[idx])
