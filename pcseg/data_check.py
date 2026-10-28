"""Sanity-check the processed S3DIS folders (Part 1 output) before spending GPU hours.

    python -m pcseg.data_check
Expected layout (Pointcept format): <data>/s3dis/Area_k/<room>/{coord,color,segment}.npy
and the same for s3dis_sparse25 (and optionally s3dis_sparse12)."""
import sys

import numpy as np
from scipy.spatial import cKDTree

from pcseg import settings
from pcseg.common import room_dirs

AREAS = [f"Area_{i}" for i in range(1, 7)]


def _check_root(root, name, need_segment, problems, expect_a5=68):
    if not root.is_dir():
        problems.append(f"{name}: folder not found: {root}")
        return None
    n_rooms = {}
    for a in AREAS:
        if not (root / a).is_dir():
            problems.append(f"{name}: missing {a}")
            continue
        rooms = room_dirs(root, a)
        n_rooms[a] = len(rooms)
        for rd in rooms:
            need = ["coord.npy", "color.npy"] + (["segment.npy"] if need_segment else [])
            miss = [f for f in need if not (rd / f).exists()]
            if miss:
                problems.append(f"{name}/{a}/{rd.name}: missing {miss}")
                break
    if n_rooms.get("Area_5") not in (None, expect_a5):
        problems.append(f"{name}: Area_5 has {n_rooms.get('Area_5')} rooms (expected {expect_a5})")
    return n_rooms


def check(verbose=True):
    problems = []
    d = _check_root(settings.DENSE, "s3dis (dense)", True, problems)
    s25 = _check_root(settings.SP25, "s3dis_sparse25", False, problems)
    s12 = _check_root(settings.SP12, "s3dis_sparse12 (optional)", False, [], ) if settings.SP12.is_dir() else None
    if verbose:
        print("data folder:", settings.DATA)
        print("rooms per area, dense   :", d)
        print("rooms per area, sparse25:", s25)
        print("sparse12:", s12 if s12 else "not found (only needed for the optional 12.5 % ablation)")
    if d and s25 and not problems:
        # spot check one Area 5 room: shapes, sparsity ratio, sparse points are a subset of the dense points
        rd = room_dirs(settings.DENSE, "Area_5")[0]
        dn, sp = np.load(rd / "coord.npy"), np.load(settings.SP25 / "Area_5" / rd.name / "coord.npy")
        seg = np.load(rd / "segment.npy")
        ratio = len(sp) / len(dn)
        dist = cKDTree(dn).query(sp[:20000], k=1)[0].max()
        if verbose:
            print(f"spot check {rd.name}: dense {len(dn):,} pts, sparse {len(sp):,} pts (ratio {ratio:.3f}), "
                  f"labels {seg.min()}..{seg.max()}, max distance sparse->dense {dist:.2e}")
        if not 0.2 < ratio < 0.3:
            problems.append(f"sparse25 ratio {ratio:.3f} is not about 0.25")
        if dist > 1e-4:
            problems.append("sparse25 points are not an exact subset of the dense points")
        if len(seg) != len(dn):
            problems.append("segment.npy length differs from coord.npy")
    if verbose:
        print("\nOK: data looks right" if not problems else "\nPROBLEMS:\n  " + "\n  ".join(problems))
    return not problems


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
