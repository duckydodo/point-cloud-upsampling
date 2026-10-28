"""Shared constants and small I/O helpers for the pcseg project (S3DIS, Pointcept layout)."""
import json
import os
from pathlib import Path

import numpy as np

CLASS_NAMES = ["ceiling", "floor", "wall", "beam", "column", "window", "door",
               "table", "chair", "sofa", "bookcase", "board", "clutter"]
NUM_CLASSES = len(CLASS_NAMES)
TRAIN_AREAS = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"]
TEST_AREA = "Area_5"

# colour palette (RGB 0-255) for visualisation; 13 distinct colours (the Colab version used the same colour for beam and window)
PALETTE = np.array([
    [233, 229, 107], [95, 156, 196], [179, 116, 81], [81, 163, 148], [241, 149, 131],
    [160, 120, 210], [77, 174, 84], [108, 135, 75], [41, 49, 101], [223, 52, 52],
    [89, 47, 95], [214, 109, 77], [100, 100, 100]], dtype=np.uint8)


def room_dirs(root, areas):
    """Sorted list of room directories (Path) under root/<area>/ for the given areas."""
    root = Path(root)
    areas = [areas] if isinstance(areas, str) else list(areas)
    out = []
    for a in areas:
        out += sorted(d for d in (root / a).iterdir() if d.is_dir())
    return out


def load_room(path, keys=("coord", "color", "segment")):
    """Load per-point arrays of one room folder. coord->float32 (N,3), color->float32 (N,3), segment->int32 (N,)."""
    path = Path(path)
    out = {}
    for k in keys:
        f = path / f"{k}.npy"
        if not f.exists():
            continue
        a = np.load(f)
        if k in ("coord", "color"):
            a = a.astype(np.float32)
        elif k == "segment":
            a = a.reshape(-1).astype(np.int32)
        out[k] = a
    return out


def save_room(out_dir, coord, color=None, segment=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "coord.npy", np.asarray(coord, dtype=np.float32))
    if color is not None:
        np.save(out_dir / "color.npy", np.clip(np.rint(color), 0, 255).astype(np.uint8))
    if segment is not None:
        np.save(out_dir / "segment.npy", np.asarray(segment, dtype=np.int32).reshape(-1))


def read_json(p):
    with open(p) as f:
        return json.load(f)


def write_json(p, obj):
    os.makedirs(os.path.dirname(str(p)) or ".", exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
