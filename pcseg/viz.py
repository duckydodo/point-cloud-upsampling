"""Qualitative figures: oblique views of a room for sparse / upsampled / prediction / ground truth / error."""
import numpy as np

from pcseg.common import PALETTE


def _sub(n, m, seed=0):
    return np.arange(n) if n <= m else np.sort(np.random.default_rng(seed).choice(n, m, replace=False))


def plot_panels(panels, out_png, max_points=60000, elev=55, azim=-60, figsize=4.2):
    """panels: list of (title, coord (N,3), colour (N,3) in 0-1 or 0-255). Same camera for every panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ref = panels[0][1]
    lo, hi = ref.min(0), ref.max(0)
    fig = plt.figure(figsize=(figsize * len(panels), figsize))
    for i, (title, xyz, col) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        j = _sub(len(xyz), max_points)
        c = np.asarray(col)[j].astype(np.float32)
        c = c / 255.0 if c.max() > 1.0 else c
        ax.scatter(xyz[j, 0], xyz[j, 1], xyz[j, 2], c=c, s=0.8, linewidths=0, depthshade=False)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f"{title}\n{len(xyz):,} pts", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)
    return str(out_png)


def label_colors(seg):
    seg = np.asarray(seg)
    out = np.full((len(seg), 3), 255, dtype=np.uint8)
    ok = seg >= 0
    out[ok] = PALETTE[seg[ok] % len(PALETTE)]
    return out


def error_colors(pred, gt):
    c = np.zeros((len(gt), 3), dtype=np.uint8)
    ok = gt >= 0
    c[ok & (pred == gt)] = (200, 200, 200)
    c[ok & (pred != gt)] = (220, 30, 30)
    return c
