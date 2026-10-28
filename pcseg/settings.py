"""Paths and experiment knobs shared by every notebook and script.

Everything lives under one workspace folder (default: <project>/workspace, override with the PCSEG_ROOT
environment variable). Data folders can be moved elsewhere with PCSEG_DATA (e.g. a fast NVMe disk).

    workspace/
      data/s3dis, data/s3dis_sparse25, data/s3dis_sparse12     <- you copy these from the Part 1 backups
      data/s3dis_up25x4, data/s3dis_interp25x4, ...            <- written by Part 4
      Pointcept/            cloned automatically (pinned commit)
      exp/<run>/            PTv3 configs, logs, checkpoints, test predictions
      results/*.json        one file per evaluated row
      banks/*.npz           upsampler patch banks
      upsampler/<model>/    upsampler checkpoints + logs
      figures/              plots
      pcseg_settings.json   the training knobs (batch size, epochs, ...) shared by all notebooks
"""
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PCSEG_ROOT", PROJECT / "workspace")).expanduser().resolve()
DATA = Path(os.environ.get("PCSEG_DATA", ROOT / "data")).expanduser().resolve()
POINTCEPT = Path(os.environ.get("POINTCEPT_DIR", ROOT / "Pointcept")).expanduser().resolve()
EXP, RESULTS, BANKS = ROOT / "exp", ROOT / "results", ROOT / "banks"
UPSAMPLER, FIGURES, STUBS = ROOT / "upsampler", ROOT / "figures", ROOT / "stubs"

DENSE, SP25, SP12 = DATA / "s3dis", DATA / "s3dis_sparse25", DATA / "s3dis_sparse12"
UP25X4, INTERP25X4 = DATA / "s3dis_up25x4", DATA / "s3dis_interp25x4"
SETTINGS_FILE = ROOT / "pcseg_settings.json"


def ensure_dirs():
    for d in (ROOT, DATA, EXP, RESULTS, BANKS, UPSAMPLER, FIGURES):
        d.mkdir(parents=True, exist_ok=True)


@dataclass
class Knobs:
    """Training budget shared by ALL PTv3 runs (A, B, D, H). Never change them between rows."""
    batch: int = 4              # total batch size (rooms per step); LR is scaled linearly (paper: 0.006 at 12)
    point_max: int = 100000     # SphereCrop point_max per room (paper config: 204800)
    epoch: int = 100            # total passes over the training rooms (paper: 3000)
    eval_epoch: int = 10        # number of checkpoint/"eval" passes; epoch must be a multiple of it
    num_worker: int = 4         # data loader workers
    amp: bool = True            # mixed precision (set by env_setup.check_amp)
    stride: int = 1             # row H: upsample every k-th training room only (1 = all rooms)

    def save(self, path=None):
        path = Path(path or SETTINGS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def load_knobs(path=None) -> Knobs:
    """Knobs from pcseg_settings.json (defaults if the file does not exist yet)."""
    path = Path(path or SETTINGS_FILE)
    k = Knobs(num_worker=max(2, min(8, (os.cpu_count() or 4) // 2)))
    if path.exists():
        for key, val in json.loads(path.read_text()).items():
            if hasattr(k, key):
                setattr(k, key, val)
    return k


def gpu_info():
    """(name, total VRAM in GB, compute capability) of GPU 0, or None."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        p = torch.cuda.get_device_properties(0)
        return p.name, p.total_memory / 2 ** 30, f"{p.major}.{p.minor}"
    except Exception:
        return None


def suggest_knobs(vram_gb: float):
    """Starting point for (batch, point_max) by VRAM. These are ESTIMATES scaled from the 16 GB T4 setting of the
    Colab guide (batch 4, 100k points); run the benchmark and lower point_max / batch if it reports out-of-memory."""
    if vram_gb >= 40:
        return dict(batch=12, point_max=204800)      # the paper's setting
    if vram_gb >= 22:
        return dict(batch=8, point_max=100000)
    if vram_gb >= 14:
        return dict(batch=4, point_max=100000)       # what the T4 runs used
    if vram_gb >= 10:
        return dict(batch=4, point_max=60000)
    return dict(batch=2, point_max=60000)
