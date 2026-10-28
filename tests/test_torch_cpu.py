"""End-to-end smoke test of the upsampler half on CPU with synthetic data (needs PyTorch, no GPU, no Pointcept, ~1-2 min).
    python tests/test_torch_cpu.py
"""
import os
import sys
import tempfile
from pathlib import Path

tmp = tempfile.mkdtemp(prefix="pcseg_test_")
os.environ["PCSEG_ROOT"] = tmp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from pcseg import pu_infer, pu_train, settings
from pcseg.common import load_room, room_dirs
from pcseg.pu_core import PUNet, interp_patch, load_upsampler, total_loss
from pcseg.pu_data import build_patch_bank, upsample_scene
from pcseg.up_metrics import measure_upsampler_cost, scene_metrics
from synth import make_dataset

settings.ensure_dirs()
make_dataset(settings.DATA, rooms_per_area=2)

# model + loss shapes
x = torch.randn(2, 256, 3) * 0.5
m = PUNet(up_ratio=4)
y = m(x)
assert y.shape == (2, 1024, 3) and interp_patch(x, 4).shape == (2, 1024, 3)
loss, parts = total_loss(y, torch.randn(2, 1024, 3) * 0.5)
loss.backward()

# banks -> train 2 epochs -> checkpoints
build_patch_bank(settings.DENSE, settings.SP25, ["Area_1", "Area_2", "Area_3", "Area_4"], settings.BANKS / "tr.npz", 0.25, 4, patches_per_room=24)
build_patch_bank(settings.DENSE, settings.SP25, ["Area_6"], settings.BANKS / "va.npz", 0.25, 4, patches_per_room=8, seed=1)
out = settings.UPSAMPLER / "tiny"
pu_train.main(["--train-bank", str(settings.BANKS / "tr.npz"), "--val-bank", str(settings.BANKS / "va.npz"), "--out", str(out), "--epochs", "2", "--batch", "8"])
assert (out / "last.pt").exists() and (out / "best.pt").exists() and (out / "log.json").exists()
pu_train.main(["--train-bank", str(settings.BANKS / "tr.npz"), "--val-bank", str(settings.BANKS / "va.npz"), "--out", str(out)])   # must skip

# scene inference through the CLI path (CPU), with and without labels
model = load_upsampler(out / "last.pt", "cpu")
sp = load_room(room_dirs(settings.SP25, "Area_5")[0], ("coord", "color"))
xyz, rgb, info = upsample_scene(model, sp["coord"], sp["color"], up_ratio=4, device="cpu", verbose=False)
assert xyz.shape[1] == 3 and len(xyz) <= 4 * len(sp["coord"]) and len(xyz) == len(rgb) and info["coverage"] > 0.5
dst = settings.DATA / "s3dis_up25x4"
pu_infer.main(["--method", "model", "--ckpt", str(out / "last.pt"), "--sparse-root", str(settings.SP25), "--dense-root", str(settings.DENSE),
               "--dst", str(dst), "--areas", "Area_5", "Area_1", "--label-areas", "Area_1"])
assert (np.load(dst / "Area_1" / "office_1" / "segment.npy") >= 0).all() and (np.load(dst / "Area_5" / "office_1" / "segment.npy") == -1).all()
dn = load_room(room_dirs(settings.DENSE, "Area_5")[0], ("coord",))["coord"]
print("scene metrics:", {k: round(v, 3) for k, v in scene_metrics(xyz, dn).items()})
print("cost:", measure_upsampler_cost(model, settings.SP25, "Area_5", device="cpu", rooms=1))
print("\nALL TORCH-CPU TESTS PASSED")
