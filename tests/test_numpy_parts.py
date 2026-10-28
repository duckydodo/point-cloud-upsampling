"""Smoke test of everything that does not need PyTorch/GPU (data layout, patch banks, scoring, tables).
    python tests/test_numpy_parts.py
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

from pcseg import settings, data_check, eval_dense, results
from pcseg.common import load_room, room_dirs, write_json
from pcseg.pu_data import build_patch_bank, interp_scene, write_upsampled_dataset
from pcseg.up_metrics import scene_metrics
from synth import make_dataset

settings.ensure_dirs()
make_dataset(settings.DATA)

# 1. data check (Area_5 has 2 rooms here, not 68, so the room-count rule must trigger; nothing else may)
problems = []
data_check._check_root(settings.DENSE, "dense", True, problems)
assert problems == ["dense: Area_5 has 2 rooms (expected 68)"], problems
assert data_check._check_root(settings.SP25, "sp", False, [], expect_a5=2)

# 2. patch bank
bank = settings.BANKS / "b.npz"
build_patch_bank(settings.DENSE, settings.SP25, ["Area_1", "Area_2"], bank, keep_ratio=0.25, up_ratio=4, patches_per_room=10)
z = np.load(bank)
assert z["x"].shape == (40, 256, 3) and z["y"].shape == (40, 1024, 3), (z["x"].shape, z["y"].shape)
assert np.abs(z["x"]).max() <= 1.0 + 1e-5                     # unit sphere
assert not (settings.BANKS / "b.npz.tmp.npz").exists()
build_patch_bank(settings.DENSE, settings.SP25, ["Area_1"], settings.BANKS / "b2.npz", keep_ratio=0.25, up_ratio=2, patches_per_room=5)
assert np.load(settings.BANKS / "b2.npz")["y"].shape == (10, 512, 3)       # x2 on a 25 % scan -> GT subset of the dense ball

# 3. interpolation baseline + scene metrics
sp = load_room(room_dirs(settings.SP25, "Area_5")[0], ("coord", "color"))
dn = load_room(room_dirs(settings.DENSE, "Area_5")[0], ("coord", "color", "segment"))
xyz, rgb = interp_scene(sp["coord"], sp["color"], 4)
assert len(xyz) == 4 * len(sp["coord"])
m_sp, m_in = scene_metrics(sp["coord"], dn["coord"]), scene_metrics(xyz, dn["coord"])
assert m_in["f@2cm"] >= 0 and "uniformity_cv" in m_in
print("scene metrics sparse :", {k: round(v, 3) for k, v in m_sp.items()})
print("scene metrics interp :", {k: round(v, 3) for k, v in m_in.items()})

# 4. writing a Pointcept-format upsampled dataset (labels only for label_areas)
dst = settings.DATA / "s3dis_interp25x4"
write_upsampled_dataset(dst, settings.SP25, ["Area_1", "Area_5"], lambda c, col: interp_scene(c, col, 4),
                        dense_root=settings.DENSE, label_areas={"Area_1"})
seg1 = np.load(dst / "Area_1" / "office_1" / "segment.npy"); seg5 = np.load(dst / "Area_5" / "office_1" / "segment.npy")
assert (seg1 >= 0).all() and (seg5 == -1).all()
assert len(seg1) == len(np.load(dst / "Area_1" / "office_1" / "coord.npy"))

# 5. scoring: perfect dense predictions -> mIoU 1; predictions on a sparse input are mapped to the dense points
pred_dir = Path(tmp) / "pred_dense"; pred_dir.mkdir()
for rd in room_dirs(settings.DENSE, "Area_5"):
    np.save(pred_dir / f"Area_5-{rd.name}_pred.npy", load_room(rd, ("segment",))["segment"])
r = eval_dense.evaluate(pred_dir, settings.DENSE, settings.DENSE, "Area_5", verbose=False)
assert abs(r["mIoU"] - 1.0) < 1e-9 and abs(r["OA"] - 1.0) < 1e-9
pred_sp = Path(tmp) / "pred_sparse"; pred_sp.mkdir()
for rd in room_dirs(settings.SP25, "Area_5"):
    np.save(pred_sp / f"Area_5-{rd.name}_pred.npy", load_room(rd, ("segment",))["segment"])
r2 = eval_dense.evaluate(pred_sp, settings.SP25, settings.DENSE, "Area_5", verbose=False)
assert 0.5 < r2["mIoU"] < 1.0, r2["mIoU"]                       # nearest-neighbour transfer loses a little at plane edges
assert r2["points"] == r["points"]

# 6. results table
write_json(settings.RESULTS / "A_dense_on_dense.json", r)
write_json(settings.RESULTS / "C_dense_model_on_sparse25.json", r2)
rows = results.collect()
assert [x["name"] for x in rows] == ["A_dense_on_dense", "C_dense_model_on_sparse25"]
print(results.to_markdown(rows))

# 7. knobs file round trip + suggestions
k = settings.load_knobs(); k.batch, k.epoch = 6, 50; k.save()
k2 = settings.load_knobs(); assert (k2.batch, k2.epoch) == (6, 50)
assert settings.suggest_knobs(8)["batch"] == 2 and settings.suggest_knobs(48)["point_max"] == 204800
print("\nALL NUMPY-ONLY TESTS PASSED  (temp dir:", tmp, ")")
