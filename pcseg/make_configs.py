"""Build Pointcept configs for the experiments from the official PTv3 S3DIS config (RGB only, no normals).

Changes w.r.t. configs/s3dis/semseg-pt-v3m1-1-rpe.py (PTv3 + RPE, flash attention already disabled):
  * in_channels 6 -> 3 and feat_keys ("color","normal") -> ("color",)   (our data has no normals; new upsampled points have none)
  * budget knobs: batch size, workers, SphereCrop point_max, total epochs, linear LR scaling
  * evaluate=False (no validation on Area 5 during training -> no checkpoint selection on the test set; last checkpoint is used)
  * wandb off; optional density augmentation (RandomSparsify); optional single-view test (no TTA)
"""
import argparse
import sys
from pathlib import Path

from pcseg import settings

BASE = "configs/s3dis/semseg-pt-v3m1-1-rpe.py"
TRAIN_AREAS = ("Area_1", "Area_2", "Area_3", "Area_4", "Area_6")


def _set_color_only(transforms):
    for t in transforms:
        if t["type"] == "Collect":
            t["feat_keys"] = ("color",)


def make_config(name, train_root, test_root, *, pointcept_dir=None, batch_size=4, num_worker=4, point_max=100000,
                epoch=100, eval_epoch=10, sparsify_aug=False, keep_range=(0.1, 0.5), tta="single",
                base_lr=0.006, base_bs=12, seed=0, train_areas=TRAIN_AREAS, out_dir=None, amp=True):
    pointcept_dir = Path(pointcept_dir or settings.POINTCEPT)
    if str(pointcept_dir) not in sys.path:
        sys.path.insert(0, str(pointcept_dir))
    from pointcept.utils.config import Config

    cfg = Config.fromfile(str(pointcept_dir / BASE))
    cfg.model.backbone.in_channels = 3
    cfg.batch_size, cfg.num_worker = batch_size, num_worker
    cfg.epoch, cfg.eval_epoch = epoch, eval_epoch
    assert epoch % eval_epoch == 0, "epoch must be a multiple of eval_epoch (checkpoint every epoch//eval_epoch passes)"
    cfg.evaluate = False
    cfg.enable_amp = bool(amp)
    cfg.enable_wandb = False
    cfg.seed = seed
    if "hooks" in cfg:
        cfg.hooks = [h for h in cfg.hooks if h.get("type") not in ("PreciseEvaluator", "SemSegEvaluator")]
    lr = base_lr * batch_size / base_bs
    cfg.optimizer.lr = lr
    cfg.scheduler.max_lr = [lr, lr / 10]
    cfg.param_dicts = [dict(keyword="block", lr=lr / 10)]

    d = cfg.data
    d.train.data_root, d.val.data_root, d.test.data_root = str(train_root), str(test_root), str(test_root)
    d.train.split = tuple(train_areas)
    _set_color_only(d.train.transform)
    _set_color_only(d.val.transform)
    _set_color_only(d.test.test_cfg.post_transform)
    tr = d.train.transform
    for t in tr:
        if t["type"] == "SphereCrop" and "point_max" in t:
            t["point_max"] = point_max
    if sparsify_aug:
        gi = next(i for i, t in enumerate(tr) if t["type"] == "GridSample")
        tr.insert(gi, dict(type="RandomSparsify", keep_range=tuple(keep_range), p=0.8))
    if tta == "single":
        d.test.test_cfg.aug_transform = [[dict(type="RandomScale", scale=[1, 1])]]
    elif tta != "full":
        raise ValueError("tta must be 'single' or 'full'")
    cfg.save_path = str(settings.EXP / name)

    out_dir = Path(out_dir or pointcept_dir / "configs" / "s3dis")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pcseg_{name}.py"
    cfg.dump(str(path))
    return str(path)


def iterations(n_rooms, batch_size, epoch):
    """Total optimisation steps = rooms x epoch / batch (Pointcept repeats the room list epoch//eval_epoch times per pass)."""
    return n_rooms * epoch // batch_size


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--point-max", type=int, default=100000)
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--eval-epoch", type=int, default=10)
    ap.add_argument("--sparsify-aug", action="store_true")
    ap.add_argument("--tta", default="single")
    a = ap.parse_args()
    print(make_config(a.name, a.train_root, a.test_root, batch_size=a.batch_size, point_max=a.point_max,
                      epoch=a.epoch, eval_epoch=a.eval_epoch, sparsify_aug=a.sparsify_aug, tta=a.tta))
