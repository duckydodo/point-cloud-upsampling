"""Score Pointcept predictions on the ORIGINAL dense points of a split.

Pointcept's tester saves <result>/<Area>-<room>_pred.npy aligned with the points of the dataset it was run on
(dense, sparse or upsampled). Here every prediction is mapped to the original dense points by 1-nearest-neighbour
and compared with the original S3DIS ground truth, so all variants are scored on exactly the same points.
"""
import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pcseg.common import CLASS_NAMES, NUM_CLASSES, load_room, room_dirs, write_json


def metrics_from_confusion(conf):
    tp = np.diag(conf).astype(np.float64)
    gt = conf.sum(1).astype(np.float64)
    pr = conf.sum(0).astype(np.float64)
    union = gt + pr - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    acc = np.where(gt > 0, tp / np.maximum(gt, 1), np.nan)
    return dict(mIoU=float(np.nanmean(iou)), mAcc=float(np.nanmean(acc)), OA=float(tp.sum() / max(conf.sum(), 1)),
                per_class_iou={n: (None if np.isnan(v) else float(v)) for n, v in zip(CLASS_NAMES, iou)},
                per_class_acc={n: (None if np.isnan(v) else float(v)) for n, v in zip(CLASS_NAMES, acc)})


def evaluate(pred_dir, input_root, dense_root, split="Area_5", out_json=None, verbose=True):
    pred_dir, input_root, dense_root = Path(pred_dir), Path(input_root), Path(dense_root)
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    t0, n_rooms = time.time(), 0
    for rd in room_dirs(dense_root, split):
        room = rd.name
        pf = pred_dir / f"{split}-{room}_pred.npy"
        if not pf.exists():
            raise FileNotFoundError(f"missing prediction {pf}")
        pred = np.load(pf).reshape(-1)
        dn = load_room(rd, ("coord", "segment"))
        if input_root.resolve() == dense_root.resolve():
            dense_pred = pred
        else:
            src = load_room(input_root / split / room, ("coord",))["coord"]
            assert len(src) == len(pred), f"{room}: {len(pred)} predictions for {len(src)} input points"
            _, nn = cKDTree(src).query(dn["coord"], k=1, workers=-1)
            dense_pred = pred[nn]
        gt = dn["segment"]
        ok = (gt >= 0) & (gt < NUM_CLASSES)
        conf += np.bincount(gt[ok].astype(np.int64) * NUM_CLASSES + dense_pred[ok].astype(np.int64), minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
        n_rooms += 1
        if verbose:
            print(f"  {split}/{room}: {len(gt):,} dense points ({time.time() - t0:.0f}s)", flush=True)
    res = metrics_from_confusion(conf)
    res.update(split=split, rooms=n_rooms, confusion=conf.tolist(), points=int(conf.sum()))
    if out_json:
        write_json(out_json, res)
    if verbose:
        print(f"mIoU {res['mIoU'] * 100:.2f} | mAcc {res['mAcc'] * 100:.2f} | OA {res['OA'] * 100:.2f}  ({n_rooms} rooms)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="Pointcept <save_path>/result")
    ap.add_argument("--input-root", required=True, help="dataset root the predictions were computed on")
    ap.add_argument("--dense-root", required=True)
    ap.add_argument("--split", default="Area_5")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    evaluate(a.pred_dir, a.input_root, a.dense_root, a.split, a.out)
