"""Collect result JSONs written by eval_dense.evaluate() into one table."""
import json
from pathlib import Path

from pcseg import settings
from pcseg.common import CLASS_NAMES


def collect(results_dir=None):
    rows = []
    for f in sorted(Path(results_dir or settings.RESULTS).glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            r = json.load(fp)
        if "mIoU" not in r:
            continue
        pc_iou = r.get("per_class_iou", {})
        rows.append(dict(name=f.stem, mIoU=100 * r["mIoU"], OA=100 * r["OA"], mAcc=100 * r["mAcc"],
                         **{c: (None if pc_iou.get(c) is None else 100 * pc_iou[c]) for c in CLASS_NAMES}))
    return rows


def to_markdown(rows, cols=("mIoU", "OA", "mAcc")):
    head = "| experiment | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols)
    body = "\n".join("| " + r["name"] + " | " + " | ".join("-" if r[c] is None else f"{r[c]:.2f}" for c in cols) + " |" for r in rows)
    return head + "\n" + body
