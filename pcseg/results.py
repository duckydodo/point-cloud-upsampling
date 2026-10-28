"""Collect result JSONs written by eval_dense.evaluate() into one table."""
import json
from pathlib import Path

from pcseg import settings
from pcseg.common import CLASS_NAMES


def collect(results_dir=None):
    rows = []
    for f in sorted(Path(results_dir or settings.RESULTS).glob("*.json")):
        r = json.load(open(f))
        if "mIoU" not in r:
            continue
        rows.append(dict(name=f.stem, mIoU=100 * r["mIoU"], OA=100 * r["OA"], mAcc=100 * r["mAcc"],
                         **{c: (None if r["per_class_iou"].get(c) is None else 100 * r["per_class_iou"][c]) for c in CLASS_NAMES}))
    return rows


def to_markdown(rows, cols=("mIoU", "OA", "mAcc")):
    head = "| experiment | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols)
    body = "\n".join("| " + r["name"] + " | " + " | ".join("-" if r[c] is None else f"{r[c]:.2f}" for c in cols) + " |" for r in rows)
    return head + "\n" + body
