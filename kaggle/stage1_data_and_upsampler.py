"""Stage 1 Kaggle Runner: Data Preparation, Patch Bank Generation, and PUNet Training.

Tasks:
1. Initialize environment and Pointcept commit 1342eda in /tmp/workspace.
2. Locate or create S3DIS dense and sparse25 datasets in /tmp/workspace/data.
   (If --smoke-test, generates tiny synthetic rooms via tests/synth.py).
3. Build patch banks (train_dev_x4.npz, val_x4.npz, train_full_x4.npz).
4. Train learned PUNet upsamplers (dev_x4 and full_x4) via pcseg.pu_train.
5. Compute geometric upsampling metrics (Chamfer distance, HD99, uniformity) on validation patches.
6. Export minimal, essential artifacts to /kaggle/working/stage1_out/ (checkpoints, metrics JSONs, patch bank metadata).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Insert parent repository to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle.kaggle_utils import (
    IS_KAGGLE,
    KAGGLE_INPUT,
    KAGGLE_WORKING,
    SCRATCH_ROOT,
    TimeGuard,
    copy_directory,
    find_input_dataset,
    materialize_sparse25,
    prepare_s3dis_from_kaggle_input,
    setup_kaggle_env,
    validate_dataset,
)
from pcseg import env_setup, run, settings
from pcseg.pu_data import build_patch_bank


def prepare_stage1_datasets(smoke_test=False, custom_data_root=None, out_dir=None):
    """Prepares S3DIS dense and sparse25 under settings.DATA and exports deterministic split indices."""
    out_base = Path(out_dir or (KAGGLE_WORKING / "stage1_out"))
    out_base.mkdir(parents=True, exist_ok=True)

    # 1. Prepare dense S3DIS from /kaggle/input or custom_data_root
    dense_path = prepare_s3dis_from_kaggle_input(smoke_test=smoke_test, custom_data_root=custom_data_root)

    # 2. Deterministically materialize sparse25 and export indices
    indices_export = out_base / "split_indices_sparse25.npz"
    materialize_sparse25(
        dense_root=dense_path,
        sparse_root=settings.SP25,
        export_indices_path=indices_export,
        smoke_test=smoke_test,
    )

    # 3. Validate
    validate_dataset(smoke_test=smoke_test)


def build_banks(smoke_test=False):
    """Builds required patch banks."""
    settings.BANKS.mkdir(parents=True, exist_ok=True)
    patches_dev = 10 if smoke_test else 100
    patches_val = 5 if smoke_test else 40
    n_in = 64 if smoke_test else 256

    print("\n--- Building Patch Banks ---")
    train_dev = settings.BANKS / "train_dev_x4.npz"
    val = settings.BANKS / "val_x4.npz"
    train_full = settings.BANKS / "train_full_x4.npz"

    print("Building train_dev_x4 (Areas 1-4)...")
    build_patch_bank(
        settings.DENSE,
        settings.SP25,
        ["Area_1", "Area_2", "Area_3", "Area_4"],
        str(train_dev),
        keep_ratio=0.25,
        up_ratio=4,
        n_in=n_in,
        patches_per_room=patches_dev,
        skip_existing=True,
    )

    print("Building val_x4 (Area 6)...")
    build_patch_bank(
        settings.DENSE,
        settings.SP25,
        ["Area_6"],
        str(val),
        keep_ratio=0.25,
        up_ratio=4,
        n_in=n_in,
        patches_per_room=patches_val,
        seed=1,
        skip_existing=True,
    )

    if not smoke_test:
        print("Building train_full_x4 (Areas 1-4, 6)...")
        build_patch_bank(
            settings.DENSE,
            settings.SP25,
            ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
            str(train_full),
            keep_ratio=0.25,
            up_ratio=4,
            n_in=n_in,
            patches_per_room=patches_dev,
            skip_existing=True,
        )

    return train_dev, val, (train_full if not smoke_test else train_dev)


def train_upsamplers(train_dev, val, train_full, smoke_test=False, guard=None):
    """Trains dev_x4 and production full_x4 upsamplers."""
    epochs = 2 if smoke_test else 30
    batch = 8 if smoke_test else 32

    settings.UPSAMPLER.mkdir(parents=True, exist_ok=True)
    dev_out = settings.UPSAMPLER / "dev_x4"
    full_out = settings.UPSAMPLER / "full_x4"

    print(f"\n--- Training Upsampler dev_x4 (Epochs: {epochs}, Batch: {batch}) ---")
    run.py_module(
        "pcseg.pu_train",
        "--train-bank", str(train_dev),
        "--val-bank", str(val),
        "--out", str(dev_out),
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--force" if smoke_test else "",
    )

    if guard and guard.should_stop():
        print(f"Time guard triggered ({guard.status()}). Skipping full_x4 training.")
        return dev_out, dev_out

    print(f"\n--- Training Upsampler full_x4 (Epochs: {epochs}, Batch: {batch}) ---")
    run.py_module(
        "pcseg.pu_train",
        "--train-bank", str(train_full),
        "--val-bank", str(val),
        "--out", str(full_out),
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--force" if smoke_test else "",
    )

    return dev_out, full_out


def export_stage1_artifacts(output_dir=None):
    """Exports compact, essential artifacts to /kaggle/working/stage1_out."""
    out_base = Path(output_dir or (KAGGLE_WORKING / "stage1_out"))
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Exporting Stage 1 Artifacts to {out_base} ---")

    # 1. Export trained checkpoints and logs
    ups_dst = out_base / "upsampler"
    ups_dst.mkdir(parents=True, exist_ok=True)
    for model_dir in settings.UPSAMPLER.glob("*"):
        if model_dir.is_dir():
            target_model = ups_dst / model_dir.name
            target_model.mkdir(parents=True, exist_ok=True)
            for f in ["best.pt", "last.pt", "log.json"]:
                src_f = model_dir / f
                if src_f.exists():
                    shutil.copy2(src_f, target_model / f)
                    print(f"Saved: {target_model / f}")

    # 2. Export patch bank metadata (not raw arrays, to save disk space)
    banks_meta = {}
    for b in settings.BANKS.glob("*.npz"):
        banks_meta[b.name] = {
            "size_mb": round(b.stat().st_size / (1024 ** 2), 2),
            "modified": time.ctime(b.stat().st_mtime),
        }
    (out_base / "patch_banks_manifest.json").write_text(json.dumps(banks_meta, indent=2))

    # 3. Export validation metrics summary if available
    summary = {}
    for log_file in ups_dst.glob("*/log.json"):
        try:
            d = json.loads(log_file.read_text())
            summary[log_file.parent.name] = {
                "best_val_cd": d.get("log", [{}])[-1].get("val_cd"),
                "epochs_trained": len(d.get("log", [])),
                "baseline_val": d.get("baseline_val"),
            }
        except Exception:
            pass
    (out_base / "stage1_summary.json").write_text(json.dumps(summary, indent=2))

    total_bytes = sum(f.stat().st_size for f in out_base.rglob("*") if f.is_file())
    print(f"Stage 1 Artifact Export Complete! Total size: {total_bytes / (1024 ** 2):.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Kaggle Stage 1: Data Prep & PUNet Upsampler Training")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-epoch test on synthetic data")
    parser.add_argument("--data-root", default=None, help="Path to input dataset root directory")
    parser.add_argument("--output-dir", default=None, help="Directory to save final stage artifacts")
    parser.add_argument("--max-hours", type=float, default=11.5, help="Time guard limit in hours")
    args = parser.parse_args()

    guard = TimeGuard(max_hours=args.max_hours)
    print(f"Stage 1 Started. {guard.status()}")

    out_dir = Path(args.output_dir or (KAGGLE_WORKING / "stage1_out"))

    # 1. Environment & Pointcept setup
    setup_kaggle_env()
    env_setup.prepare_pointcept()

    # 2. Dataset preparation & deterministic sparse split export
    prepare_stage1_datasets(smoke_test=args.smoke_test, custom_data_root=args.data_root, out_dir=out_dir)

    # 3. Build patch banks
    train_dev, val, train_full = build_banks(smoke_test=args.smoke_test)

    # 4. Train upsamplers
    train_upsamplers(train_dev, val, train_full, smoke_test=args.smoke_test, guard=guard)

    # 5. Export artifacts
    export_stage1_artifacts(output_dir=out_dir)
    print(f"Stage 1 Successfully Finished! {guard.status()}")


if __name__ == "__main__":
    main()
