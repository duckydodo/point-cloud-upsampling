"""Stage 2 Kaggle Runner: PTv3 Baselines Training and Area 5 Dense Evaluation.

Tasks:
1. Initialize environment and Pointcept commit 1342eda in /tmp/workspace.
2. Link or restore S3DIS datasets and Stage 1 upsampler artifacts (dev_x4, full_x4).
3. Train baseline PTv3 models:
   - Row A: A_dense (trained dense -> evaluated on dense Area 5)
   - Row B: B_sparse25 (trained sparse 25% -> evaluated on sparse 25% Area 5)
   - Row D: D_dense_aug (trained dense with RandomSparsify -> evaluated on sparse 25% Area 5)
4. Run comprehensive test-time Area 5 evaluations projected back to original dense ground-truth:
   - Row A: A_dense_on_dense
   - Row B: B_sparse25_on_sparse25
   - Row C: C_dense_model_on_sparse25 (Row A checkpoint on sparse 25% points)
   - Row E: E_dense_model_on_interp25x4 (Row A checkpoint on naive midpoint interpolation x4)
   - Row G: G_dense_model_on_up25x4 (Row A checkpoint on learned PUNet upsampling x4)
5. Save evaluation JSONs and model checkpoints to /kaggle/working/stage2_out/.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

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
from pcseg import env_setup, experiments, run, settings


def locate_stage1_artifacts(stage1_input_dir=None):
    """Finds trained upsampler weights and split indices from Stage 1."""
    candidate = None
    if stage1_input_dir and Path(stage1_input_dir).exists():
        candidate = Path(stage1_input_dir)
    else:
        found = find_input_dataset(["stage1", "upsampler", "stage-1"])
        if found:
            candidate = found

    if candidate:
        print(f"Found Stage 1 inputs at: {candidate}")
        # Copy upsampler weights to settings.UPSAMPLER
        src_ups = candidate / "upsampler" if (candidate / "upsampler").exists() else candidate
        if src_ups.exists():
            for m in src_ups.glob("*"):
                if m.is_dir():
                    dst_m = settings.UPSAMPLER / m.name
                    dst_m.mkdir(parents=True, exist_ok=True)
                    for f in ["best.pt", "last.pt", "log.json"]:
                        if (m / f).exists():
                            shutil.copy2(m / f, dst_m / f)
            print("Imported Stage 1 upsampler checkpoints.")

    # Check if full_x4 checkpoint is present
    full_ckpt = settings.UPSAMPLER / "full_x4" / "last.pt"
    if not full_ckpt.exists():
        dev_ckpt = settings.UPSAMPLER / "dev_x4" / "last.pt"
        if dev_ckpt.exists():
            print(f"full_x4 not found, falling back to dev_x4: {dev_ckpt}")
            (settings.UPSAMPLER / "full_x4").mkdir(parents=True, exist_ok=True)
            shutil.copy2(dev_ckpt, full_ckpt)
        else:
            print("Notice: No Stage 1 checkpoint found. If not running smoke-test, Row G will require upsampler.")

    return candidate


def prepare_stage2_datasets(smoke_test=False, custom_data_root=None, stage1_dir=None):
    """Ensures S3DIS dense and sparse25 datasets are available, reusing Stage 1 split indices if present."""
    dense_path = prepare_s3dis_from_kaggle_input(smoke_test=smoke_test, custom_data_root=custom_data_root)

    indices_file = None
    if stage1_dir:
        cand_indices = Path(stage1_dir) / "split_indices_sparse25.npz"
        if cand_indices.exists():
            indices_file = cand_indices
            print(f"Reusing Stage 1 sparse split indices from: {indices_file}")

    materialize_sparse25(
        dense_root=dense_path,
        sparse_root=settings.SP25,
        indices_file=indices_file,
        smoke_test=smoke_test,
    )

    validate_dataset(smoke_test=smoke_test)


def checkpoint_mirror(name, output_dir):
    """Copies current checkpoints for experiment `name` to output_dir immediately."""
    src = settings.EXP / name
    dst = Path(output_dir) / "exp" / name
    if src.exists():
        dst.mkdir(parents=True, exist_ok=True)
        for sub in ["model", "test_dense", "test_sparse25", "test_up25x4", "test_interp25x4"]:
            sub_src = src / sub
            if sub_src.exists():
                sub_dst = dst / sub
                sub_dst.mkdir(parents=True, exist_ok=True)
                for f in sub_src.glob("*"):
                    if f.is_file():
                        shutil.copy2(f, sub_dst / f.name)
        for meta in ["config.py", "train.log", "train_done.json"]:
            if (src / meta).exists():
                shutil.copy2(src / meta, dst / meta)
        print(f"[Checkpoint-Mirror] Mirrored {name} to {dst}")


def train_baselines(knobs, smoke_test=False, guard=None, output_dir=None):
    """Trains Row A, Row B, and Row D baselines."""
    D, S25 = str(settings.DENSE), str(settings.SP25)
    cfgs = experiments.make_all(knobs)

    rows = ["A_dense", "B_sparse25", "D_dense_aug"]
    for row in rows:
        if guard and guard.should_stop():
            print(f"Time guard triggered ({guard.status()}). Stopping before {row}.")
            break

        print(f"\n--- Training Baseline: {row} ---")
        run.train(row, cfgs[row])
        if output_dir:
            checkpoint_mirror(row, output_dir)


def evaluate_baselines(output_dir=None):
    """Evaluates Rows A, B, C, E, G on Area 5 and saves to settings.RESULTS."""
    D, S25 = str(settings.DENSE), str(settings.SP25)
    INT, UP = str(settings.INTERP25X4), str(settings.UP25X4)
    settings.RESULTS.mkdir(parents=True, exist_ok=True)

    print("\n--- Running Area 5 Evaluations ---")

    # 1. Row A: A_dense on dense Area 5
    if (settings.EXP / "A_dense" / "model" / "model_last.pth").exists():
        print("Evaluating Row A: A_dense on dense...")
        run.evaluate_run("A_dense", "dense", D, "A_dense_on_dense")

        # 2. Row C: A_dense on sparse 25% Area 5
        print("Evaluating Row C: A_dense model on sparse25...")
        run.evaluate_run("A_dense", "sparse25", S25, "C_dense_model_on_sparse25")

        # 3. Prepare upsampled Area 5 scenes for Rows E and G
        print("Preparing Area 5 naive interpolation (Row E)...")
        run.py_module("pcseg.pu_infer", "--method", "interp", "--sparse-root", S25, "--dense-root", D, "--dst", INT, "--areas", "Area_5")

        ckpt_up = settings.UPSAMPLER / "full_x4" / "last.pt"
        if not ckpt_up.exists():
            ckpt_up = settings.UPSAMPLER / "dev_x4" / "last.pt"

        if ckpt_up.exists():
            print(f"Preparing Area 5 learned upsampler inference (Row G) using {ckpt_up}...")
            run.py_module("pcseg.pu_infer", "--method", "model", "--ckpt", str(ckpt_up), "--sparse-root", S25, "--dense-root", D, "--dst", UP, "--areas", "Area_5")

            print("Evaluating Row G: A_dense model on learned upsampling (x4)...")
            run.evaluate_run("A_dense", "up25x4", UP, "G_dense_model_on_up25x4")
        else:
            print("Notice: Upsampler checkpoint missing; skipping Row G evaluation.")

        print("Evaluating Row E: A_dense model on naive interpolation (x4)...")
        run.evaluate_run("A_dense", "interp25x4", INT, "E_dense_model_on_interp25x4")

    # 4. Row B: B_sparse25 on sparse 25% Area 5
    if (settings.EXP / "B_sparse25" / "model" / "model_last.pth").exists():
        print("Evaluating Row B: B_sparse25 on sparse25...")
        run.evaluate_run("B_sparse25", "sparse25", S25, "B_sparse25_on_sparse25")

    # 5. Row D: D_dense_aug on sparse 25% Area 5
    if (settings.EXP / "D_dense_aug" / "model" / "model_last.pth").exists():
        print("Evaluating Row D: D_dense_aug on sparse25...")
        run.evaluate_run("D_dense_aug", "sparse25", S25, "D_dense_aug_on_sparse25")


def export_stage2_artifacts(output_dir=None):
    """Exports checkpoints and evaluation results to /kaggle/working/stage2_out."""
    out_base = Path(output_dir or (KAGGLE_WORKING / "stage2_out"))
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Exporting Stage 2 Artifacts to {out_base} ---")

    # 1. Export results JSONs
    dst_results = out_base / "results"
    dst_results.mkdir(parents=True, exist_ok=True)
    for f in settings.RESULTS.glob("*.json"):
        shutil.copy2(f, dst_results / f.name)
        print(f"Saved metric JSON: {f.name}")

    # 2. Export PTv3 experiment checkpoints and logs
    dst_exp = out_base / "exp"
    for exp_dir in settings.EXP.glob("*"):
        if exp_dir.is_dir():
            target_exp = dst_exp / exp_dir.name
            target_exp.mkdir(parents=True, exist_ok=True)
            for f in ["config.py", "train.log", "train_done.json"]:
                if (exp_dir / f).exists():
                    shutil.copy2(exp_dir / f, target_exp / f)
            # Model weights
            model_src = exp_dir / "model"
            if model_src.exists():
                (target_exp / "model").mkdir(parents=True, exist_ok=True)
                for w in model_src.glob("*.pth"):
                    shutil.copy2(w, target_exp / "model" / w.name)
                    print(f"Saved checkpoint: {exp_dir.name}/model/{w.name}")

    total_bytes = sum(f.stat().st_size for f in out_base.rglob("*") if f.is_file())
    print(f"Stage 2 Artifact Export Complete! Total size: {total_bytes / (1024 ** 2):.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Kaggle Stage 2: PTv3 Baselines and Area 5 Evaluation")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch test on synthetic data")
    parser.add_argument("--stage1-input", default=None, help="Directory containing Stage 1 artifacts")
    parser.add_argument("--data-root", default=None, help="Path to input dataset root directory")
    parser.add_argument("--output-dir", default=None, help="Directory to save final stage artifacts")
    parser.add_argument("--max-hours", type=float, default=11.5, help="Time guard limit in hours")
    args = parser.parse_args()

    guard = TimeGuard(max_hours=args.max_hours)
    print(f"Stage 2 Started. {guard.status()}")

    # 1. Environment & Pointcept setup
    setup_kaggle_env(install_spconv=True, setup_pointcept=True)

    # 2. Setup knobs and dataset
    stage1_dir = locate_stage1_artifacts(stage1_input_dir=args.stage1_input)
    prepare_stage2_datasets(smoke_test=args.smoke_test, custom_data_root=args.data_root, stage1_dir=stage1_dir)

    knobs = settings.load_knobs()
    if args.smoke_test:
        knobs.epoch = 1
        knobs.eval_epoch = 1
        knobs.batch = 2
        knobs.point_max = 5000
        knobs.amp = False
    knobs.save()
    print("Knobs configured:", knobs)

    # 3. Train baselines
    out_dir = args.output_dir or (KAGGLE_WORKING / "stage2_out")
    train_baselines(knobs, smoke_test=args.smoke_test, guard=guard, output_dir=out_dir)

    # 4. Evaluate on Area 5
    evaluate_baselines(output_dir=out_dir)

    # 5. Final export
    export_stage2_artifacts(output_dir=out_dir)
    print(f"Stage 2 Successfully Finished! {guard.status()}")


if __name__ == "__main__":
    main()
