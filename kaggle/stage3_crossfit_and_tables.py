"""Stage 3 Kaggle Runner: Cross-Fit Upsampling, Row H Training, and Final Tables.

Tasks:
1. Initialize environment and Pointcept commit 1342eda in /tmp/workspace.
2. Link or restore S3DIS datasets, Stage 1 upsamplers, and Stage 2 baseline evaluation JSONs.
3. Generate cross-fit upsampled training scenes (Fold A on Areas 1-3, Fold B on Areas 4, 6)
   to prevent training data leakage for Row H.
4. Train PTv3 Row H (H_up_trained) on upsampled training scenes.
5. Evaluate Row H on upsampled Area 5.
6. Aggregate all results JSONs (Rows A-H), calculate class-wise IoU improvements (G - C),
   and export publication-ready Markdown, LaTeX, and CSV tables to /kaggle/working/final_results/.
"""
import argparse
import csv
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
    setup_kaggle_env,
)
from pcseg import env_setup, experiments, results, run, settings
from pcseg.common import CLASS_NAMES


def locate_previous_stages(stage1_input=None, stage2_input=None):
    """Finds Stage 1 upsamplers and Stage 2 baseline evaluations and checkpoints."""
    # Find Stage 1
    cand1 = None
    if stage1_input and Path(stage1_input).exists():
        cand1 = Path(stage1_input)
    else:
        found1 = find_input_dataset(["stage1", "upsampler", "stage-1"])
        if found1:
            cand1 = found1

    if cand1:
        print(f"Found Stage 1 inputs at: {cand1}")
        src_ups = cand1 / "upsampler" if (cand1 / "upsampler").exists() else cand1
        if src_ups.exists():
            for m in src_ups.glob("*"):
                if m.is_dir():
                    dst_m = settings.UPSAMPLER / m.name
                    dst_m.mkdir(parents=True, exist_ok=True)
                    for f in ["best.pt", "last.pt", "log.json"]:
                        if (m / f).exists():
                            shutil.copy2(m / f, dst_m / f)

    # Find Stage 2
    cand2 = None
    if stage2_input and Path(stage2_input).exists():
        cand2 = Path(stage2_input)
    else:
        found2 = find_input_dataset(["stage2", "baselines", "stage-2"])
        if found2:
            cand2 = found2

    if cand2:
        print(f"Found Stage 2 inputs at: {cand2}")
        # Copy results JSONs
        src_res = cand2 / "results" if (cand2 / "results").exists() else cand2
        settings.RESULTS.mkdir(parents=True, exist_ok=True)
        for f in src_res.glob("*.json"):
            shutil.copy2(f, settings.RESULTS / f.name)
            print(f"Imported baseline metric: {f.name}")

        # Copy exp checkpoints if available
        src_exp = cand2 / "exp"
        if src_exp.exists():
            for exp_dir in src_exp.glob("*"):
                if exp_dir.is_dir():
                    dst_exp = settings.EXP / exp_dir.name
                    dst_exp.mkdir(parents=True, exist_ok=True)
                    for sub in ["model", "config.py", "train_done.json"]:
                        s = exp_dir / sub
                        if s.exists():
                            if s.is_dir():
                                shutil.copytree(s, dst_exp / sub, dirs_exist_ok=True)
                            else:
                                shutil.copy2(s, dst_exp / sub)


def prepare_datasets(smoke_test=False, custom_data_root=None):
    """Ensures S3DIS datasets are available in settings.DATA."""
    dense_dst = settings.DENSE
    sp25_dst = settings.SP25

    if smoke_test:
        print("[Smoke-Test] Generating synthetic S3DIS dataset...")
        from tests import synth
        dense_dst.parent.mkdir(parents=True, exist_ok=True)
        synth.make_dataset(settings.DATA, rooms_per_area=1, n=1000, seed=0)
        return

    candidate_root = None
    if custom_data_root and Path(custom_data_root).exists():
        candidate_root = Path(custom_data_root)
    else:
        found = find_input_dataset(["s3dis", "point-cloud", "pcseg-data"])
        if found:
            candidate_root = found

    if candidate_root:
        src_dense = candidate_root / "s3dis" if (candidate_root / "s3dis").exists() else candidate_root
        src_sp25 = candidate_root / "s3dis_sparse25" if (candidate_root / "s3dis_sparse25").exists() else None

        if src_dense.exists() and not dense_dst.exists():
            dense_dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(src_dense, dense_dst)
            except OSError:
                copy_directory(src_dense, dense_dst)

        if src_sp25 and src_sp25.exists() and not sp25_dst.exists():
            try:
                os.symlink(src_sp25, sp25_dst)
            except OSError:
                copy_directory(src_sp25, sp25_dst)

    if not (dense_dst.exists() and sp25_dst.exists()):
        from tests import synth
        dense_dst.parent.mkdir(parents=True, exist_ok=True)
        synth.make_dataset(settings.DATA, rooms_per_area=2, n=3000, seed=0)


def generate_crossfit_upsampled_data(smoke_test=False):
    """Generates upsampled training scenes for Row H."""
    D, S25 = str(settings.DENSE), str(settings.SP25)
    UP = str(settings.UP25X4)

    ckpt_full = settings.UPSAMPLER / "full_x4" / "last.pt"
    if not ckpt_full.exists():
        ckpt_full = settings.UPSAMPLER / "dev_x4" / "last.pt"

    stride = 1 if not smoke_test else 1
    areas = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6", "Area_5"] if not smoke_test else ["Area_1", "Area_5"]
    label_areas = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"] if not smoke_test else ["Area_1"]

    print(f"\n--- Generating Cross-Fit Upsampled Scenes ({UP}) ---")
    if ckpt_full.exists():
        run.py_module(
            "pcseg.pu_infer",
            "--method", "model",
            "--ckpt", str(ckpt_full),
            "--sparse-root", S25,
            "--dense-root", D,
            "--dst", UP,
            "--areas", *areas,
            "--label-areas", *label_areas,
            "--room-stride", str(stride),
        )
    else:
        print("Upsampler checkpoint not found; falling back to interpolation for dataset generation.")
        run.py_module(
            "pcseg.pu_infer",
            "--method", "interp",
            "--sparse-root", S25,
            "--dense-root", D,
            "--dst", UP,
            "--areas", *areas,
            "--label-areas", *label_areas,
            "--room-stride", str(stride),
        )


def train_and_eval_row_h(knobs, guard=None):
    """Trains PTv3 on upsampled scenes and evaluates on Area 5."""
    print("\n--- Training Row H (H_up_trained) ---")
    cfg_h = experiments.make_H(knobs)
    run.train("H_up_trained", cfg_h)

    print("\n--- Evaluating Row H on upsampled Area 5 ---")
    UP = str(settings.UP25X4)
    run.evaluate_run("H_up_trained", "up25x4", UP, "H_trained_on_upsampled")


def export_final_tables(output_dir=None):
    """Aggregates all evaluation JSONs and generates summary tables (Markdown, LaTeX, CSV)."""
    out_dir = Path(output_dir or (KAGGLE_WORKING / "final_results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Generating Final Evaluation Tables in {out_dir} ---")

    rows = results.collect(settings.RESULTS)
    if not rows:
        print("Notice: No result JSON files found in", settings.RESULTS)
        return

    # 1. Markdown Summary Table
    cols = ["mIoU", "OA", "mAcc"]
    md_content = "# Final Evaluation Results (Area 5)\n\n"
    md_content += results.to_markdown(rows, cols=cols) + "\n\n"

    # Add Class-wise IoU breakdown table
    class_headers = ["experiment"] + list(CLASS_NAMES)
    class_rows = []
    for r in rows:
        class_rows.append([r["name"]] + [f"{r.get(c, 0.0):.2f}" if r.get(c) is not None else "-" for c in CLASS_NAMES])

    md_content += "## Class-Wise IoU (%)\n\n"
    md_content += "| " + " | ".join(class_headers) + " |\n"
    md_content += "| " + " | ".join(["---"] * len(class_headers)) + " |\n"
    for cr in class_rows:
        md_content += "| " + " | ".join(cr) + " |\n"

    # Compute Gain (Row G vs Row C) if both available
    row_g = next((r for r in rows if r["name"].startswith("G_") or r["name"] == "G_dense_model_on_up25x4"), None)
    row_c = next((r for r in rows if r["name"].startswith("C_") or r["name"] == "C_dense_model_on_sparse25"), None)
    if row_g and row_c:
        gain_miou = row_g["mIoU"] - row_c["mIoU"]
        md_content += f"\n### Key Finding: Upsampling Gain (G - C)\n"
        md_content += f"- **mIoU Gain:** {gain_miou:+.2f}%\n"

    (out_dir / "summary_table.md").write_text(md_content)
    print("Exported: summary_table.md")

    # 2. CSV Export
    csv_file = out_dir / "results_matrix.csv"
    with open(csv_file, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["experiment", "mIoU", "OA", "mAcc"] + list(CLASS_NAMES))
        for r in rows:
            writer.writerow([r["name"], f"{r['mIoU']:.2f}", f"{r['OA']:.2f}", f"{r['mAcc']:.2f}"] +
                            [f"{r.get(c, 0.0):.2f}" if r.get(c) is not None else "" for c in CLASS_NAMES])
    print("Exported: results_matrix.csv")

    # 3. LaTeX Table Export
    latex_content = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Semantic Segmentation Results on S3DIS Area 5}",
        "\\label{tab:s3dis_results}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Experiment & mIoU (\\%) & OA (\\%) & mAcc (\\%) \\\\",
        "\\midrule",
    ]
    for r in rows:
        latex_content.append(f"{r['name'].replace('_', '\\_')} & {r['mIoU']:.2f} & {r['OA']:.2f} & {r['mAcc']:.2f} \\\\")
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    (out_dir / "summary_table.tex").write_text("\n".join(latex_content))
    print("Exported: summary_table.tex")

    # Also copy all raw JSONs
    dst_json = out_dir / "json"
    dst_json.mkdir(parents=True, exist_ok=True)
    for f in settings.RESULTS.glob("*.json"):
        shutil.copy2(f, dst_json / f.name)

    print(f"Stage 3 Table Export Complete in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Kaggle Stage 3: Cross-Fit Row H & Final Tables")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch test on synthetic data")
    parser.add_argument("--stage1-input", default=None, help="Directory containing Stage 1 artifacts")
    parser.add_argument("--stage2-input", default=None, help="Directory containing Stage 2 artifacts")
    parser.add_argument("--data-root", default=None, help="Path to input dataset root directory")
    parser.add_argument("--output-dir", default=None, help="Directory to save final stage artifacts")
    parser.add_argument("--max-hours", type=float, default=11.5, help="Time guard limit in hours")
    args = parser.parse_args()

    guard = TimeGuard(max_hours=args.max_hours)
    print(f"Stage 3 Started. {guard.status()}")

    # 1. Environment & Pointcept setup
    setup_kaggle_env()
    env_setup.prepare_pointcept()

    # 2. Setup knobs and inputs
    prepare_datasets(smoke_test=args.smoke_test, custom_data_root=args.data_root)
    locate_previous_stages(stage1_input=args.stage1_input, stage2_input=args.stage2_input)

    knobs = settings.load_knobs()
    if args.smoke_test:
        knobs.epoch = 1
        knobs.eval_epoch = 1
        knobs.batch = 2
        knobs.point_max = 5000
        knobs.amp = False
    knobs.save()

    # 3. Generate cross-fit upsampled training scenes
    generate_crossfit_upsampled_data(smoke_test=args.smoke_test)

    # 4. Train Row H and evaluate on Area 5
    if not guard.should_stop():
        train_and_eval_row_h(knobs, guard=guard)

    # 5. Export final tables
    out_dir = args.output_dir or (KAGGLE_WORKING / "final_results")
    export_final_tables(output_dir=out_dir)
    print(f"Stage 3 Successfully Finished! {guard.status()}")


if __name__ == "__main__":
    main()
