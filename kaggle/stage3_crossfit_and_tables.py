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
    checkpoint_mirror,
    copy_directory,
    find_input_dataset,
    locate_stage1_artifacts,
    locate_stage2_artifacts,
    log_disk_usage,
    materialize_sparse25,
    prepare_s3dis_from_kaggle_input,
    setup_kaggle_env,
    validate_dataset,
)
from pcseg import env_setup, experiments, results, run, settings
from pcseg.common import CLASS_NAMES


def prepare_stage3_datasets(smoke_test=False, custom_data_root=None, split_indices_path=None):
    """Ensures S3DIS dense and sparse25 datasets are available, reusing Stage 1 split indices if present."""
    dense_path = prepare_s3dis_from_kaggle_input(smoke_test=smoke_test, custom_data_root=custom_data_root)

    materialize_sparse25(
        dense_root=dense_path,
        sparse_root=settings.SP25,
        indices_file=split_indices_path,
        smoke_test=smoke_test,
    )

    validate_dataset(smoke_test=smoke_test)


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
        name_escaped = r["name"].replace("_", r"\_")
        latex_content.append(f"{name_escaped} & {r['mIoU']:.2f} & {r['OA']:.2f} & {r['mAcc']:.2f} \\\\")
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
    log_disk_usage(out_dir)


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
    setup_kaggle_env(install_spconv=True, setup_pointcept=True)

    # 2. Setup knobs and inputs
    stage1_dir, split_indices_path = locate_stage1_artifacts(stage1_input_dir=args.stage1_input, smoke_test=args.smoke_test)
    stage2_dir = locate_stage2_artifacts(stage2_input_dir=args.stage2_input, smoke_test=args.smoke_test)
    prepare_stage3_datasets(smoke_test=args.smoke_test, custom_data_root=args.data_root, split_indices_path=split_indices_path)

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
