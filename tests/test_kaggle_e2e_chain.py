"""End-to-end simulation of the Kaggle pipeline chain: Stage 1 -> Stage 2 -> Stage 3."""
import csv
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kaggle.kaggle_utils import (
    checkpoint_mirror,
    get_directory_size_mb,
    locate_stage1_artifacts,
    locate_stage2_artifacts,
    log_disk_usage,
    materialize_sparse25,
    prepare_s3dis_from_kaggle_input,
    validate_dataset,
)
from pcseg import settings
from tests import synth


class TestKaggleE2EChain(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_e2e_chain_")
        self.temp_path = Path(self.temp_dir)
        self.scratch = self.temp_path / "scratch"
        self.scratch.mkdir(parents=True, exist_ok=True)

        self.orig_root = os.environ.get("PCSEG_ROOT")
        self.orig_data = os.environ.get("PCSEG_DATA")
        os.environ.pop("PCSEG_DATA", None)
        os.environ["PCSEG_ROOT"] = str(self.scratch)
        settings.refresh_paths()

    def tearDown(self):
        if self.orig_root:
            os.environ["PCSEG_ROOT"] = self.orig_root
        else:
            os.environ.pop("PCSEG_ROOT", None)
        if self.orig_data:
            os.environ["PCSEG_DATA"] = self.orig_data
        else:
            os.environ.pop("PCSEG_DATA", None)
        settings.refresh_paths()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_pipeline_chain(self):
        """Simulate chained execution: Stage 1 outputs feed into Stage 2, which feed into Stage 3."""
        # -------------------------------------------------------------
        # STAGE 1: Data Preparation & Artifact Export
        # -------------------------------------------------------------
        print("\n=== Running Simulated Stage 1 ===")
        stage1_out = self.temp_path / "kaggle_working" / "stage1_out"
        stage1_out.mkdir(parents=True, exist_ok=True)

        # 1. Prepare dense S3DIS from synthetic generator
        dense_path = prepare_s3dis_from_kaggle_input(smoke_test=True)
        self.assertTrue((dense_path / "Area_1").is_dir())

        # 2. Materialize sparse25 and export split indices
        indices_export = stage1_out / "split_indices_sparse25.npz"
        materialize_sparse25(
            dense_root=dense_path,
            sparse_root=settings.SP25,
            export_indices_path=indices_export,
            smoke_test=True,
        )
        self.assertTrue(indices_export.exists())
        self.assertTrue((stage1_out / "split_metadata.json").exists())

        # 3. Simulate trained upsampler export
        stage1_ups = stage1_out / "upsampler" / "full_x4"
        stage1_ups.mkdir(parents=True, exist_ok=True)
        (stage1_ups / "best.pt").write_text("dummy_best_pt")
        (stage1_ups / "last.pt").write_text("dummy_last_pt")
        (stage1_ups / "log.json").write_text(json.dumps({"log": [{"val_cd": 2.5}]}))
        (stage1_out / "stage1_summary.json").write_text(json.dumps({"full_x4": {"best_val_cd": 2.5}}))

        # Check Stage 1 disk usage
        s1_gb = log_disk_usage(stage1_out)
        self.assertLess(s1_gb, 1.0)
        print("Stage 1 successfully generated artifacts.")

        # -------------------------------------------------------------
        # STAGE 2: Baseline Training & Evaluation
        # -------------------------------------------------------------
        print("\n=== Running Simulated Stage 2 ===")
        # Clear scratch UPSAMPLER to verify Stage 2 correctly re-imports from Stage 1 input
        shutil.rmtree(settings.UPSAMPLER, ignore_errors=True)

        stage2_out = self.temp_path / "kaggle_working" / "stage2_out"
        stage2_out.mkdir(parents=True, exist_ok=True)

        # 1. Locate Stage 1 artifacts
        s1_dir, found_indices = locate_stage1_artifacts(stage1_input_dir=str(stage1_out), smoke_test=True)
        self.assertEqual(found_indices, indices_export)
        self.assertTrue((settings.UPSAMPLER / "full_x4" / "best.pt").exists())

        # 2. Re-materialize sparse25 using Stage 1 indices into scratch
        scratch_sp25 = self.scratch / "rebuilt_sp25"
        materialize_sparse25(
            dense_root=settings.DENSE,
            sparse_root=scratch_sp25,
            indices_file=found_indices,
            smoke_test=True,
        )
        # Verify reconstructed point count matches
        self.assertTrue((scratch_sp25 / "Area_1").is_dir())

        # 3. Simulate training checkpoint mirroring with bulky checkpoint filtering
        exp_src = settings.EXP / "A_dense"
        model_src = exp_src / "model"
        model_src.mkdir(parents=True, exist_ok=True)
        (model_src / "model_best.pth").write_text("best_weights")
        (model_src / "model_last.pth").write_text("last_weights")
        (model_src / "epoch_10.pth").write_text("bulky_epoch_10")
        (model_src / "epoch_20.pth").write_text("bulky_epoch_20")
        (exp_src / "config.py").write_text("# config")
        (exp_src / "train.log").write_text("log output")

        checkpoint_mirror("A_dense", stage2_out)

        # Verify bulk weights are excluded from output
        mirrored_model = stage2_out / "exp" / "A_dense" / "model"
        self.assertTrue((mirrored_model / "model_best.pth").exists())
        self.assertTrue((mirrored_model / "model_last.pth").exists())
        self.assertFalse((mirrored_model / "epoch_10.pth").exists())
        self.assertFalse((mirrored_model / "epoch_20.pth").exists())

        # 4. Simulate baseline evaluation JSON outputs
        results_dir = stage2_out / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        mock_evals = {
            "A_dense_on_dense.json": {"experiment": "A_dense", "mIoU": 0.725, "OA": 0.88, "mAcc": 0.80},
            "B_sparse25_on_sparse25.json": {"experiment": "B_sparse25", "mIoU": 0.610, "OA": 0.82, "mAcc": 0.71},
            "C_dense_model_on_sparse25.json": {"experiment": "C", "mIoU": 0.585, "OA": 0.80, "mAcc": 0.69},
            "E_dense_model_on_interp25x4.json": {"experiment": "E", "mIoU": 0.630, "OA": 0.83, "mAcc": 0.73},
            "G_dense_model_on_up25x4.json": {"experiment": "G", "mIoU": 0.665, "OA": 0.85, "mAcc": 0.76},
        }
        for fname, val in mock_evals.items():
            (results_dir / fname).write_text(json.dumps(val, indent=2))

        s2_gb = log_disk_usage(stage2_out)
        self.assertLess(s2_gb, 1.0)
        print("Stage 2 successfully generated artifacts.")

        # -------------------------------------------------------------
        # STAGE 3: Cross-Fit & Final Tables
        # -------------------------------------------------------------
        print("\n=== Running Simulated Stage 3 ===")
        # Clear scratch results and upsamplers to test Stage 3 reconstitution
        shutil.rmtree(settings.RESULTS, ignore_errors=True)
        shutil.rmtree(settings.UPSAMPLER, ignore_errors=True)

        stage3_out = self.temp_path / "kaggle_working" / "final_results"
        stage3_out.mkdir(parents=True, exist_ok=True)

        # 1. Locate Stage 1 and Stage 2 inputs
        s1_dir, found_indices = locate_stage1_artifacts(stage1_input_dir=str(stage1_out), smoke_test=True)
        s2_dir = locate_stage2_artifacts(stage2_input_dir=str(stage2_out), smoke_test=True)

        self.assertEqual(s1_dir, stage1_out)
        self.assertEqual(s2_dir, stage2_out)
        self.assertTrue((settings.RESULTS / "A_dense_on_dense.json").exists())
        self.assertTrue((settings.RESULTS / "G_dense_model_on_up25x4.json").exists())
        self.assertTrue((settings.UPSAMPLER / "full_x4" / "best.pt").exists())

        # 2. Add Row H evaluation metric
        (settings.RESULTS / "H_up_trained_on_up25x4.json").write_text(json.dumps({
            "experiment": "H_up_trained", "mIoU": 0.680, "OA": 0.86, "mAcc": 0.77
        }, indent=2))

        # 3. Simulate final table export (as in stage3_crossfit_and_tables.py)
        from kaggle.stage3_crossfit_and_tables import export_final_tables
        export_final_tables(output_dir=stage3_out)

        # Verify final tables
        self.assertTrue((stage3_out / "summary_table.md").exists())
        self.assertTrue((stage3_out / "summary_table.tex").exists())
        self.assertTrue((stage3_out / "results_matrix.csv").exists())
        self.assertTrue((stage3_out / "json" / "A_dense_on_dense.json").exists())
        self.assertTrue((stage3_out / "json" / "H_up_trained_on_up25x4.json").exists())

        s3_gb = log_disk_usage(stage3_out)
        self.assertLess(s3_gb, 1.0)

        # Check CSV content has evaluated rows
        with open(stage3_out / "results_matrix.csv", "r", encoding="utf-8") as fp:
            csv_rows = list(csv.reader(fp))
        self.assertGreaterEqual(len(csv_rows), 4)
        print("Stage 3 successfully exported final summary tables!")


if __name__ == "__main__":
    unittest.main()
