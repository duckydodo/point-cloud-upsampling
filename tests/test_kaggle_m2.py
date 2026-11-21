"""Unit tests for Milestone 2: Kaggle container constraints, recursive discovery, and disk safety."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from kaggle.kaggle_utils import (
    checkpoint_mirror,
    get_directory_size_mb,
    list_kaggle_inputs,
    locate_stage1_artifacts,
    locate_stage2_artifacts,
    log_disk_usage,
)
from pcseg import settings

try:
    import torch
    from pcseg.pu_core import safe_torch_load
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestKaggleMilestone2(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_m2_")
        self.temp_path = Path(self.temp_dir)
        self.mock_workspace = self.temp_path / "workspace"
        self.mock_workspace.mkdir(parents=True, exist_ok=True)

        self.orig_root = os.environ.get("PCSEG_ROOT")
        self.orig_data = os.environ.get("PCSEG_DATA")
        os.environ.pop("PCSEG_DATA", None)
        os.environ["PCSEG_ROOT"] = str(self.mock_workspace)
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

    def test_settings_refresh_paths(self):
        """Test that settings.refresh_paths dynamically updates all derived Path attributes."""
        new_dir = self.temp_path / "dynamic_workspace"
        os.environ["PCSEG_ROOT"] = str(new_dir)
        settings.refresh_paths()
        self.assertEqual(settings.ROOT, new_dir.resolve())
        self.assertEqual(settings.DATA, (new_dir / "data").resolve())
        self.assertEqual(settings.EXP, (new_dir / "exp").resolve())
        self.assertEqual(settings.UPSAMPLER, (new_dir / "upsampler").resolve())
        self.assertEqual(settings.RESULTS, (new_dir / "results").resolve())

    def test_safe_shm_workers(self):
        """Test safe_shm_workers capping logic."""
        # When requesting 8 workers, should return integer <= 8
        w = settings.safe_shm_workers(8)
        self.assertIsInstance(w, int)
        self.assertGreaterEqual(w, 0)
        self.assertLessEqual(w, 8)

        # In Kaggle environment, caps at <= 2
        os.environ["KAGGLE_KERNEL_RUN_TYPE"] = "Interactive"
        try:
            w_kaggle = settings.safe_shm_workers(8)
            self.assertLessEqual(w_kaggle, 2)
        finally:
            os.environ.pop("KAGGLE_KERNEL_RUN_TYPE", None)

    def test_checkpoint_mirror_filters_bulky_checkpoints(self):
        """Test checkpoint_mirror copies model_best/last.pth but excludes bulky epoch_*.pth."""
        exp_name = "test_exp"
        exp_src = settings.EXP / exp_name
        model_dir = exp_src / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy weights
        (model_dir / "model_best.pth").write_text("best")
        (model_dir / "model_last.pth").write_text("last")
        (model_dir / "epoch_10.pth").write_text("epoch10")
        (model_dir / "epoch_20.pth").write_text("epoch20")
        (exp_src / "config.py").write_text("# config")
        (exp_src / "train.log").write_text("log")

        out_dir = self.temp_path / "kaggle_working"
        checkpoint_mirror(exp_name, out_dir)

        dst_model = out_dir / "exp" / exp_name / "model"
        self.assertTrue((dst_model / "model_best.pth").exists())
        self.assertTrue((dst_model / "model_last.pth").exists())
        # Bulk epoch weights MUST be excluded
        self.assertFalse((dst_model / "epoch_10.pth").exists())
        self.assertFalse((dst_model / "epoch_20.pth").exists())

        self.assertTrue((out_dir / "exp" / exp_name / "config.py").exists())
        self.assertTrue((out_dir / "exp" / exp_name / "train.log").exists())

    def test_recursive_stage1_artifact_discovery(self):
        """Test locate_stage1_artifacts recursively finds best.pt and split_indices_sparse25.npz."""
        nested_stage1 = self.temp_path / "deep" / "nested" / "stage1_run"
        nested_ups = nested_stage1 / "upsampler" / "full_x4"
        nested_ups.mkdir(parents=True, exist_ok=True)
        (nested_ups / "best.pt").write_text("mock_best_pt")
        (nested_ups / "last.pt").write_text("mock_last_pt")
        (nested_stage1 / "split_indices_sparse25.npz").write_text("mock_npz")

        stage1_dir, found_indices = locate_stage1_artifacts(stage1_input_dir=str(self.temp_path / "deep"))
        self.assertIsNotNone(stage1_dir)
        self.assertIsNotNone(found_indices)
        self.assertTrue(found_indices.exists())
        self.assertTrue((settings.UPSAMPLER / "full_x4" / "best.pt").exists())
        self.assertTrue((settings.UPSAMPLER / "full_x4" / "last.pt").exists())

    def test_recursive_stage2_artifact_discovery(self):
        """Test locate_stage2_artifacts recursively finds baseline results and checkpoints."""
        nested_stage2 = self.temp_path / "deep" / "nested" / "stage2_run"
        res_dir = nested_stage2 / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "A_dense_on_dense.json").write_text(json.dumps({"mIoU": 0.5}))
        (res_dir / "B_sparse25_on_sparse25.json").write_text(json.dumps({"mIoU": 0.45}))

        exp_model = nested_stage2 / "exp" / "A_dense" / "model"
        exp_model.mkdir(parents=True, exist_ok=True)
        (exp_model / "model_best.pth").write_text("mock_weight")

        stage2_dir = locate_stage2_artifacts(stage2_input_dir=str(self.temp_path / "deep"))
        self.assertIsNotNone(stage2_dir)
        self.assertTrue((settings.RESULTS / "A_dense_on_dense.json").exists())
        self.assertTrue((settings.RESULTS / "B_sparse25_on_sparse25.json").exists())
        self.assertTrue((settings.EXP / "A_dense" / "model" / "model_best.pth").exists())

    def test_fail_fast_on_missing_artifacts(self):
        """Test locate_stage1_artifacts and locate_stage2_artifacts fail fast when required inputs are missing."""
        empty_dir = self.temp_path / "empty_input"
        empty_dir.mkdir(parents=True, exist_ok=True)

        with self.assertRaises(RuntimeError) as ctx1:
            locate_stage1_artifacts(stage1_input_dir=str(empty_dir), smoke_test=False)
        self.assertIn("ERROR: Stage 1 artifacts not found!", str(ctx1.exception))

        with self.assertRaises(RuntimeError) as ctx2:
            locate_stage2_artifacts(stage2_input_dir=str(empty_dir), smoke_test=False)
        self.assertIn("ERROR: Stage 2 artifacts not found!", str(ctx2.exception))


if __name__ == "__main__":
    unittest.main()
