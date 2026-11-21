"""Unit tests for Kaggle S3DIS data ingestion, layout normalization, and deterministic sparse25 materialization."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from kaggle.kaggle_utils import (
    _validate_s3dis_dense,
    materialize_sparse25,
    prepare_s3dis_from_kaggle_input,
    validate_dataset,
)
from pcseg import settings
from tests import synth


class TestKaggleDataIngestion(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_kaggle_data_")
        self.temp_path = Path(self.temp_dir)
        # Mock settings.ROOT, settings.DATA, settings.DENSE, settings.SP25
        self.mock_data = self.temp_path / "mock_workspace" / "data"
        self.mock_data.mkdir(parents=True, exist_ok=True)
        self.orig_data = settings.DATA
        self.orig_dense = settings.DENSE
        self.orig_sp25 = settings.SP25

        settings.DATA = self.mock_data
        settings.DENSE = self.mock_data / "s3dis"
        settings.SP25 = self.mock_data / "s3dis_sparse25"

    def tearDown(self):
        settings.DATA = self.orig_data
        settings.DENSE = self.orig_dense
        settings.SP25 = self.orig_sp25
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_synth_source(self, src_dir, include_office19=True):
        src_path = Path(src_dir)
        synth.make_dataset(src_path, rooms_per_area=2, n=500, seed=42)
        if include_office19:
            # Add an Area_5/office_19
            off19 = src_path / "s3dis" / "Area_5" / "office_19"
            off19.mkdir(parents=True, exist_ok=True)
            xyz, col, seg = synth.make_room(np.random.default_rng(123), n=600)
            np.save(off19 / "coord.npy", xyz)
            np.save(off19 / "color.npy", col)
            np.save(off19 / "segment.npy", seg)
        return src_path / "s3dis"

    def test_layout_a_nested_s3dis(self):
        """Test Layout (a): custom_data_root contains s3dis/Area_1..Area_6."""
        source_root = self.temp_path / "source_a"
        self._create_synth_source(source_root)

        res = prepare_s3dis_from_kaggle_input(smoke_test=False, custom_data_root=str(source_root))
        self.assertTrue(res.is_dir())
        self.assertTrue((settings.DENSE / "Area_1").is_dir())
        self.assertTrue((settings.DENSE / "Area_5" / "office_19" / "coord.npy").exists())

    def test_layout_b_direct_area_folders(self):
        """Test Layout (b): custom_data_root contains Area_1..Area_6 directly."""
        source_root = self.temp_path / "source_b"
        dense_src = self._create_synth_source(source_root)
        # dense_src has Area_1..Area_6 directly inside it

        res = prepare_s3dis_from_kaggle_input(smoke_test=False, custom_data_root=str(dense_src))
        self.assertTrue(res.is_dir())
        self.assertTrue((settings.DENSE / "Area_1").is_dir())
        self.assertTrue((settings.DENSE / "Area_5" / "office_19" / "coord.npy").exists())

    def test_layout_c_tar_archive(self):
        """Test Layout (c): custom_data_root contains s3dis_dense.tar."""
        source_root = self.temp_path / "source_c"
        dense_src = self._create_synth_source(source_root)

        # Create tar archive in an input folder
        input_archive_dir = self.temp_path / "archive_input"
        input_archive_dir.mkdir(parents=True, exist_ok=True)
        tar_path = input_archive_dir / "s3dis_dense.tar"
        subprocess.run(["tar", "-cf", str(tar_path), "-C", str(dense_src.parent), "s3dis"], check=True)

        res = prepare_s3dis_from_kaggle_input(smoke_test=False, custom_data_root=str(input_archive_dir))
        self.assertTrue(res.is_dir())
        self.assertTrue((settings.DENSE / "Area_1").is_dir())
        self.assertTrue((settings.DENSE / "Area_5" / "office_19" / "coord.npy").exists())

    def test_deterministic_sparse25_materialization(self):
        """Test deterministic sparse25 subsampling, export, and index reuse."""
        source_root = self.temp_path / "source_det"
        self._create_synth_source(source_root)
        prepare_s3dis_from_kaggle_input(smoke_test=False, custom_data_root=str(source_root))

        # 1. First run: materialize and export indices
        export_indices = self.temp_path / "stage1_out" / "split_indices_sparse25.npz"
        sparse_out1 = self.temp_path / "sparse_run1"
        materialize_sparse25(
            dense_root=settings.DENSE,
            sparse_root=sparse_out1,
            export_indices_path=export_indices,
            keep_ratio=0.25,
            seed=42,
        )

        self.assertTrue(export_indices.exists())
        self.assertTrue((export_indices.parent / "split_metadata.json").exists())

        # Verify metadata contents
        meta = json.loads((export_indices.parent / "split_metadata.json").read_text())
        self.assertEqual(meta["keep_ratio"], 0.25)
        self.assertEqual(meta["seed"], 42)
        self.assertGreater(meta["total_sparse_pts"], 0)

        # 2. Second run: fresh sparse directory without indices file (pure deterministic seed formula)
        sparse_out2 = self.temp_path / "sparse_run2"
        materialize_sparse25(
            dense_root=settings.DENSE,
            sparse_root=sparse_out2,
            keep_ratio=0.25,
            seed=42,
        )

        # 3. Third run: fresh sparse directory reusing indices_file
        sparse_out3 = self.temp_path / "sparse_run3"
        materialize_sparse25(
            dense_root=settings.DENSE,
            sparse_root=sparse_out3,
            indices_file=export_indices,
        )

        # Verify all rooms match across run 1, run 2, and run 3
        rooms = list(settings.DENSE.glob("Area_*/*"))
        self.assertGreater(len(rooms), 0)

        for room in rooms:
            if not room.is_dir():
                continue
            a_name = room.parent.name
            r_name = room.name

            dense_coord = np.load(room / "coord.npy")
            sp1_coord = np.load(sparse_out1 / a_name / r_name / "coord.npy")
            sp2_coord = np.load(sparse_out2 / a_name / r_name / "coord.npy")
            sp3_coord = np.load(sparse_out3 / a_name / r_name / "coord.npy")

            # Exact array equality
            np.testing.assert_array_equal(sp1_coord, sp2_coord)
            np.testing.assert_array_equal(sp1_coord, sp3_coord)

            # Ratio is close to 0.25
            ratio = len(sp1_coord) / len(dense_coord)
            self.assertAlmostEqual(ratio, 0.25, delta=0.05)

            # Exact subset property (nearest neighbor distance == 0)
            tree = cKDTree(dense_coord)
            dists, _ = tree.query(sp1_coord, k=1)
            self.assertEqual(dists.max(), 0.0)

    def test_validate_dataset_smoke_test(self):
        """Test validate_dataset in smoke_test mode."""
        source_root = self.temp_path / "source_val"
        self._create_synth_source(source_root)
        prepare_s3dis_from_kaggle_input(smoke_test=False, custom_data_root=str(source_root))
        materialize_sparse25(dense_root=settings.DENSE, sparse_root=settings.SP25, seed=42)

        # Validate dataset
        ok = validate_dataset(smoke_test=True)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
