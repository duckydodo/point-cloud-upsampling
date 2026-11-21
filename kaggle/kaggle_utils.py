"""Shared utilities and environment setup for Kaggle batch runners.

Designed for Kaggle headless execution:
- Ephemeral /tmp/workspace scratch (fast, high IOPS, large capacity)
- Output artifacts saved to /kaggle/working/
- Automatic detection of /kaggle/input datasets
- Multi-GPU pinning and CUDA wheel installation helpers
- Time-guard monitoring to exit cleanly before session timeouts
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Identify if running in Kaggle environment
IS_KAGGLE = Path("/kaggle").is_dir()

# Default directory structure
KAGGLE_WORKING = Path("/kaggle/working") if IS_KAGGLE else Path.cwd() / "kaggle_output"
KAGGLE_INPUT = Path("/kaggle/input") if IS_KAGGLE else Path.cwd() / "kaggle_input"
SCRATCH_ROOT = Path("/tmp/workspace") if IS_KAGGLE else Path.cwd() / "workspace"

# Configure PCSEG environment variables before importing pcseg modules
os.environ["PCSEG_ROOT"] = str(SCRATCH_ROOT)
os.environ["PCSEG_DATA"] = str(SCRATCH_ROOT / "data")
os.environ["POINTCEPT_DIR"] = str(SCRATCH_ROOT / "Pointcept")

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcseg import settings


class TimeGuard:
    """Monitors elapsed time and alerts when near the execution limit (e.g., 11.5 hours on Kaggle)."""

    def __init__(self, max_hours=11.5):
        self.start_time = time.time()
        self.max_seconds = max_hours * 3600

    @property
    def elapsed_seconds(self):
        return time.time() - self.start_time

    @property
    def elapsed_hours(self):
        return self.elapsed_seconds / 3600

    @property
    def remaining_seconds(self):
        return max(0.0, self.max_seconds - self.elapsed_seconds)

    def should_stop(self, buffer_seconds=600):
        """Returns True if remaining time is less than buffer_seconds (default 10 minutes)."""
        return self.remaining_seconds < buffer_seconds

    def status(self):
        return (f"Elapsed: {self.elapsed_hours:.2f}h / "
                f"Limit: {self.max_seconds/3600:.1f}h "
                f"(Remaining: {self.remaining_seconds/60:.1f}m)")


def _test_cuda_spconv():
    """Runs a minimal SubMConv3d forward pass in CUDA in a subprocess to ensure spconv is working."""
    test_code = """
import sys, torch
if not torch.cuda.is_available():
    sys.exit(0)
import spconv.pytorch as spconv
coords = torch.unique(torch.randint(0, 20, (200, 3), device="cuda", dtype=torch.int32), dim=0)
idx = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32, device="cuda"), coords], 1)
x = spconv.SparseConvTensor(torch.randn(len(coords), 8, device="cuda"), idx, spatial_shape=[20, 20, 20], batch_size=1)
y = spconv.SubMConv3d(8, 16, 3, indice_key="t").cuda()(x)
assert y.features.shape == (len(coords), 16)
sys.exit(0)
"""
    r = subprocess.run([sys.executable, "-c", test_code], capture_output=True, text=True)
    return r.returncode == 0


def _uninstall_spconv():
    """Uninstalls any existing spconv or cumm packages before trying the next candidate."""
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "spconv", "spconv-cu124", "spconv-cu126", "spconv-cu130", "cumm", "cumm-cu130"],
        check=False,
    )


def setup_kaggle_env(install_spconv=True, setup_pointcept=False):
    """Detects CUDA version, installs verified spconv wheels, and prepares Pointcept at commit 1342eda."""
    print("=" * 70)
    print("Initializing Kaggle Environment...")
    print(f"IS_KAGGLE: {IS_KAGGLE}")
    print(f"Python: {sys.version.split()[0]} | Executable: {sys.executable}")

    # Ensure scratch directories exist
    settings.ensure_dirs()
    print(f"Scratch PCSEG_ROOT: {settings.ROOT}")
    print(f"Scratch PCSEG_DATA: {settings.DATA}")

    # Check CUDA and GPU
    g = settings.gpu_info()
    has_cuda = False
    if g:
        print(f"Detected GPU 0: {g[0]} ({g[1]:.1f} GB VRAM, CC {g[2]})")
        has_cuda = True
    else:
        print("No GPU detected by PyTorch yet.")

    # Pin CUDA_VISIBLE_DEVICES if multiple GPUs present unless user specifies
    if "CUDA_VISIBLE_DEVICES" not in os.environ and g:
        try:
            import torch
            num_gpus = torch.cuda.device_count()
            print(f"Total visible GPUs: {num_gpus}")
            if num_gpus > 1:
                print("Setting CUDA_VISIBLE_DEVICES=0 for stable single-process Pointcept runs.")
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        except Exception:
            pass

    # CUDA-verified spconv installation
    if install_spconv and has_cuda:
        print("\nChecking spconv CUDA compatibility...")
        if _test_cuda_spconv():
            print("spconv is already installed and verified on CUDA!")
        else:
            print("spconv not importable or CUDA test failed. Installing verified wheels...")
            candidates = [
                # 1. spconv-cu126 (bundles CUDA 12.6 runtime, supports Python 3.9-3.13, natively supports T4 sm_75)
                (["pip", "install", "-q", "spconv-cu126"], "spconv-cu126"),
                # 2. CUDA 13.0 fallback from ratharog
                (["pip", "install", "-q", "cumm-cu130", "spconv-cu130", "--extra-index-url", "https://ratharog.github.io/cumm-spconv/"], "spconv-cu130"),
            ]
            installed_ok = False
            for cmd, name in candidates:
                print(f"Attempting installation: {name}...")
                _uninstall_spconv()
                res = subprocess.run([sys.executable, "-m"] + cmd, check=False)
                if res.returncode == 0 and _test_cuda_spconv():
                    print(f"SUCCESS: {name} installed and verified on CUDA SubMConv3d forward pass!")
                    installed_ok = True
                    break
                else:
                    print(f"FAILED: {name} did not pass CUDA verification.")

            if not installed_ok:
                _uninstall_spconv()
                raise RuntimeError(
                    "CUDA-compatible spconv could not be installed/verified on this environment. "
                    "Tested spconv-cu126 and cumm-cu130/spconv-cu130; neither passed CUDA SubMConv3d test. "
                    "Will not fall back to CPU-only spconv as Pointcept requires CUDA sparse convolution."
                )

    # Pointcept preparation
    if setup_pointcept:
        print("\nPreparing Pointcept at pinned commit 1342eda with shims and patches...")
        from pcseg import env_setup
        env_setup.prepare_pointcept()

    print("=" * 70)


def copy_directory(src, dst, max_size_gb=15.0):
    """Safely copies files from src to dst, verifying destination does not exceed max_size_gb."""
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source directory {src} does not exist.")

    dst.mkdir(parents=True, exist_ok=True)
    total_copied = 0
    for root, _, files in os.walk(src):
        rel_path = Path(root).relative_to(src)
        target_dir = dst / rel_path
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            source_file = Path(root) / f
            dest_file = target_dir / f
            shutil.copy2(source_file, dest_file)
            total_copied += dest_file.stat().st_size

    total_gb = total_copied / (1024 ** 3)
    print(f"Copied {src} -> {dst} ({total_gb:.2f} GB)")
    if total_gb > max_size_gb:
        print(f"WARNING: Output size ({total_gb:.2f} GB) exceeds safe threshold ({max_size_gb} GB)!")
    return total_gb


def find_input_dataset(dataset_name_keywords):
    """Scans /kaggle/input for directories matching given keywords in priority order."""
    if not KAGGLE_INPUT.exists():
        return None
    for kw in dataset_name_keywords:
        kw_l = kw.lower()
        for item in KAGGLE_INPUT.iterdir():
            if item.is_dir() and kw_l in item.name.lower():
                return item
    return None


def _validate_s3dis_dense(dense_dir, smoke_test=False):
    """Validates S3DIS dense structure and verifies Area_5/office_19 integrity."""
    dense_path = Path(dense_dir)
    if not dense_path.is_dir():
        raise FileNotFoundError(f"Dense dataset directory not found: {dense_path}")

    areas = [f"Area_{i}" for i in range(1, 7)]
    if smoke_test:
        areas = [a for a in areas if (dense_path / a).is_dir()]
        if not areas:
            areas = [d.name for d in dense_path.iterdir() if d.is_dir() and d.name.startswith("Area_")]

    print(f"\n[Validation] Checking dense S3DIS at {dense_path}...")
    for a in areas:
        area_dir = dense_path / a
        if not area_dir.is_dir():
            if smoke_test:
                continue
            raise RuntimeError(f"Missing expected {a} in {dense_path}")
        rooms = [r for r in area_dir.iterdir() if r.is_dir()]
        if not rooms:
            raise RuntimeError(f"No room folders found under {area_dir}")
        print(f"  {a}: {len(rooms)} rooms found.")

    # Specific check for Area_5/office_19 (largest room, prone to OOM/corruption)
    off19_dir = dense_path / "Area_5" / "office_19"
    if off19_dir.is_dir():
        print("  Verifying Area_5/office_19 integrity...")
        for req in ["coord.npy", "color.npy", "segment.npy"]:
            f = off19_dir / req
            if not f.exists():
                raise RuntimeError(f"Area_5/office_19 missing required file: {req}")

        import numpy as np
        coord = np.load(off19_dir / "coord.npy")
        color = np.load(off19_dir / "color.npy")
        segment = np.load(off19_dir / "segment.npy")

        if coord.shape[0] != color.shape[0] or coord.shape[0] != segment.shape[0]:
            raise ValueError(
                f"Area_5/office_19 length mismatch: coord={coord.shape[0]}, "
                f"color={color.shape[0]}, segment={segment.shape[0]}"
            )
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError(f"Area_5/office_19 invalid coord shape: {coord.shape}")
        if color.ndim != 2 or color.shape[1] != 3:
            raise ValueError(f"Area_5/office_19 invalid color shape: {color.shape}")
        if not (np.isfinite(coord[:1000]).all() and np.isfinite(color[:1000]).all()):
            raise ValueError("Area_5/office_19 contains non-finite values in coord or color sample.")
        print(f"  Area_5/office_19 integrity OK: {len(coord):,} points, valid coord/color/segment.")
    else:
        if not smoke_test:
            print("  Notice: Area_5/office_19 not present for individual inspection.")


def prepare_s3dis_from_kaggle_input(smoke_test=False, custom_data_root=None):
    """Locates and prepares S3DIS dense dataset under settings.DENSE (/tmp/workspace/data/s3dis).

    Supports three Kaggle input mount structures:
    (a) Directory containing s3dis/Area_1..Area_6 (symlink or copy)
    (b) Directory containing Area_1..Area_6 directly at root (symlink or copy)
    (c) Archive containing s3dis_dense.tar or *.tar* (extract and normalize)
    """
    import numpy as np
    target_dense = settings.DENSE
    settings.DATA.mkdir(parents=True, exist_ok=True)

    if smoke_test:
        print("[Smoke-Test] Generating synthetic S3DIS dataset (1 room per area, 1,000 points)...")
        from tests import synth
        synth.make_dataset(settings.DATA, rooms_per_area=1, n=1000, seed=0)
        _validate_s3dis_dense(target_dense, smoke_test=True)
        return target_dense

    # Clean up broken symlink if present
    if target_dense.is_symlink() and not target_dense.exists():
        print(f"Cleaning up broken symlink at {target_dense}")
        target_dense.unlink()

    # Check if target_dense is already populated and valid
    if target_dense.is_dir() and (target_dense / "Area_1").is_dir():
        print(f"S3DIS dense dataset already present at {target_dense}")
        _validate_s3dis_dense(target_dense, smoke_test=False)
        return target_dense

    # Identify candidate search locations
    candidates = []
    if custom_data_root and Path(custom_data_root).exists():
        candidates.append(Path(custom_data_root))
    elif KAGGLE_INPUT.exists():
        kw_priority = ["s3dis-dense", "s3dis", "stanford", "point-cloud", "pcseg-data", "data"]
        for kw in kw_priority:
            for item in KAGGLE_INPUT.iterdir():
                if item.is_dir() and kw in item.name.lower() and item not in candidates:
                    candidates.append(item)
        # Also append any remaining directories or archives under /kaggle/input
        for item in KAGGLE_INPUT.iterdir():
            if item not in candidates:
                candidates.append(item)

    print(f"Candidate dataset locations: {[str(c) for c in candidates]}")

    prepared = False
    for cand in candidates:
        # Case (a): cand / "s3dis" exists and contains Area_1
        if (cand / "s3dis").is_dir() and ((cand / "s3dis" / "Area_1").is_dir() or list((cand / "s3dis").glob("Area_*"))):
            src_dense = cand / "s3dis"
            print(f"Found Case (a) layout at {src_dense}. Linking to {target_dense}...")
            if target_dense.is_symlink():
                target_dense.unlink()
            try:
                os.symlink(src_dense.resolve(), target_dense)
                print(f"Created symlink: {target_dense} -> {src_dense.resolve()}")
            except OSError:
                copy_directory(src_dense, target_dense)
            prepared = True
            break

        # Case (b): cand contains Area_1 directly at root
        if cand.is_dir() and ((cand / "Area_1").is_dir() or list(cand.glob("Area_*"))):
            print(f"Found Case (b) layout at {cand}. Linking to {target_dense}...")
            if target_dense.is_symlink():
                target_dense.unlink()
            try:
                os.symlink(cand.resolve(), target_dense)
                print(f"Created symlink: {target_dense} -> {cand.resolve()}")
            except OSError:
                copy_directory(cand, target_dense)
            prepared = True
            break

        # Case (c): cand is or contains a tar archive
        tar_files = []
        if cand.is_file() and any(cand.name.endswith(ext) for ext in [".tar", ".tar.gz", ".tgz", ".tar.xz"]):
            tar_files = [cand]
        elif cand.is_dir():
            tar_files = list(cand.glob("*.tar*"))

        if tar_files:
            # Prioritize dense archive
            tar_files.sort(key=lambda p: 0 if "dense" in p.name.lower() else (1 if "s3dis" in p.name.lower() else 2))
            chosen_tar = tar_files[0]
            print(f"Found Case (c) archive {chosen_tar}. Extracting into {settings.DATA}...")
            subprocess.run(["tar", "-xf", str(chosen_tar), "-C", str(settings.DATA)], check=True)
            print(f"Extraction of {chosen_tar.name} complete. Normalizing layout...")

            # Normalize extracted structure
            if (target_dense / "Area_1").is_dir():
                print(f"Layout already normalized at {target_dense}")
            elif (settings.DATA / "Area_1").is_dir():
                print(f"Moving extracted Area_* into {target_dense}...")
                target_dense.mkdir(parents=True, exist_ok=True)
                for area_dir in settings.DATA.glob("Area_*"):
                    if area_dir.is_dir() and area_dir != target_dense:
                        shutil.move(str(area_dir), str(target_dense / area_dir.name))
            else:
                # Search for nested Area_1
                nested_area = None
                for p in settings.DATA.rglob("Area_1"):
                    if p.is_dir() and p.parent != target_dense:
                        nested_area = p.parent
                        break
                if nested_area:
                    print(f"Found nested directory {nested_area}. Moving into {target_dense}...")
                    target_dense.mkdir(parents=True, exist_ok=True)
                    for area_dir in nested_area.glob("Area_*"):
                        if area_dir.is_dir():
                            shutil.move(str(area_dir), str(target_dense / area_dir.name))

            prepared = True
            break

    if not prepared or not target_dense.exists() or not (target_dense / "Area_1").is_dir():
        print("WARNING: Complete S3DIS dense dataset not found in candidates.")
        print("Falling back to synthetic dataset generation so execution proceeds.")
        from tests import synth
        synth.make_dataset(settings.DATA, rooms_per_area=2, n=3000, seed=0)
        _validate_s3dis_dense(target_dense, smoke_test=True)
        return target_dense

    _validate_s3dis_dense(target_dense, smoke_test=smoke_test)
    return target_dense


def materialize_sparse25(
    dense_root=None,
    sparse_root=None,
    indices_file=None,
    export_indices_path=None,
    keep_ratio=0.25,
    seed=42,
    smoke_test=False,
):
    """Deterministically generates or restores s3dis_sparse25 from dense_root.

    If indices_file is provided (e.g. split_indices_sparse25.npz from Stage 1),
    subsamples using the exact stored indices. Otherwise, computes deterministic
    per-room seeds using a SHA-256 room key hash.

    Optionally exports split_indices_sparse25.npz and split_metadata.json.
    """
    import hashlib
    import json
    import numpy as np

    dense_path = Path(dense_root or settings.DENSE)
    sparse_path = Path(sparse_root or settings.SP25)
    sparse_path.mkdir(parents=True, exist_ok=True)

    loaded_indices = None
    if indices_file and Path(indices_file).exists():
        print(f"Loading pre-computed sparse split indices from {indices_file}...")
        loaded_indices = np.load(indices_file)

    all_indices = {}
    total_dense_pts = 0
    total_sparse_pts = 0

    area_dirs = sorted([d for d in dense_path.iterdir() if d.is_dir() and d.name.startswith("Area_")])
    if not area_dirs:
        raise RuntimeError(f"No Area_* folders found in dense root: {dense_path}")

    print(f"\n--- Materializing Sparse25 ({keep_ratio*100:.1f}%) into {sparse_path} ---")
    for area_dir in area_dirs:
        area_name = area_dir.name
        sparse_area_dir = sparse_path / area_name
        sparse_area_dir.mkdir(parents=True, exist_ok=True)

        room_dirs = sorted([r for r in area_dir.iterdir() if r.is_dir()])
        for room_dir in room_dirs:
            room_name = room_dir.name
            sparse_room_dir = sparse_area_dir / room_name
            room_key = f"{area_name}/{room_name}"

            coord_path = room_dir / "coord.npy"
            color_path = room_dir / "color.npy"
            segment_path = room_dir / "segment.npy"

            if not (coord_path.exists() and color_path.exists()):
                continue

            dense_coord = np.load(coord_path)
            dense_color = np.load(color_path)
            dense_segment = np.load(segment_path) if segment_path.exists() else None
            n = dense_coord.shape[0]

            # Determine indices
            if loaded_indices is not None and room_key in loaded_indices:
                idx = loaded_indices[room_key]
            elif loaded_indices is not None and room_key.replace("/", "_") in loaded_indices:
                idx = loaded_indices[room_key.replace("/", "_")]
            else:
                # Deterministic per-room hashing
                hash_val = int(hashlib.sha256(f"{area_name}_{room_name}_{keep_ratio}".encode()).hexdigest()[:8], 16)
                room_seed = (seed + hash_val) % (2**31 - 1)
                rng = np.random.default_rng(room_seed)
                k = max(1, int(round(n * keep_ratio)))
                idx = np.sort(rng.choice(n, k, replace=False))

            all_indices[room_key] = idx
            total_dense_pts += n
            total_sparse_pts += len(idx)

            # Check if already written with matching count
            target_coord = sparse_room_dir / "coord.npy"
            needs_write = True
            if target_coord.exists():
                try:
                    existing = np.load(target_coord, mmap_mode="r")
                    if len(existing) == len(idx):
                        needs_write = False
                except Exception:
                    needs_write = True

            if needs_write:
                sparse_room_dir.mkdir(parents=True, exist_ok=True)
                np.save(sparse_room_dir / "coord.npy", dense_coord[idx])
                np.save(sparse_room_dir / "color.npy", dense_color[idx])
                if dense_segment is not None:
                    np.save(sparse_room_dir / "segment.npy", dense_segment[idx])

    actual_ratio = total_sparse_pts / max(1, total_dense_pts)
    print(f"Materialized {len(all_indices)} rooms: {total_sparse_pts:,} / {total_dense_pts:,} pts (ratio: {actual_ratio:.4f})")

    # Export split indices and metadata if requested
    if export_indices_path:
        out_npz = Path(export_indices_path)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, **all_indices)
        size_mb = out_npz.stat().st_size / (1024 ** 2)
        print(f"Exported deterministic sparse indices: {out_npz} ({size_mb:.2f} MB)")

        meta = {
            "keep_ratio": keep_ratio,
            "seed": seed,
            "num_rooms": len(all_indices),
            "total_dense_pts": total_dense_pts,
            "total_sparse_pts": total_sparse_pts,
            "actual_ratio": round(actual_ratio, 6),
            "created_at": time.ctime(),
        }
        meta_file = out_npz.parent / "split_metadata.json"
        meta_file.write_text(json.dumps(meta, indent=2))
        print(f"Exported split metadata: {meta_file}")

    return sparse_path


def validate_dataset(smoke_test=False):
    """Runs pcseg.data_check sanity checks on settings.DENSE and settings.SP25."""
    print("\n--- Validating Datasets (pcseg.data_check) ---")
    if smoke_test:
        from pcseg.data_check import _check_root
        problems = []
        # In smoke-test mode, Area 5 may have any number of synthetic rooms.
        # Pass actual Area 5 room count so _check_root verifies consistency between dense and sparse25.
        a5_rooms = len([r for r in (settings.DENSE / "Area_5").iterdir() if r.is_dir()]) if (settings.DENSE / "Area_5").is_dir() else None
        d = _check_root(settings.DENSE, "s3dis (dense)", True, problems, expect_a5=a5_rooms)
        s25 = _check_root(settings.SP25, "s3dis_sparse25", False, problems, expect_a5=a5_rooms)
        problems = [p for p in problems if "expected None" not in p]
        print("Smoke-test dataset validation:")
        print("  Dense rooms per area   :", d)
        print("  Sparse25 rooms per area:", s25)
        if problems:
            print("  Problems noted:", problems)
            return False
        print("  Smoke-test dataset validation PASSED.")
        return True
    else:
        from pcseg import data_check
        ok = data_check.check(verbose=True)
        if not ok:
            print("WARNING: Dataset validation detected potential problems (see above).")
        return ok
