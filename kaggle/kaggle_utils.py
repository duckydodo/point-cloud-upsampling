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
    """Scans /kaggle/input for directories matching given keywords."""
    if not KAGGLE_INPUT.exists():
        return None
    for item in KAGGLE_INPUT.iterdir():
        if item.is_dir():
            name_lower = item.name.lower()
            if any(k.lower() in name_lower for k in dataset_name_keywords):
                return item
    return None
