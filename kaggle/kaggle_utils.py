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


def setup_kaggle_env(install_spconv=True):
    """Detects CUDA version, installs fast wheels, and prepares Pointcept at commit 1342eda."""
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
    if g:
        print(f"Detected GPU 0: {g[0]} ({g[1]:.1f} GB VRAM, CC {g[2]})")
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

    # Install spconv if missing in environment and on CUDA
    if install_spconv:
        try:
            import spconv.pytorch  # noqa: F401
            print("spconv already installed.")
        except ImportError:
            print("Installing spconv-cu124...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "spconv-cu124"], check=False)
            try:
                import spconv.pytorch  # noqa: F401
                print("spconv-cu124 installed successfully.")
            except ImportError:
                print("spconv-cu124 failed; attempting generic spconv...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "spconv"], check=False)

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
