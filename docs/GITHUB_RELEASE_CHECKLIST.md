# GitHub Publication and Release Checklist

**Project:** Learning-Based Point-Cloud Upsampling for Improved Semantic Segmentation of Sparse 3D Point Clouds  
**Repository:** `/home/chitta/Projects/point-cloud-upsampling`

Before publishing this repository publicly on GitHub or tagging a final release, verify each item in this checklist.

---

## 1. File Classification Matrix

### Category A: Required Source Code (Track in Git)
- [x] `pcseg/__init__.py`: Package init.
- [x] `pcseg/settings.py`: Path and hardware knob management.
- [x] `pcseg/common.py`: Shared constants, classes, and I/O.
- [x] `pcseg/compat.py`: Compatibility stubs, PyTorch shims, Pointcept patcher.
- [x] `pcseg/data_check.py`: Dataset validation CLI.
- [x] `pcseg/env_setup.py`: Environment verification and Pointcept bootstrap.
- [x] `pcseg/make_configs.py`: Pointcept config generator with hook & crop fixes.
- [x] `pcseg/experiments.py`: Unified experiment config definitions.
- [x] `pcseg/pu_core.py`: Upsampler architecture, losses, patch metrics.
- [x] `pcseg/pu_data.py`: Patch extraction, scene inference, interpolation.
- [x] `pcseg/pu_train.py`: Upsampler training loop CLI.
- [x] `pcseg/pu_infer.py`: Upsampling inference CLI.
- [x] `pcseg/up_metrics.py`: Geometric quality metrics.
- [x] `pcseg/run.py`: Pointcept execution and test runner.
- [x] `pcseg/eval_dense.py`: 1-NN projection and metric evaluation.
- [x] `pcseg/results.py`: Metric collation and Markdown/LaTeX reporting.
- [x] `pcseg/viz.py`: Qualitative 3D visualization utilities.

### Category B: Reproducible Notebooks & Automation Scripts (Track in Git)
- [x] `notebooks/pcseg_part2_pointcept_baselines.ipynb` (outputs stripped).
- [x] `notebooks/pcseg_part3_upsampler.ipynb` (outputs stripped).
- [x] `notebooks/pcseg_part4_pipeline_results.ipynb` (outputs stripped; `.reshape(-1)` fix applied).
- [x] `scripts/run_ptv3.py`: Headless batch runner.

### Category C: Unit & Regression Test Suite (Track in Git)
- [x] `tests/synth.py`: Synthetic data generator.
- [x] `tests/test_numpy_parts.py`: Data and evaluation unit test.
- [x] `tests/test_torch_cpu.py`: Upsampler CPU forward/backward test.
- [x] `tests/test_make_configs.py`: Config generation & hook filtering test.
- [x] `tests/test_pred_shape.py`: Prediction-shape broadcasting hazard test.
- [x] `tests/test_sphere_crop.py`: Evaluation-time fragmentation regression test.

### Category D: Documentation (Track in Git)
- [x] `README.md`: Comprehensive, verified research documentation.
- [x] `requirements.txt`: Python package dependencies.
- [x] `.gitignore`: Robust ignore rules for data, checkpoints, and caches.
- [x] `docs/LOCAL_RUN_GUIDE.md`: Local setup and execution operator manual.
- [x] `docs/PROJECT_FILE_INVENTORY.md`: Comprehensive file inventory.
- [x] `docs/EXPERIMENT_STATUS.md`: Provenance and execution tracking.
- [x] `docs/PRESENTATION_AND_VIVA.md`: Defense preparation and viva Q&A guide.
- [x] `docs/GITHUB_RELEASE_CHECKLIST.md`: This pre-release checklist.

### Category E: Large Data Artifacts (NEVER Track in Git)
- [ ] `workspace/data/s3dis/` (~4.9 GB): Dense S3DIS processed point clouds.
- [ ] `workspace/data/s3dis_sparse25/` (~1.8 GB): 25% subsampled S3DIS scenes.
- [ ] `workspace/data/s3dis_sparse12/` (~0.9 GB): 12.5% subsampled S3DIS scenes.
- [ ] `workspace/data/s3dis_up25x4/` (~7.0 GB): Upsampled S3DIS scenes.
- [ ] `workspace/data/s3dis_interp25x4/` (~7.0 GB): Interpolated S3DIS scenes.
*Verification:* `.gitignore` must contain `workspace/` and `data/`.

### Category F: Model Checkpoints & Raw Logs (Store on Release Storage / Drive)
- [ ] `workspace/exp/*/model/*.pth`: PTv3 model weights (e.g. `model_last.pth` ~130 MB each).
- [ ] `workspace/upsampler/*/*.pt`: Upsampler weights (e.g. `best.pt` ~2.5 MB).
- [ ] `workspace/exp/*/test_*/result/*_pred.npy`: Raw test predictions (~10–50 MB per room).
*Rule:* Model checkpoints and raw per-point predictions belong in GitHub Releases / Zenodo / Drive, not in ordinary Git commit history.

---

## 2. Dependency Specification Audit (`requirements.txt`)

- **Pinned Core Dependencies:**
  - `numpy==1.26.4` (Ensures NumPy 1.x compatibility with SpConv and Pointcept C-extensions; avoids NumPy 2.x breaking changes).
  - `spconv-cu124` (Matches CUDA 12.4).
  - `torch-geometric`, `addict`, `yapf`, `timm`, `einops` (Pointcept required dependencies).
  - `scipy`, `pandas`, `matplotlib`, `tabulate` (Evaluation, collation, and plotting).
- **External Framework:**
  - Pointcept is cloned dynamically via `git clone` pinned to commit `1342eda` (Pointcept v1.6.0 era). It is NOT installed as a PyPI wheel.
- **PyTorch Installation Note:**
  - PyTorch 2.5.0 + CUDA 12.4 must be installed prior to `pip install -r requirements.txt`:
    ```bash
    pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
    ```

---

## 3. Secret and Artifact Hygiene Scan

Prior to public push, verify:
1. **Regex Scan for Secrets:**
   ```bash
   grep -rEi "api[_-]?key|secret|password|bearer|auth[_-]?token" --exclude-dir=".git" --exclude-dir="__pycache__" .
   ```
   *Expected:* 0 matches for live credentials or personal access tokens.
2. **File Size Check:**
   ```bash
   find . -type f -size +1M -not -path "*/.git/*" -not -path "*/workspace/*"
   ```
   *Expected:* 0 tracked files over 1 MB.
3. **Notebook Output Check:**
   Confirm that all `.ipynb` files in `notebooks/` have clean, cleared output cells before committing.

---

## 4. Google Drive Synchronization Protocol

When consolidating artifacts between Google Drive and the final GitHub repository:

| Asset Type | Google Drive Path | Target Repository Path | Sync Action |
|---|---|---|---|
| **Raw Datasets** | `MyDrive/pcseg/backup/s3dis.tar.gz` | External host / scratch SSD | Do not commit to Git. Provide download link in documentation. |
| **Processed Data** | `MyDrive/pcseg/backup/s3dis/` | `workspace/data/s3dis/` | Kept in local `workspace/data/`. Excluded from Git. |
| **Checkpoints** | `MyDrive/pcseg/exp/*/model_last.pth` | GitHub Release / Drive | Upload final `A_dense`, `B_sparse25`, `full_x4` checkpoints as GitHub Release assets. |
| **Result JSONs** | `MyDrive/pcseg/results/*.json` | `workspace/results/*.json` | Lightweight result summaries (< 50 KB) may be committed into a `results/` folder for reproducibility. |
| **Source Code** | `MyDrive/pcseg/pcseg/` | `pcseg/` | GitHub repository is the canonical source of truth. Overwrite Drive copies with GitHub version. |
