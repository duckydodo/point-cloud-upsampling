# Project File Inventory

This inventory documents all source code, notebooks, configuration files, test scripts, and documentation in the `point-cloud-upsampling` repository as of October 2026.

---

## 1. Core Python Package (`pcseg/`)

| File | Purpose | Inputs | Outputs | Pipeline Stage | Git Status | Origin / Maintenance | Execution Status |
|---|---|---|---|---|---|---|---|
| `pcseg/__init__.py` | Package initializer. Exposes version and package namespace. | None | Python package namespace | All | Tracked | Maintained | Verified (imported in tests) |
| `pcseg/settings.py` | Central paths and experiment knobs (`Knobs` dataclass). Resolves `ROOT`, `DATA`, `POINTCEPT`, `EXP`, `RESULTS`, `BANKS`, `UPSAMPLER`. | Environment variables (`PCSEG_ROOT`, `PCSEG_DATA`, `POINTCEPT_DIR`), optional `pcseg_settings.json` | Directory paths, `Knobs` instance | Stages 0–7 | Tracked | Maintained | Verified (loaded in tests and scripts) |
| `pcseg/common.py` | Shared constants (13 S3DIS classes, `TRAIN_AREAS`, `TEST_AREA`) and I/O helpers (`load_room`, `save_room`, `room_dirs`, `write_json`). | Room directories containing `.npy` or `.pth` point clouds | Dict of arrays (`coord`, `color`, `segment`, `normal`), JSON files | Stages 0–7 | Tracked | Maintained | Verified |
| `pcseg/compat.py` | Compatibility layer: pure-PyTorch fallback shims for `torch_scatter`, inert stubs for uncalled Pointcept dependencies, and idempotent Pointcept patcher (`patch_pointcept`). | Target Pointcept directory | Pure-PyTorch stubs in `workspace/stubs/`, patches appended to `pointcept/datasets/transform.py` | Stage 1 (Setup) | Tracked | Maintained | Verified (idempotency & AST validated) |
| `pcseg/data_check.py` | Dataset integrity verifier for processed S3DIS directories. Checks point counts, bounding boxes, feature dimensions, and sparsity ratios. | Processed S3DIS folders (`s3dis`, `s3dis_sparse25`) | Terminal verification report, boolean pass/fail | Stage 0 (Sanity check) | Tracked | Maintained | Unit verified; dataset execution pending local data |
| `pcseg/env_setup.py` | System check and Pointcept bootstrap. Checks CUDA, PyTorch, SpConv, and AMP; clones Pointcept at commit `1342eda` and runs GPU smoke test. | CUDA driver, PyTorch installation, Git | Cloned `workspace/Pointcept`, applied patches, smoke-test logs | Stage 1 (Setup) | Tracked | Maintained | Environment check verified; GPU smoke test pending GPU |
| `pcseg/make_configs.py` | Pointcept configuration builder. Generates PTv3 S3DIS configs, filters failing post-training evaluation hooks (`PreciseEvaluator`, `SemSegEvaluator`), sets RGB-only channels, and configures `SphereCrop(mode='all')`. | Base Pointcept config, experiment parameters, hardware knobs | Python config files in `configs/s3dis/pcseg_<name>.py` and `workspace/exp/<name>/config.py` | Stage 2 (Config gen) | Tracked | Maintained | Verified (unit test passes) |
| `pcseg/experiments.py` | Standardized experiment config factory (`make_all`, `mk`, `make_H`). Guarantees identical schedules across baseline rows A, B, D, and H. | Hardware knobs from `pcseg_settings.json` | Dictionary of generated config paths | Stages 2, 5 | Tracked | Maintained | Verified |
| `pcseg/run.py` | Subprocess orchestrator for Pointcept training (`tools/train.py`), testing (`tools/test.py`), speed benchmarking, and evaluation collation. | Experiment names, generated configs, trained checkpoint weights | Execution logs, `train_done.json`, predictions in `workspace/exp/<run>/test_<tag>/result/` | Stages 2, 4, 5 | Tracked | Maintained | Verified logic; live training pending GPU |
| `pcseg/eval_dense.py` | Standalone metric evaluator. Projects predictions back to original dense points via 1-NN KD-Tree, computes confusion matrix, and calculates mIoU, OA, and mAcc. | Prediction `.npy` files, sparse/upsampled room coords, dense ground truth coords/segments | Metrics dictionary, JSON results in `workspace/results/<label>.json` | Stages 5, 6 | Tracked | Maintained | Logic verified; live evaluation executed on lab GPU |
| `pcseg/pu_core.py` | PyTorch point cloud upsampler network (`PUNet`), custom EdgeConv, Chamfer Distance, Repulsion loss, Uniformity loss, and patch metrics. | Batched patches of shape `(B, N, 3)` | Upsampled patches `(B, N*r, 3)`, loss values, metric dicts | Stage 3 (Upsampler) | Tracked | Maintained | Verified (tested in pure Python & PyTorch CPU) |
| `pcseg/pu_data.py` | Patch bank builder (`build_patch_bank`), scene inference (`upsample_scene`), naive interpolation baseline (`interp_scene`), and dataset generator. | Dense and sparse room point clouds | Compressed patch banks (`.npz`), upsampled scene files in Pointcept format | Stages 3, 5 | Tracked | Maintained | Verified logic; dataset run pending data |
| `pcseg/pu_train.py` | CLI training loop for the learned upsampler. Supports OneCycleLR schedule, data augmentation (rotations, flips, scaling), and metric tracking. | Patch banks (`train_dev_x4.npz`, `val_x4.npz`) | Checkpoints (`last.pt`, `best.pt`), training curve log (`log.json`) | Stage 3 (Upsampler) | Tracked | Maintained | Tested via synthetic CPU test |
| `pcseg/pu_infer.py` | CLI entry point for full-scene upsampling inference or naive midpoint interpolation over whole S3DIS rooms. | Trained upsampler checkpoint (`best.pt` or `last.pt`), sparse S3DIS directory | Upsampled scene directory (`s3dis_up25x4` or `s3dis_interp25x4`) | Stage 5 (Inference) | Tracked | Maintained | CLI interface verified |
| `pcseg/up_metrics.py` | Geometric quality evaluator for upsampled scenes: L1 Chamfer distance (cm), 99th-percentile Hausdorff distance (cm), F-score, and uniformity CV. | Upsampled point cloud, dense ground-truth point cloud | Dictionary of geometric metrics, cost profiling (params, latency, peak VRAM) | Stages 3, 6 | Tracked | Maintained | Verified |
| `pcseg/viz.py` | Visualisation utilities: qualitative 3D point cloud plotting (`plot_panels`), label color mapping, and error highlight rendering (`error_colors`). | Room coordinates, RGB colors, semantic predictions, ground truth labels | PNG figure files in `workspace/figures/*.png` | Stage 6 (Analysis) | Tracked | Maintained | Verified |
| `pcseg/results.py` | Collation utility for aggregating result JSON files from `workspace/results/` into summary tables and Markdown/LaTeX formats. | JSON files in `workspace/results/` | Structured result rows, Markdown tables | Stage 6 (Reporting) | Tracked | Maintained | Verified |

---

## 2. Notebooks (`notebooks/`)

| File | Purpose | Main Functions Used | Expected Outputs | Git Status | Execution Status |
|---|---|---|---|---|---|
| `notebooks/pcseg_part2_pointcept_baselines.ipynb` | Pointcept baseline training and evaluation: env checks, PTv3 speed benchmarking, Phase 0 gate (`P0_dense`, `P0_sparse25`), full baselines Rows A, B, C, D. | `env_setup.check_env()`, `env_setup.prepare_pointcept()`, `experiments.make_all()`, `run.train()`, `run.evaluate_run()` | Benchmark speed report, Phase 0 gate decision, trained baseline models, Area 5 baseline predictions | Tracked | Verified clean syntax (0 saved outputs; Phase 0 completed on lab GPU) |
| `notebooks/pcseg_part3_upsampler.ipynb` | Learned upsampler development: patch bank construction, `dev_x4` training, loss ablation studies (CD only, CD+Rep, CD+Rep+Uni), full production `full_x4` training, cross-fit folds. | `pu_data.build_patch_bank()`, `pu_train.main()`, `up_metrics.scene_metrics()`, `viz.plot_panels()` | Patch banks in `workspace/banks/`, checkpoints in `workspace/upsampler/`, curve plots, scene metrics JSON | Tracked | Verified clean syntax (0 saved outputs) |
| `notebooks/pcseg_part4_pipeline_results.ipynb` | End-to-end pipeline evaluation: upsampling Area 5, evaluating Row E (interpolation) and Row G (learned upsampling), cross-fit Row H training, ablation matrix, final tables & figures. | `pu_infer.main()`, `run.evaluate_run()`, `results.collect()`, `viz.plot_panels()` | Upsampled datasets in `workspace/data/`, evaluation JSONs in `workspace/results/`, summary tables, `main_results.png` | Tracked | Verified clean syntax (0 saved outputs; prediction-shape bug fixed) |

---

## 3. Automation Scripts (`scripts/`)

| File | Purpose | Arguments | Expected Outputs | Git Status | Execution Status |
|---|---|---|---|---|---|
| `scripts/run_ptv3.py` | Headless, resumable batch runner for long PTv3 training and evaluation jobs (Rows A, B, D, H). | List of rows to execute (e.g. `A B D` or `H`) | Standard output log, checkpoints in `workspace/exp/`, test predictions | Tracked | Maintained | Verified interface; intended for long background runs |

---

## 4. Test Suite (`tests/`)

| File | Purpose | Dependencies | Target Component | Git Status | Execution Status |
|---|---|---|---|---|---|
| `tests/synth.py` | Tiny synthetic S3DIS generator (points on box faces + furniture blobs with colors and labels). | `numpy` | Test data fixture | Tracked | Maintained | Verified |
| `tests/test_numpy_parts.py` | CPU tests for patch bank creation, data checks, evaluation confusion matrix, and result collation. | `scipy`, `numpy` | `pcseg.pu_data`, `pcseg.eval_dense`, `pcseg.results` | Tracked | Maintained | Verified on environments with `scipy` |
| `tests/test_torch_cpu.py` | CPU end-to-end smoke test for upsampler network forward pass, backward pass, loss computation, and naive interpolation. | `torch`, `numpy` | `pcseg.pu_core` | Tracked | Maintained | Verified on environments with `torch` |
| `tests/test_make_configs.py` | Unit test for Pointcept configuration generation, verifying filtering of `PreciseEvaluator` and `SemSegEvaluator`, and presence of `SphereCrop(mode='all')`. | None (self-contained mock) | `pcseg.make_configs` | Tracked | Maintained | **PASSED** (Exit code 0) |
| `tests/test_pred_shape.py` | Regression test demonstrating outer-product broadcasting hazard between 2D predictions `(N, 1)` and 1D labels `(N,)`, and verifying `.reshape(-1)` fix. | None (pure-Python fallback + NumPy support) | Notebook Part 4 Cell 19, `pcseg.viz` | Tracked | Maintained | **PASSED** (Exit code 0) |
| `tests/test_sphere_crop.py` | Regression test for evaluation-time fragmentation: verifies point count upper bound, 100% point coverage (zero points dropped), index preservation, and prediction aggregation. | None (pure-Python reference + NumPy support) | `pcseg.compat.SPHERE_CROP_ALL`, `pcseg.make_configs` | Tracked | Maintained | **PASSED** (Exit code 0) |

---

## 5. Documentation (`docs/` & Root)

| File | Purpose | Git Status | Origin / Maintenance |
|---|---|---|---|
| `README.md` | Primary project overview, research question, architecture, setup instructions, verified experimental findings, and citations. | Tracked | Manually maintained |
| `docs/LOCAL_RUN_GUIDE.md` | Comprehensive operator guide for running experiments locally or in WSL2: requirements, environment setup, data placement, step-by-step instructions. | Tracked | Manually maintained |
| `docs/COLAB_DEBUGGING_WORKFLOW.md` | Reference guide describing legacy Google Colab workflows, Drive mount structures, and debugging procedures. | Tracked | Historical documentation |
| `docs/REPOSITORY_LAYOUT.md` | High-level repository structural overview. | Tracked | Manually maintained |
| `docs/PROJECT_FILE_INVENTORY.md` | Detailed file inventory (this document). | Created | Manually maintained |
| `docs/EXPERIMENT_STATUS.md` | Precise experiment execution tracking, provenance records, and known limitations. | Created | Manually maintained |
| `docs/PRESENTATION_AND_VIVA.md` | Presentation pitches, architectural breakdowns, loss derivations, PTv3 details, and 20+ viva Q&As. | Created | Manually maintained |
| `docs/GITHUB_RELEASE_CHECKLIST.md` | Pre-release audit checklist, file classifications, Google Drive sync protocol, and secret scans. | Created | Manually maintained |

---

## 6. Runtime & External Directories (`workspace/` - Untracked)

The entire `workspace/` tree is excluded from Git via `.gitignore`. Its internal layout is:

- `workspace/data/`: Processed S3DIS datasets (`s3dis`, `s3dis_sparse25`, `s3dis_up25x4`, `s3dis_interp25x4`).
- `workspace/Pointcept/`: Pointcept repository checkout (commit `1342eda`).
- `workspace/banks/`: Compressed patch banks (`*.npz`).
- `workspace/upsampler/`: Trained upsampler checkpoints (`last.pt`, `best.pt`, `log.json`).
- `workspace/exp/`: PTv3 experiment runs (`bench`, `P0_dense`, `P0_sparse25`, `A_dense`, etc.) containing checkpoints and evaluation predictions.
- `workspace/results/`: Evaluation metric outputs (`*.json`).
- `workspace/figures/`: Generated visualisations and quantitative figures (`*.png`).
- `workspace/stubs/`: Pure-PyTorch compatibility shims generated at runtime.
- `workspace/pcseg_settings.json`: Hardware-tuned budget knobs shared across all notebooks.
