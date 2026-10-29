# Experiment Status and Provenance Report

**Project:** Learning-Based Point-Cloud Upsampling for Improved Semantic Segmentation of Sparse 3D Point Clouds  
**Last Updated:** October 2026  
**Hardware Platforms:**  
- **Host / Local Audit Machine:** Fedora Linux x86_64, Python 3.14 (Code audit, syntax validation, self-contained unit/regression tests).
- **Lab GPU Machine:** NVIDIA RTX 2000 Ada (16 GB VRAM, Ada Lovelace sm_89), Python 3.10, PyTorch 2.5.0 + CUDA 12.4, SpConv-cu124.
- **Historical Google Colab Environment:** Tesla T4 (15.7 GB VRAM, Turing sm_75), CUDA 12.4 (historical run records, benchmark trace).

---

## 1. Executive Summary of Experiment Status

| Experiment Row / Stage | Objective | Target Data (Train -> Test) | Status | Checkpoints Available? | Evaluation Metrics Verified? | Provenance Notes |
|---|---|---|---|---|---|---|
| **PTv3 Benchmark (`bench`)** | Iteration speed & VRAM profiling | `s3dis` -> None (train only, 102 iters) | **Completed** | Yes (`model_last.pth`) | N/A (speed: ~0.45s/it on T4) | Completed on Colab; post-training hook crash identified and resolved via hook filter in `pcseg/make_configs.py`. |
| **Phase 0 Gate: Dense (`P0_dense`)** | Short schedule (20 epochs) sanity check on dense data | `s3dis` -> `Area_5` dense | **Trained; Eval Pending** | Yes (Saved on Lab Machine: `model_last.pth`) | Pending Area 5 Eval | Training completed (20/20 epochs). Initial Area 5 evaluation crashed with CUDA OOM inside PTv3 attention. Evaluation fragmentation patch prepared. |
| **Phase 0 Gate: Sparse (`P0_sparse25`)** | Short schedule (20 epochs) sanity check on 25% sparse data | `s3dis_sparse25` -> `Area_5` sparse | **Trained; Eval Pending** | Yes (Saved on Lab Machine: `model_last.pth`) | Pending Area 5 Eval | Training completed (20/20 epochs). Initial Area 5 evaluation crashed with CUDA OOM inside PTv3 attention. Evaluation fragmentation patch prepared. |
| **Row A (`A_dense`)** | Upper baseline: trained on dense S3DIS | `s3dis` -> `Area_5` dense | **Planned / Ready** | No | No | Config generation and pipeline code verified; full training (100 epochs) pending Phase 0 gate decision. |
| **Row B (`B_sparse25`)** | Direct sparse baseline: trained on 25% sparse data | `s3dis_sparse25` -> `Area_5` sparse25 | **Planned / Ready** | No | No | Pipeline code verified; awaiting execution. |
| **Row C (`A_dense_on_sparse25`)** | Degradation test: dense-trained model tested on sparse data | `s3dis` -> `Area_5` sparse25 | **Planned / Ready** | Reuses Row A checkpoint | No | Test-only row using `A_dense` weights. |
| **Row D (`D_dense_aug`)** | Data augmentation baseline: dense trained with RandomSparsify | `s3dis` (augmented) -> `Area_5` sparse25 | **Planned / Ready** | No | No | `RandomSparsify` transform implemented and registered; awaiting execution. |
| **Upsampler Development (`dev_x4`)** | Validate PUNet architecture on Areas 1–4, val on Area 6 | Areas 1–4 patches -> Area 6 val patches | **Planned / Ready** | No (Drive folder was empty) | No | Patch bank generation code and training CLI verified; synthetic CPU forward/backward tests pass. |
| **Upsampler Loss Ablation** | CD only vs CD+Rep vs CD+Rep+Uni | Areas 1–4 patches -> Area 6 val patches | **Planned / Ready** | No | No | Implemented in Part 3 Notebook Cell 10; ready to execute. |
| **Upsampler Production (`full_x4`)** | Train upsampler on all training areas (1–4, 6) | Areas 1–4, 6 patches -> Area 6 | **Planned / Ready** | No | No | Code ready for patch bank construction and training. |
| **Upsampler Cross-Fit (`foldA`, `foldB`)** | Two-fold training to upsample training scenes for Row H | Fold A (1–3), Fold B (4, 6) | **Planned / Ready** | No | No | Implemented in Part 3 Notebook Cell 14. |
| **Row E (`interp25x4`)** | Naive baseline: dense-trained model on 4x interpolated points | `s3dis` -> `Area_5` interp25x4 | **Planned / Ready** | Reuses Row A checkpoint | No | Midpoint interpolation implemented in `pcseg/pu_data.py`. |
| **Row G (`up25x4`)** | Proposed test-time method: dense-trained model on upsampled points | `s3dis` -> `Area_5` up25x4 | **Planned / Ready** | Reuses Row A checkpoint + `full_x4` upsampler | No | Inference pipeline and 1-NN projection implemented in `pcseg/pu_infer.py` and `pcseg/eval_dense.py`. |
| **Row H (`H_up_trained`)** | Proposed full method: PTv3 trained and tested on upsampled scenes | `s3dis_up25x4` (cross-fit) -> `Area_5` up25x4 | **Planned / Ready** | No | No | End-to-end workflow ready in Part 4. |

---

## 2. Hardware and Environment Audit

### A. Lab GPU Machine (NVIDIA RTX 2000 Ada)
- **GPU Architecture:** Ada Lovelace (Compute Capability 8.9), 16 GB GDDR6 VRAM with ECC.
- **Driver & CUDA:** Driver R550+, CUDA 12.4.
- **Python / PyTorch Stack:** Python 3.10, PyTorch 2.5.0 (`cu124`), `spconv-cu124` 2.3.6.
- **Verified Execution on Lab Machine:**
  - `P0_dense` completed training (20 epochs). Checkpoint `model_last.pth` successfully written to `workspace/exp/P0_dense/model/`.
  - `P0_sparse25` completed training (20 epochs). Checkpoint `model_last.pth` successfully written to `workspace/exp/P0_sparse25/model/`.
  - Area 5 evaluation encountered CUDA OOM inside PTv3 attention during full-room testing.

### B. Fedora Local Host
- **Environment:** Fedora Linux x86_64, Python 3.14.8.
- **Role:** Code inspection, AST validation, static analysis, unit test development.
- **Verified Local Tests:**
  - `tests/test_make_configs.py`: PASSED (Exit code 0).
  - `tests/test_pred_shape.py`: PASSED (Exit code 0).
  - `tests/test_sphere_crop.py`: PASSED (Exit code 0).

### C. Google Drive / Colab Historical Trace
- **Inspection Finding:** An earlier Google Drive audit verified that `pcseg/results/` and `pcseg/upsampler/dev_x4/` were empty.
- **Implication:** No prior training curves, checkpoints, or final evaluation results can be claimed as completed on Google Drive. All production results will originate from the lab GPU.

---

## 3. Investigated Defects and Verified Resolutions

### Defect 1: Benchmark Post-Training Crash (`PreciseEvaluator`)
- **Symptom:** `tools/train.py` exited with code 1 after completing all 102 iterations because `PreciseEvaluator.after_train()` attempted to load `model_best.pth`.
- **Root Cause:** `pcseg/make_configs.py` sets `cfg.evaluate = False` (disabling validation during training to eliminate test-set leakage). `CheckpointSaver` only writes `model_best.pth` when evaluation runs. The inherited hook list still contained `PreciseEvaluator(test_last=False)`, which unconditionally demanded `model_best.pth`.
- **Resolution:** Added hook filtering in [pcseg/make_configs.py:42-43](file:///home/chitta/Projects/point-cloud-upsampling/pcseg/make_configs.py#L42-L43):
  ```python
  if "hooks" in cfg:
      cfg.hooks = [h for h in cfg.hooks if h.get("type") not in ("PreciseEvaluator", "SemSegEvaluator")]
  ```
- **Validation:** Tested via `tests/test_make_configs.py`. Generated configs retain `CheckpointLoader`, `IterationTimer`, `InformationWriter`, and `CheckpointSaver`, while omitting the crashing evaluators.

### Defect 2: Area 5 Evaluation CUDA Out-of-Memory
- **Symptom:** Evaluating Phase 0 checkpoints on Area 5 crashed with CUDA Out-of-Memory inside PTv3's relative-position attention operation.
- **Root Cause:** `configs/s3dis/semseg-pt-v3m1-1-rpe.py` sets `test_cfg.crop = None`. In `workspace/Pointcept/pointcept/datasets/defaults.py`, voxelization fragments containing 300,000–500,000+ points in large Area 5 rooms were passed directly to PTv3 without spatial partitioning. Furthermore, Pointcept's `SphereCrop` had `mode="all"` listed in `__init__` but raised `NotImplementedError` in `__call__`.
- **Resolution:**
  1. Implemented complete `mode="all"` greedy sphere partitioning in `pcseg/compat.py` via `SPHERE_CROP_ALL`, registered into `TRANSFORMS` with `force=True`.
  2. Updated `pcseg/make_configs.py` to set `d.test.test_cfg.crop = dict(type="SphereCrop", point_max=point_max, mode="all")`.
  3. Updated `pcseg/run.py` to resolve the generated configuration containing `test_cfg.crop`.
- **Validation:** Tested via `tests/test_sphere_crop.py` (verifying point count bounds, 100% point coverage, index preservation, and prediction aggregation).

### Defect 3: Array Broadcasting Memory Hazard in Visualization
- **Symptom:** Potential terabyte-scale memory allocation during qualitative error plotting.
- **Root Cause:** In Notebook Part 4 Cell 19, predictions loaded from `*_pred.npy` were shaped `(N, 1)` and compared against 1D ground truth `(N,)` without `.reshape(-1)`. Array comparison `(pred == gt)` triggered NumPy broadcasting into an $(N \times N)$ boolean matrix (~2.25 TB RAM for $N=1.5\text{M}$ points).
- **Resolution:** Added `.reshape(-1)` to `pred_up` in Notebook Part 4 Cell 19.
- **Validation:** Tested via `tests/test_pred_shape.py`. Verified that flattened 1D arrays prevent broadcasting and evaluate safely.

---

## 4. Current Limitations and Pending Work

1. **Phase 0 Evaluation Execution:** Checkpoints for `P0_dense` and `P0_sparse25` exist on the lab machine. Evaluation on Area 5 must be rerun using the `SphereCrop(mode="all")` patch to obtain the mIoU sparsity gap.
2. **Upsampler Training on Lab Machine:** `dev_x4` and `full_x4` must be trained on the lab GPU after patch bank extraction.
3. **Full Baseline Training (Rows A, B, D, H):** Awaiting Phase 0 gate confirmation before running the full 100-epoch training schedule.
4. **Honest Reporting Commitment:** Results will be reported strictly as observed. If learned upsampling does not improve mIoU over naive interpolation or direct sparse training, that negative result will be documented and analyzed scientifically.
