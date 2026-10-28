# Local run guide: Parts 2, 3 and 4 in VS Code on your own GPU

This replaces the Colab guide. The same experiments (rows A–H, ablations, cost, figures) run from three notebooks in VS Code, on your machine, with no Google Drive and no runtime resets.

**Contents:** 1 What changed · 2 Requirements · 3 One-time setup · 4 Order of work · 5 Step by step · 6 Choosing the training budget · 7 What is saved where · 8 Working habits · 9 If something fails · 10 Changes to the code

---

## 1. What changed compared with Colab

| Colab | Local |
|---|---|
| Google Drive mount, `/content`, restore cells (section 0b) | Everything lives in `workspace/` next to the notebooks; nothing to restore |
| `%writefile` cells that wrote the code to Drive | The code is ordinary files in `pcseg/` (open and edit them in VS Code) |
| A separate Python 3.10 environment built by `uv` in every session (8–12 min) | **One** conda environment, built once; it is both the notebook kernel and the environment Pointcept runs in |
| "Reset routine" after each disconnect | Not needed. If the kernel restarts, run the setup cell and the section you were in |
| Budget knobs typed into two notebooks (and "MUST equal" warnings) | One file, `workspace/pcseg_settings.json`, written by Part 2 section 3 and read by Parts 2 and 4 |
| T4 (16 GB) defaults: batch 4, 100 000 points | The notebook suggests starting values for your VRAM; the benchmark decides |

The experiments, losses, data handling, rows and metrics are unchanged. The few code changes are listed in section 10.

## 2. Requirements

- **NVIDIA GPU** with a recent driver. The pinned stack is Python 3.10, PyTorch 2.5.0 with CUDA 12.4 and `spconv-cu124`; the wheels bundle their own CUDA runtime, so you only need an NVIDIA driver recent enough for CUDA 12.x wheels (R525 or newer is the minimum, R550 or newer is recommended). You do **not** need to install the CUDA toolkit.
- **Linux, or Windows with WSL2 (Ubuntu)**. Pointcept and spconv are developed for Linux; I have not tested native Windows and do not recommend it. In VS Code on Windows install the *WSL* extension and open the project folder inside WSL, then everything below is identical to Linux.
- **RAM** 16 GB or more, **free disk** about 40 GB (processed data about 7.6 GB, upsampled scenes for row H about 7 GB, checkpoints and test predictions 5–10 GB). Put `workspace/` on an SSD.
- `git`, `conda` (Miniconda is fine), VS Code with the *Python* and *Jupyter* extensions.
- A GPU newer than the RTX 40 series (for example RTX 50xx "Blackwell") may need a newer PyTorch/CUDA build than 2.5.0 + 12.4. The environment check in Part 2 section 1 tells you if the installed PyTorch has no kernels for your GPU; see section 9.

## 3. One-time setup

### 3.1 Create the environment

```bash
conda create -n pcseg python=3.10 -y
conda activate pcseg
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# optional, faster than the built-in replacement (skip it if it fails; the notebooks work without it):
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

### 3.2 Open the project in VS Code

1. *File → Open Folder* → the `pcseg_local` folder (open the **root** folder, not `notebooks/`, so that `pcseg/` is found).
2. Open a notebook from `notebooks/`, click the kernel picker (top right) and choose the **pcseg** conda environment.
3. Optional check without a GPU: `python tests/test_numpy_parts.py` (data handling, patch banks, scoring). With PyTorch: `python tests/test_torch_cpu.py` (upsampler end to end on tiny synthetic data, CPU).

### 3.3 Put the data in place

Part 1 is finished, so you only need its **processed backups** from Google Drive (`MyDrive/pcseg/backup/`), not the 31 GB raw backup:

| Drive folder | Copy to | Size |
|---|---|---|
| `backup/s3dis` | `workspace/data/s3dis` | 4.9 GB |
| `backup/s3dis_sparse25` | `workspace/data/s3dis_sparse25` | 1.8 GB |
| `backup/s3dis_sparse12` (optional, only for the 12.5 % ablation) | `workspace/data/s3dis_sparse12` | 0.9 GB |

(Run every command in this guide from the project root folder, `pcseg_local`.) Download with the Drive web page (it zips folders), or with `rclone` (`rclone copy gdrive:pcseg/backup/s3dis workspace/data/s3dis --progress`; many small files copy faster that way). To keep the data elsewhere, set the environment variable `PCSEG_DATA` before starting VS Code. Then check it:

```bash
python -m pcseg.data_check
```

Expected: six areas in each folder, `Area_5` with 68 rooms, a spot check that the 25 % scan is about 0.25 of the dense room and an exact subset of it, and `OK: data looks right`.

### 3.4 If you already finished something on Colab (optional)

Anything already finished can be reused, **provided the knobs are the same** (Colab values: batch 4, point_max 100000, epoch 100, eval_epoch 10; set them in Part 2 section 3):

- Copy Drive `results/*.json` to `workspace/results/`: finished rows are skipped.
- Copy `upsampler/*` to `workspace/upsampler/` and `banks/*.npz` to `workspace/banks/`: finished upsamplers and banks are skipped.
- For a finished PTv3 run copy `exp/<run>/config.py` and `exp/<run>/model/model_last.pth` to `workspace/exp/<run>/` and mark it done once: `python -c "from pcseg import run; run.mark_done('A_dense')"`. Testing from such a copy normally works because the data paths are overridden at test time; if it does not, retrain that run.
- An unfinished Colab run cannot be resumed locally (its config contains Colab paths); start it again.

If you start fresh, ignore this section.

## 4. Order of work at a glance

The upsampler is small and independent of the long PTv3 trainings, and Phase 0 decides whether the whole study makes sense, so the order is: set up, measure, decide, build the upsampler, then spend the long GPU hours.

| Step | Notebook, sections | Needs | Result |
|---|---|---|---|
| 1 | Part 2: 0, 1, 2, 3 | GPU | Environment and data verified; PTv3 forward pass passes |
| 2 | Part 2: 4 (benchmark) | GPU | Seconds per iteration; you choose `epoch` |
| 3 | Part 2: 5 (Phase 0 gate) | GPU | Does sparsity hurt PTv3? Go or change plan |
| 4 | Part 3: 0, 1 (banks) | CPU | Patch banks in `workspace/banks` |
| 5 | Part 3: 2, 3, 4, 7 | GPU | Upsampler trained and measured |
| 6 | Part 2: 6, 7, 8 | GPU (long) | Rows A, C, B, D |
| 7 | Part 4: 0, 1, 2 | GPU | Rows E and G |
| 8 | Part 3: 5, 6 (optional) | GPU | Cross-fit folds, ×2 and 12.5 % models |
| 9 | Part 4: 3, 4, 5, 6, 7 | GPU (long) | Row H, tables, ablations, cost, figures |

Steps 6 and 9 are the long ones. Everything is resumable. Run cells one at a time, not "Run All".

## 5. Step by step

### Step 1. Part 2, sections 0–3 (about 15 minutes, no long downloads)

1. **Section 0** (two cells). Expect torch version, CUDA version, your GPU name and VRAM, the workspace path, free disk, then the data check ending in `OK: data looks right`.
2. **Section 1** (environment check). Expect `python 3.10.x | torch 2.5.0 | cuda 12.4 | gpu available: True`, your GPU, `spconv OK -> (N, 16)`, one `torch_scatter` line, a table of fp16/fp32 spconv tests, and a suggestion of starting knobs for your VRAM. The result (`AMP: True/False`) is stored in the settings file. If fp16 is not supported on your GPU, training runs in fp32 (slower, more memory).
3. **Section 2** (Pointcept, two cells). The first time it clones Pointcept at the pinned commit (needs internet and git). Expect `Pointcept 1342eda (...)`, `RandomSparsify patch applied: True`, a `stubbed / shimmed` list, then four `ok pointcept...` lines. A line about PointROPE using a slower PyTorch fallback is harmless.
4. **Section 3** (knobs and configs, two cells). Set the knobs (section 6 of this guide), run the cell: it prints the room counts (about 200 training rooms, 68 for Area 5), the config names and the iterations per run. The next cell is the PTv3 forward smoke test: expect `PTv3 forward OK: logits (20000, 13)`.

### Step 2. Part 2, section 4: benchmark (about 10 minutes)

Trains for one short pass (about 50 iterations) and prints `X s/iteration` plus a table of hours per run for several `epoch` values. If it fails with *CUDA out of memory*, lower `point_max` (then `batch`) in section 3, re-run section 3 and the benchmark. Choose `epoch` (section 6 below), write it into the section 3 knobs and re-run that cell.

### Step 3. Part 2, section 5: Phase 0 gate

Trains two short models (20 passes each), tests them, and prints `dense->dense`, `dense->sparse25`, `sparse->sparse25` and the mIoU drop caused by sparsity. Absolute numbers are low because the schedule is short; only the drop matters.

| Drop (dense→dense minus dense→sparse25) | What to do |
|---|---|
| 3 points or more | Sparsity clearly hurts. Continue with 25 % as planned |
| 1.5 to 3 points | Continue, but consider 12.5 % as the main setting (its voxel occupancy is only 0.38). Tell me before step 4 and I will adapt the notebooks |
| Below 1.5 points | The segmenter barely notices 25 % sparsity. Do not build the study on it: switch the main setting to 12.5 % or coarser. Tell me and I will adapt the notebooks |

Your Part 1 statistics already predict a visible but moderate effect: voxel occupancy at PTv3's 2 cm grid is 0.62 for 25 % and 0.38 for 12.5 %, and 0.94–0.96 at 4–5 cm, so deeper stages partly hide sparsity.

### Step 4. Part 3, sections 0–1: patch banks (20–40 minutes of CPU)

Builds the development bank (Areas 1–4, about 100 patches per room) and the validation bank (Area 6), printing one line per room. A bank that exists is skipped; **do not stop a bank midway**, an interrupted bank restarts from the beginning (files are written atomically, so a stopped run never leaves a broken bank).

### Step 5. Part 3, sections 2, 3, 4, 7: train and check the upsampler (GPU, minutes)

1. **Section 2.** Trains the development model for 30 epochs. First it prints the naive-interpolation validation metrics, then one JSON line per epoch, and finally `best val CD (x1e3): X | interpolation: Y`. Check that X is clearly lower than Y. The next cell saves a three-panel curve figure to `workspace/figures/`.
2. **Section 3.** Three short trainings (Chamfer only, plus repulsion, plus uniformity) and a comparison table. Keep it for the report.
3. **Section 4.** Builds the full bank (Areas 1,2,3,4,6) and trains the final ×4 model `upsampler/full_x4`, used for the Area 5 test scenes.
4. **Section 7.** Compares sparse, naive interpolation and the upsampler on five Area 5 rooms in centimetres and writes `results/upsampler_scene_metrics.json` (needed later for the cost table). How to read it: the sparse input consists of real points, so its precision is perfect and its Chamfer can look competitive; the fair comparison is learned versus interpolation, and the F-score and the segmentation rows decide the rest.

If the learned model does not beat interpolation on validation, stop and send me `workspace/upsampler/dev_x4/log.json`.

### Step 6. Part 2, sections 6, 7, 8: baselines A, C, B, D (the long step)

| Section | Cell does | Output |
|---|---|---|
| 6 | Trains `A_dense`, evaluates row A (dense) and row C (the same checkpoint on 25 % sparse data) | `A_dense_on_dense.json`, `C_dense_model_on_sparse25.json` |
| 7 | Trains `B_sparse25`, evaluates row B | `B_sparse25_on_sparse25.json` |
| 8 | Trains `D_dense_aug`, evaluates row D | `D_dense_aug_on_sparse25.json` |
| 9 | Prints the table of all results so far | markdown table |

While a cell trains it prints a few log lines every minute; the checkpoint is written after each of the `eval_epoch` passes. If you stop the cell (or the machine), run the same cell again: it continues from the last checkpoint, and a finished run is skipped. Sanity checks: A should be the highest, C below A, and mIoU well above chance (if A is below about 30 mIoU, send me the end of `workspace/exp/A_dense/train.log`; a schedule that is far too short is the usual cause).

To let the long runs go on overnight without a notebook open (Linux/WSL):

```bash
conda activate pcseg
nohup python scripts/run_ptv3.py A B D > workspace/run_ptv3.log 2>&1 &
tail -f workspace/run_ptv3.log
```

It uses the same knobs and the same resume logic. Use `nvidia-smi` to watch GPU use; if the GPU is often idle, raise `num_worker` in the knobs.

### Step 7. Part 4, sections 0, 1, 2: rows E and G (about 1–3 hours)

Section 0 checks the environment, the data and that `A_dense` and `full_x4` exist. Section 1 writes the interpolated and upsampled Area 5 scenes (`data/s3dis_interp25x4`, `data/s3dis_up25x4`); finished rooms are skipped. Section 2 tests the dense-trained PTv3 on both and prints `E (interpolation): ... | G (our upsampler): ...`.

How to read E and G against C: G above C and E means learned geometry helps; E and G both above C means more points help; G equal to C means no benefit at this sparsity.

### Step 8. Part 3, sections 5 and 6 (optional but needed for row H and some ablations)

Section 5 trains the two cross-fit folds (needed for row H). Section 6 trains the ×2 model and, if `s3dis_sparse12` exists, the 12.5 % ×8 model. Both skip what already exists.

### Step 9. Part 4, sections 3–7: row H, tables, ablations, cost, figures

- **Section 3 (row H).** Upsamples the training areas with the fold models (the heaviest part: about 200 rooms; check free disk, the output is large), then trains `H_up_trained` with the same knobs as Part 2 (read from the settings file) and evaluates it. For a cheaper run set `"stride": 2` in `workspace/pcseg_settings.json` (every second room); then H sees fewer rooms than A, B and D, and you must say so.
- **Section 4.** Main results table, per-class table and bar chart (`figures/main_results.png`, `results/main_results.md`).
- **Section 5.** Ablations; each needs one upsampling pass and one test run, so run only what you will report. Finished ones are skipped.
- **Section 6.** Cost table (upsampler parameters, seconds per room, PTv3 test seconds).
- **Section 7.** Qualitative figure (dense, sparse, upsampled, prediction, ground truth, errors) in `figures/`.

## 6. Choosing the training budget

Iterations per run = number of training rooms × `epoch` / `batch` (about 203 rooms, so with batch 4 about 51 × `epoch`). Hours per run = iterations × seconds per iteration / 3600, with the seconds from the benchmark (Part 2 section 4 prints the table for you).

Starting values for `batch` and `point_max` by VRAM (the notebook prints the matching line for your GPU). They are **estimates scaled from the 16 GB T4 setting, not measurements**; the benchmark decides.

| VRAM | batch | point_max |
|---|---|---|
| under 10 GB | 2 | 60 000 |
| 10–14 GB | 4 | 60 000 |
| 14–22 GB | 4 | 100 000 |
| 22–40 GB | 8 | 100 000 |
| 40 GB and more | 12 (the paper's) | 204 800 (the paper's) |

Rules of thumb: pick the largest `epoch` for which one run fits in about 2–6 hours; `epoch` must be a multiple of `eval_epoch` (10); a longer schedule helps all variants equally, so it raises absolute accuracy without changing the comparisons. You need four full runs (A, B, D, H) plus the two short Phase 0 runs. The learning rate follows the batch size automatically (0.006 × batch / 12). Use the **same** knobs for every row: changing `epoch`, `batch` or `point_max` halfway makes rows incomparable. With a bigger GPU than the T4 your absolute numbers will differ from any Colab results; do not mix rows from the two setups in one table.

## 7. What is saved where (under `workspace/`)

| Path | Content |
|---|---|
| `data/` | processed S3DIS (from Part 1) and the upsampled scenes written by Part 4 |
| `Pointcept/` | Pointcept checkout (pinned commit) with the `RandomSparsify` patch |
| `exp/<run>/` | `config.py`, `train.log`, `model/model_last.pth`, `train_done.json`, `test_<tag>/result/*_pred.npy` |
| `results/*.json` | one file per evaluated row (mIoU, OA, mAcc, per-class IoU, confusion matrix, test seconds) |
| `banks/*.npz` | upsampler patch banks |
| `upsampler/<model>/` | `best.pt`, `last.pt`, `log.json` per upsampler |
| `figures/` | plots and qualitative figures |
| `pcseg_settings.json` | the shared knobs (batch, point_max, epoch, eval_epoch, num_worker, amp, stride) |

The optional `bench` and `P0_*` runs can be deleted from `exp/` after Phase 0. After row H and the ablations you can delete `data/s3dis_up25x4` and `data/s3dis_interp25x4` to free space, but keep them until the figures are done.

To retrain a finished PTv3 run, delete its folder in `workspace/exp/` (and its `results/*.json` rows). To retrain an upsampler, delete its folder in `workspace/upsampler/` (or run `pcseg.pu_train` with `--force`).

## 8. Working habits that avoid lost work

- Run cells one at a time. A long training cell can be interrupted (stop button) at any time; run it again to resume.
- Do not interrupt bank building or upsampling mid-room (an interrupted bank restarts; upsampling skips finished rooms).
- After finishing a row, check that its JSON exists in `workspace/results/`; that is your proof the row is done.
- If VS Code or the kernel restarts you lose only Python variables: run the first (setup) cell of the notebook, then the section you were in. Part 2 sections 1–3 need not be repeated except the cell that defines `cfgs` (section 3) when you want to train or test in Part 2.
- Part 4 reads the knobs from the settings file; do not edit them between rows.
- On a laptop, keep it plugged in and disable sleep during long runs.

## 9. If something fails

Send me: the notebook name, the section and cell, and the last 30 lines of the error or of `workspace/exp/<run>/stdout.txt`. Typical cases:

| Symptom | What to do |
|---|---|
| `ModuleNotFoundError: pcseg` | The folder opened in VS Code is not the project root, or the kernel is not the `pcseg` environment |
| `no GPU visible to PyTorch` | Wrong kernel (CPU PyTorch), or the NVIDIA driver is missing/too old. Check `nvidia-smi` in the terminal |
| Warning that this PyTorch build has no kernels for `sm_XY`, or spconv fails on a very new GPU | The pinned PyTorch 2.5.0 + CUDA 12.4 wheels predate your GPU. You need a PyTorch build with CUDA 12.8 or newer and a matching spconv build; tell me the GPU name and the output of section 1 and I will adapt the setup |
| spconv import error | Make sure `spconv-cu124` is installed in this environment and that `pip list` shows no other `spconv-*` package |
| CUDA out of memory | Lower `point_max`, then `batch` (Part 2 section 3), re-run section 3 and the benchmark. Close other programs that use the GPU |
| Pointcept clone fails | The first run needs internet access to GitHub; or clone it yourself to `workspace/Pointcept` and `git checkout 1342eda` |
| Data loader crashes with shared-memory (`/dev/shm`) or "killed" errors, especially in WSL or Docker | Lower `num_worker` (to 2 or 0), or increase the WSL memory limit |
| The GPU is idle most of the time | Raise `num_worker`; keep the data on an SSD |
| A result breaks a sanity check from step 6 | Send me the log file named in that step |

When Phase 0 or the upsampler check says the plan needs to change, tell me the numbers and I will adapt the notebooks.

## 10. Changes to the code (for your records and the report)

The study itself (data, models, losses, schedules, metrics, scoring on the original dense points) is unchanged. Differences from the Colab notebooks:

- **Environment.** One conda environment instead of the `uv` Python 3.10 environment inside Colab. `torch-scatter` is used if installed, otherwise the same pure-PyTorch shim as before.
- **Configs from one function.** `pcseg/experiments.py` builds every config from the shared knobs. In the Colab Part 4 the row H config was created without the `amp` flag; it now always matches rows A, B and D.
- **Settings.** Paths and knobs are in `pcseg/settings.py` and `workspace/pcseg_settings.json`; `num_worker` is configurable (Colab used 2).
- **Finished runs are skipped** through a `train_done.json` marker next to the checkpoint; interrupted runs resume as before.
- **Patch banks** are written to a temporary file and renamed, so an interrupted run cannot leave a half-written bank.
- **`torch.load(..., weights_only=True)`** for upsampler checkpoints.
- **Visualisation palette:** beam and window had the same colour in the label figure; window now has its own colour.
- **New helpers:** `pcseg/data_check.py`, GPU-architecture warning in the environment check, VRAM-based knob suggestions, `scripts/run_ptv3.py` (headless long runs), `tests/`.

What I could and could not test while preparing this package: the data handling, patch banks, scoring, tables and settings were run on synthetic data (`tests/test_numpy_parts.py`), the notebooks were syntax-checked and scanned for undefined names, and the Pointcept entry points, output file names and config fields the code relies on were checked against the pinned Pointcept source. PyTorch, spconv and your GPU were **not** available to me, so the PTv3 training/testing and the upsampler training have not been run by me in this form; the first run on your machine (Part 2 sections 0–4) is the real test, and `tests/test_torch_cpu.py` is a quick check of the upsampler half.
