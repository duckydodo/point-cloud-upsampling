# Repository layout and storage policy

This repository contains only reproducible project assets: source code, notebooks, tests, dependency pins, documentation, and small text metadata.

```
point-cloud-upsampling/
├── pcseg/          # project Python modules
├── notebooks/      # Parts 2–4 execution notebooks
├── scripts/        # command-line experiment helpers
├── tests/          # synthetic/smoke tests
├── docs/           # run guide and working notes
└── requirements.txt
```

## Keep in Git

- Python source, notebooks with cleared outputs, tests, and scripts.
- `requirements.txt`, configuration templates, and documentation.
- Small result summaries only when they contain no raw predictions, data, model weights, or personal credentials.

## Keep outside Git

All data and produced runtime material belongs under `workspace/`, which is ignored. Set `PCSEG_ROOT` to a local scratch directory when needed. In Colab, point it at a Drive-backed folder. Do not add S3DIS copies, Pointcept checkouts, patch banks, training logs, checkpoints, predictions, figures, or Weights & Biases runs to Git.

## Backup and branch rules

- `main` remains the stable, reviewable branch.
- `backup/pre-import-2026-10-28` is the local snapshot from before this import.
- Use a small branch such as `fix/<short-problem>` for every notebook or code correction.
- Commit only after a focused test or notebook cell has passed; push only after inspecting `git status` and `git diff --check`.
