# Colab debugging workflow

## 1. Capture a reproducible problem

Before changing code, record the notebook filename, section/cell purpose, full traceback, the command or cell input, and whether the failure occurs after a fresh runtime restart. Never send Drive links, tokens, or private data to an AI tool.

Run the smallest diagnostic first: verify the active Python version, PyTorch/CUDA availability, the `pcseg` import, and the expected Drive/data path. Keep the notebook and local repository on the same Git commit before comparing behavior.

## 2. Use the two AI tools as reviewers, not blind editors

Use Gemini 3.8 Flash in Antigravity for a fast first triage. Give it the traceback, the failing cell, the directly imported local functions, environment versions, and the intended behavior. Ask it for (a) likely cause ranked by confidence, (b) the smallest change, (c) a test, and (d) risks to experiment comparability.

Use Gemini 4 Argon for a second opinion when the fix changes environment setup, data paths, Pointcept integration, metrics, training configuration, or more than one file. Ask it to review the proposed patch rather than rewrite the notebook. Treat suggestions as hypotheses until tested.

## 3. Apply and test a small fix locally

1. Create `fix/<short-problem>` from the known-good branch.
2. Make one focused change. Do not edit data, checkpoints, results, or shared training knobs as part of an import/path fix.
3. Inspect `git diff` and run `git diff --check`.
4. Run the smallest relevant check first: `python tests/test_numpy_parts.py`, then `python tests/test_torch_cpu.py` when PyTorch is available, then just the affected Colab cell.
5. Compare only the intended behavior: import succeeds, data check passes, smoke test completes, or the original error no longer occurs.
6. Commit with the error and validation in the message, for example: `fix: resolve pcseg import in Colab`.

## 4. Promote to Colab carefully

Push the small tested commit, pull it in Colab, and verify the commit hash there. Run setup cells in order, then the one repaired cell. If the change affects batch size, point limit, schedule, sparsity, or evaluation, record it as a new experiment setting; do not mix those results with earlier rows.

## Prompt template

```
I am debugging a point-cloud upsampling/segmentation notebook.
Goal of this cell: <goal>
Environment: Python <version>, torch <version>, CUDA <version>, GPU <name>
Exact traceback: <paste>
Relevant cell and imported functions: <paste>
Constraints: do not change dataset, checkpoints, metrics, or shared training knobs.

Please rank likely causes. Propose the smallest patch, explain why it works,
give one minimal validation command/cell, and flag any threat to reproducibility.
Do not invent APIs or claim the training result has improved.
```
