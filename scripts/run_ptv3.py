"""Headless (no notebook) runner for the long PTv3 jobs of Part 2. Resumable: run it again after any interruption.

    python scripts/run_ptv3.py A B D          # rows A+C, B, D (any subset, in this order of priority)
    python scripts/run_ptv3.py H              # row H (needs Part 4 sections 1-3 up to the upsampled training scenes)

Uses the budget in workspace/pcseg_settings.json (set it in Part 2 section 3). Tip on Linux/WSL:
    nohup python scripts/run_ptv3.py A B D > workspace/run_ptv3.log 2>&1 &
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcseg import env_setup, experiments, run, settings


def main(rows):
    settings.ensure_dirs()
    env_setup.prepare_pointcept()
    K = settings.load_knobs()
    print("knobs:", K)
    D, S25 = str(settings.DENSE), str(settings.SP25)
    cfgs = experiments.make_all(K)
    for row in rows:
        if row == "A":
            run.train("A_dense", cfgs["A_dense"])
            run.evaluate_run("A_dense", "dense", D, "A_dense_on_dense")
            run.evaluate_run("A_dense", "sparse25", S25, "C_dense_model_on_sparse25")
        elif row == "B":
            run.train("B_sparse25", cfgs["B_sparse25"])
            run.evaluate_run("B_sparse25", "sparse25", S25, "B_sparse25_on_sparse25")
        elif row == "D":
            run.train("D_dense_aug", cfgs["D_dense_aug"])
            run.evaluate_run("D_dense_aug", "sparse25", S25, "D_dense_aug_on_sparse25")
        elif row == "H":
            run.train("H_up_trained", experiments.make_H(K))
            run.evaluate_run("H_up_trained", "up25x4", str(settings.UP25X4), "H_trained_on_upsampled")
        else:
            raise SystemExit(f"unknown row {row!r}: use A, B, D or H")
    print("done:", " ".join(rows))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
