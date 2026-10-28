"""Local launcher for Pointcept training/testing with resume.

Layout:  <workspace>/exp/<name>/config.py, train.log, model/model_last.pth, test_<tag>/result/*_pred.npy
Training is resumable: run the same cell/command again and it continues from model_last.pth
(Pointcept writes a checkpoint after every one of the `eval_epoch` passes).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pcseg import settings
from pcseg.env_setup import pointcept_env

PYTHON = sys.executable
NUM_GPUS = 1


def _opts(d):
    return [f"{k}={v}" for k, v in d.items()]


# ----------------------------------------------------------------------------- generic subprocess helpers
def sh(cmd, cwd=None, env=None, tail=None):
    """Run a command, streaming its output live (or only the last `tail` lines if tail is set). Raises on failure."""
    cmd = [str(c) for c in cmd]
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = []
    for line in p.stdout:
        if tail:
            last = (last + [line.rstrip()])[-tail:]
        else:
            print(line, end="", flush=True)
    rc = p.wait()
    if tail and last:
        print("\n".join(last))
    if rc != 0:
        raise RuntimeError(f"command failed (exit code {rc}): {' '.join(cmd[:6])} ...")
    return rc


def py_module(module, *args, tail=None):
    """python -m <module> <args> with the current interpreter (used for pcseg.pu_train / pcseg.pu_infer)."""
    return sh([PYTHON, "-m", module, *args], cwd=str(settings.PROJECT), tail=tail)


# ----------------------------------------------------------------------------- training
def train_cmd(name, config_path, options=None, resume=True, exp=None):
    exp = Path(exp or settings.EXP)
    options = dict(options or {})
    save = exp / name
    last = save / "model" / "model_last.pth"
    if resume and last.exists():
        cfg = str(save / "config.py")
        options.update(save_path=str(save), resume=True, weight=str(last))
    else:
        cfg = str(config_path)
        options.update(save_path=str(save))
    return [PYTHON, "tools/train.py", "--config-file", cfg, "--num-gpus", str(NUM_GPUS), "--options", *_opts(options)]


def _stream(proc, logf, every):
    pos, t_print = 0, 0
    while proc.poll() is None:
        time.sleep(5)
        if time.time() - t_print > every:
            t_print = time.time()
            try:
                with open(logf) as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                lines = [l for l in chunk.splitlines() if l.strip()]
                if lines:
                    print("\n".join(lines[-3:]), flush=True)
            except FileNotFoundError:
                pass
    return proc.returncode


def train(name, config_path, options=None, resume=True, print_every=60, dry_run=False):
    """Run tools/train.py (resumes from <exp>/<name>/model/model_last.pth). A finished run is skipped (train_done.json)."""
    save = settings.EXP / name
    done = save / "train_done.json"
    if resume and done.exists():
        print(f"{name}: already trained (delete {done} or the folder {save} to retrain)")
        return 0
    cmd = train_cmd(name, config_path, options, resume)
    print("$", " ".join(cmd))
    if dry_run:
        return cmd
    save.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(save / "stdout.txt", "a") as out:
        proc = subprocess.Popen(cmd, cwd=str(settings.POINTCEPT), env=pointcept_env(), stdout=out, stderr=subprocess.STDOUT)
        try:
            rc = _stream(proc, str(save / "train.log"), print_every)
        except KeyboardInterrupt:           # interrupting the cell stops training; run it again to resume
            proc.terminate()
            proc.wait()
            print("\ninterrupted: run the same cell again to resume from the last checkpoint")
            raise
    if rc != 0:
        print("".join(open(save / "stdout.txt").readlines()[-40:]))
        raise RuntimeError(f"training exited with code {rc}")
    done.write_text(json.dumps(dict(seconds_this_session=time.time() - t0, finished=time.strftime("%Y-%m-%d %H:%M:%S"))))
    return rc


# ----------------------------------------------------------------------------- testing + scoring
def test_cmd(name, tag, config_path, weight, test_root, split="Area_5", options=None, exp=None):
    save = Path(exp or settings.EXP) / name / f"test_{tag}"
    o = dict(options or {})
    o.update(save_path=str(save), weight=str(weight), **{"data.test.data_root": str(test_root), "data.test.split": split})
    return [PYTHON, "tools/test.py", "--config-file", str(config_path), "--num-gpus", str(NUM_GPUS), "--options", *_opts(o)], save / "result"


def test(name, tag, test_root, weight=None, config_path=None, split="Area_5", options=None, dry_run=False):
    """Run tools/test.py on test_root/<split>; returns the folder with <Area>-<room>_pred.npy files."""
    save = settings.EXP / name
    config_path = config_path or save / "config.py"
    weight = weight or save / "model" / "model_last.pth"
    cmd, res = test_cmd(name, tag, config_path, weight, test_root, split, options)
    print("$", " ".join(cmd))
    if dry_run:
        return cmd, res
    res.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(save / f"test_{tag}.stdout.txt", "w") as out:
        rc = subprocess.run(cmd, cwd=str(settings.POINTCEPT), env=pointcept_env(), stdout=out, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print("".join(open(save / f"test_{tag}.stdout.txt").readlines()[-40:]))
        raise RuntimeError(f"testing exited with code {rc}")
    (res.parent / "seconds.txt").write_text(str(time.time() - t0))
    return res


def evaluate_run(name, tag, test_root, label, dense_root=None, weight=None, config_path=None,
                 results_dir=None, options=None, split="Area_5", reuse=True):
    """tools/test.py on test_root, then score on the ORIGINAL dense points. Writes <results>/<label>.json
    (skipped when that file already exists)."""
    from pcseg import eval_dense
    from pcseg.common import write_json
    dense_root = Path(dense_root or settings.DENSE)
    out_json = Path(results_dir or settings.RESULTS) / f"{label}.json"
    if reuse and out_json.exists():
        print("already evaluated:", out_json)
        return json.load(open(out_json))
    res = settings.EXP / name / f"test_{tag}" / "result"
    n_rooms = len(list((dense_root / split).iterdir()))
    if not (res.exists() and len(list(res.glob("*_pred.npy"))) >= n_rooms):
        res = Path(test(name, tag, test_root, weight, config_path, split, options))
    sec = res.parent / "seconds.txt"
    r = eval_dense.evaluate(res, test_root, dense_root, split, out_json=None)
    r["ptv3_test_seconds"] = float(sec.read_text()) if sec.exists() else None
    r["label"], r["experiment"], r["test_root"] = label, name, str(test_root)
    write_json(out_json, r)
    return r


def last_speed(name):
    """Average seconds per training iteration from Pointcept's train.log (None if not available yet)."""
    import re
    f = settings.EXP / name / "train.log"
    if not f.exists():
        return None
    m = re.findall(r"Batch ([\d.]+) \(([\d.]+)\)", f.read_text())
    return float(m[-1][1]) if m else None


def mark_done(name):
    """Mark a PTv3 run as finished (use after copying a finished checkpoint, e.g. from Colab, into exp/<name>/)."""
    d = settings.EXP / name
    assert (d / "model" / "model_last.pth").exists() and (d / "config.py").exists(), f"{d}: needs config.py and model/model_last.pth"
    (d / "train_done.json").write_text(json.dumps(dict(note="marked done manually")))
