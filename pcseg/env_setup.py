"""Environment checks and Pointcept preparation for a LOCAL machine.

Unlike the Colab version nothing is created here: you build one conda/venv environment once (README, step 2:
Python 3.10, PyTorch 2.5.0 + CUDA 12.4, spconv-cu124) and use it both as the notebook kernel and for Pointcept.
This module only verifies that the environment works and prepares the Pointcept checkout.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from pcseg import settings

PY = sys.executable
POINTCEPT_COMMIT = "1342eda"

CHECK = '''
import sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU visible to PyTorch"
p = torch.cuda.get_device_properties(0)
print("GPU", p.name, "| %.1f GB" % (p.total_memory / 2**30), "| compute capability %d.%d" % (p.major, p.minor))
arch = torch.cuda.get_arch_list()
if not any(a.endswith("%d%d" % (p.major, p.minor)) for a in arch):
    print("WARNING: this PyTorch build has kernels for", arch, "but not for sm_%d%d: CUDA code may fail on this GPU. A newer PyTorch/CUDA build is needed (see README, troubleshooting)." % (p.major, p.minor))
import spconv.pytorch as spconv
coords = torch.unique(torch.randint(0, 20, (2000, 3), device="cuda", dtype=torch.int32), dim=0)
idx = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32, device="cuda"), coords], 1)
x = spconv.SparseConvTensor(torch.randn(len(coords), 8, device="cuda"), idx, spatial_shape=[20, 20, 20], batch_size=1)
print("spconv OK ->", tuple(spconv.SubMConv3d(8, 16, 3, indice_key="t").cuda()(x).features.shape))
try:
    import torch_scatter
    print("torch_scatter (compiled) OK")
except Exception as e:
    print("torch_scatter not importable (" + type(e).__name__ + ") -> the pure-PyTorch shim will be used (works, slightly slower)")
'''

AMP_CHECK = '''
import json, torch, spconv.pytorch as spconv
res = {"gpu": torch.cuda.get_device_name(0), "cc": ".".join(map(str, torch.cuda.get_device_capability(0)))}


def case(cin, cout, k, dt, autocast=False):
    coords = torch.unique(torch.randint(0, 40, (6000, 3), device="cuda", dtype=torch.int32), dim=0)
    idx = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32, device="cuda"), coords], 1)
    feat = torch.randn(len(coords), cin, device="cuda", dtype=torch.float16 if (dt == "fp16" and not autocast) else torch.float32)
    x = spconv.SparseConvTensor(feat, idx, [40, 40, 40], 1)
    conv = spconv.SubMConv3d(cin, cout, k, indice_key="t").cuda()
    if dt == "fp16" and not autocast:
        conv = conv.half()
    try:
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            y = conv(x).features
        torch.cuda.synchronize()
        return True
    except Exception as e:
        return (str(e).strip().splitlines() or ["error"])[-1][:110]


for name, a in {"3->32 k5": (3, 32, 5), "32->32 k3": (32, 32, 3), "64->64 k3": (64, 64, 3), "128->128 k3": (128, 128, 3)}.items():
    res["fp32 " + name] = case(*a, "fp32")
    res["fp16 " + name] = case(*a, "fp16")
    res["autocast " + name] = case(*a, "fp16", autocast=True)
print("JSON" + json.dumps(res))
'''

IMPORT_TEST = ("import importlib\n"
               "for m in ['pointcept.datasets', 'pointcept.models', 'pointcept.engines.train', 'pointcept.engines.test']:\n"
               "    importlib.import_module(m); print('ok', m)\n")

SMOKE = '''
import torch, sys
sys.path.insert(0, r"__POINTCEPT__")
from pointcept.utils.config import Config
from pointcept.models import build_model
cfg = Config.fromfile(r"__CFG__")
model = build_model(cfg.model).cuda().eval()
n = 20000
coord = torch.rand(n, 3).cuda() * 4
data = dict(coord=coord, grid_coord=torch.div(coord, 0.02, rounding_mode="floor").int(),
            feat=torch.rand(n, 3).cuda(), offset=torch.tensor([n]).cuda())
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=__AMP__):
    out = model(data)
print("PTv3 forward OK (AMP=__AMP__): logits", tuple(out["seg_logits"].shape), "| params (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 1))
'''


def pointcept_env(pointcept_dir=None, stub_dir=None):
    pointcept_dir = str(pointcept_dir or settings.POINTCEPT)
    stub_dir = str(stub_dir or settings.STUBS)
    parts = [pointcept_dir, stub_dir] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    return {**os.environ, "PYTHONPATH": os.pathsep.join(parts)}


def _run(code, env=None, cwd=None):
    return subprocess.run([PY, "-c", code], capture_output=True, text=True, env=env, cwd=cwd)


def check_env():
    """Python / torch / CUDA / GPU / spconv checks (run in a subprocess so a crash cannot kill the notebook)."""
    v = sys.version_info
    if (v.major, v.minor) != (3, 10):
        print(f"NOTE: Python {v.major}.{v.minor}; Pointcept is tested with 3.10 (other versions often work, but if something fails, use 3.10).")
    r = _run(CHECK)
    print(r.stdout.strip())
    if r.returncode:
        print(r.stderr[-2500:])
        raise RuntimeError("environment check failed: see the message above and README, section 'Troubleshooting'")
    g = settings.gpu_info()
    if g:
        s = settings.suggest_knobs(g[1])
        print(f"Suggested starting knobs for {g[1]:.0f} GB: batch={s['batch']}, point_max={s['point_max']}  (estimates; the benchmark decides)")


def check_amp():
    """Does fp16 spconv work on this GPU? Returns True (train with AMP) or False (train in fp32: slower, more memory)."""
    r = _run(AMP_CHECK)
    line = [l for l in r.stdout.splitlines() if l.startswith("JSON")]
    if r.returncode or not line:
        print(r.stdout[-1500:], r.stderr[-2500:])
        raise RuntimeError("fp16 self-check crashed: see the message above")
    res = json.loads(line[-1][4:])
    print(f"GPU {res.pop('gpu')} (compute capability {res.pop('cc')})")
    for k, v in res.items():
        print(f"  {k:24s}", "OK" if v is True else "FAILED: " + str(v))
    fp32_ok = all(v is True for k, v in res.items() if k.startswith("fp32"))
    amp_ok = all(v is True for k, v in res.items() if not k.startswith("fp32"))
    if not fp32_ok:
        raise RuntimeError("even fp32 spconv fails on this GPU/build: see the table above and README, 'Troubleshooting'")
    print("=> AMP (fp16) works: training uses mixed precision" if amp_ok else
          "=> fp16 spconv is NOT supported here: training and testing run in fp32 (AMP = False). Slower and more GPU memory.")
    return amp_ok


def prepare_pointcept(pointcept_dir=None, stub_dir=None, commit=POINTCEPT_COMMIT):
    """Clone Pointcept (pinned commit), add RandomSparsify, write shims/stubs for whatever cannot be imported."""
    from pcseg.compat import install_shims, patch_pointcept
    pointcept_dir = Path(pointcept_dir or settings.POINTCEPT)
    if not pointcept_dir.is_dir():
        pointcept_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "https://github.com/Pointcept/Pointcept.git", str(pointcept_dir)], check=True)
        c = subprocess.run(["git", "-C", str(pointcept_dir), "checkout", "-q", commit], capture_output=True, text=True)
        if c.returncode:
            print("pinned commit not found, using latest main (configs may differ slightly)")
    log = subprocess.run(["git", "-C", str(pointcept_dir), "log", "-1", "--format=Pointcept %h (%ad)"], capture_output=True, text=True).stdout.strip()
    print(log)
    print("RandomSparsify patch applied:", patch_pointcept(pointcept_dir))
    print("stubbed / shimmed:", install_shims(stub_dir))


def import_test(pointcept_dir=None):
    r = _run(IMPORT_TEST, env=pointcept_env(pointcept_dir), cwd=str(pointcept_dir or settings.POINTCEPT))
    print(r.stdout.strip())
    if r.returncode:
        print(r.stderr[-3000:])
        raise RuntimeError("Pointcept import failed: see the message above")


def smoke_test(cfg_path, amp=True):
    """One real PTv3 forward pass on the GPU (checks spconv + shim + model build)."""
    code = SMOKE.replace("__POINTCEPT__", str(settings.POINTCEPT)).replace("__CFG__", str(cfg_path)).replace("__AMP__", str(bool(amp)))
    r = _run(code, env=pointcept_env(), cwd=str(settings.POINTCEPT))
    print(r.stdout[-800:])
    if r.returncode:
        print(r.stderr[-2500:])
        raise RuntimeError("PTv3 forward pass failed: see the message above")


def setup_all():
    """prepare_pointcept + import_test (what Part 4 needs)."""
    prepare_pointcept()
    import_test()
