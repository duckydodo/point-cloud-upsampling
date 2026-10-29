"""Compatibility layer so Pointcept's PTv3 runs without compiled pointops / torch_scatter.

* torch_scatter  -> pure-PyTorch shim (segment_csr, segment_coo, scatter, scatter_min/max/mean/sum), ONLY if the real
                    package cannot be imported (pip install torch-scatter, see README, is faster and optional).
* pointops, open3d, torch_cluster, peft, transformers -> inert stubs, ONLY if the package cannot be imported
  (never stub a package Pointcept guards with try/except ImportError, e.g. ocnn: a stub breaks its fallback);
  PTv3 never calls them, but Pointcept imports them at module level.
* transform patch -> registers RandomSparsify (density augmentation) inside pointcept/datasets/transform.py
"""
import subprocess
import sys
from pathlib import Path

from pcseg import settings

TORCH_SCATTER_SHIM = '''
"""Pure PyTorch replacement for the parts of torch_scatter used by Pointcept (dim=0 only)."""
import torch

_REDUCE = {"sum": "sum", "add": "sum", "mean": "mean", "max": "amax", "min": "amin"}


def scatter(src, index, dim=0, out=None, dim_size=None, reduce="sum"):
    assert dim == 0, "shim supports dim=0 only"
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    shape = (dim_size,) + tuple(src.shape[1:])
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    base = torch.zeros(shape, dtype=src.dtype, device=src.device) if out is None else out
    return base.scatter_reduce(0, idx, src, reduce=_REDUCE[reduce], include_self=out is not None)


def scatter_sum(src, index, dim=0, out=None, dim_size=None):
    return scatter(src, index, dim, out, dim_size, "sum")


def scatter_mean(src, index, dim=0, out=None, dim_size=None):
    return scatter(src, index, dim, out, dim_size, "mean")


def scatter_max(src, index, dim=0, out=None, dim_size=None):
    v = scatter(src, index, dim, out, dim_size, "max")
    return v, torch.zeros_like(v, dtype=torch.long)  # arg indices are not provided by the shim


def scatter_min(src, index, dim=0, out=None, dim_size=None):
    v = scatter(src, index, dim, out, dim_size, "min")
    return v, torch.zeros_like(v, dtype=torch.long)  # arg indices are not provided by the shim


def scatter_add(src, index, dim=0, out=None, dim_size=None):
    return scatter(src, index, dim, out, dim_size, "sum")


def scatter_softmax(src, index, dim=0, dim_size=None):
    assert dim == 0
    n = int(index.max().item()) + 1 if dim_size is None else dim_size
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    mx = scatter(src, index, 0, None, n, "max")
    e = (src - mx.gather(0, idx)).exp()
    den = scatter(e, index, 0, None, n, "sum")
    return e / den.gather(0, idx).clamp_min(1e-16)


def segment_csr(src, indptr, out=None, reduce="sum"):
    lengths = indptr[1:] - indptr[:-1]
    n_seg = lengths.numel()
    seg = torch.repeat_interleave(torch.arange(n_seg, device=src.device), lengths)
    return scatter(src, seg, 0, out, n_seg, reduce)


def segment_coo(src, index, out=None, dim_size=None, reduce="sum"):
    return scatter(src, index, 0, out, dim_size, reduce)
'''

GENERIC_STUB = '''
"""Inert stub for a module that Pointcept imports at top level but PTv3 never uses."""
def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    def _unavailable(*a, **k):
        raise RuntimeError("stubbed module function called: %s" % name)
    return _unavailable
'''

STUB_NAMES = ["pointops", "open3d", "torch_cluster", "peft", "transformers"]   # only the modules Pointcept imports unguarded and never calls

RANDOM_SPARSIFY = '''

# ---- pcseg patch: density augmentation (appended by compat.patch_pointcept) ----
@TRANSFORMS.register_module()
class RandomSparsify(object):
    """Randomly keep a fraction of points (uniform in keep_range) with probability p."""

    def __init__(self, keep_range=(0.1, 0.5), p=0.8):
        self.keep_range = keep_range
        self.p = p

    def __call__(self, data_dict):
        if random.random() < self.p:
            n = len(data_dict["coord"])
            keep = random.uniform(self.keep_range[0], self.keep_range[1])
            idx = np.random.choice(n, max(1, int(n * keep)), replace=False)
            data_dict = index_operator(data_dict, idx)
        return data_dict
'''


def _unimportable(python, names):
    """Names that cannot be imported by the given interpreter (a real import attempt, so a broken wheel counts as missing)."""
    code = ("import importlib, sys\n"
            "bad = []\n"
            "for n in sys.argv[1:]:\n"
            "    try:\n"
            "        importlib.import_module(n)\n"
            "    except Exception:\n"
            "        bad.append(n)\n"
            "print(' '.join(bad))\n")
    r = subprocess.run([python or sys.executable, "-c", code, *names], capture_output=True, text=True)
    return r.stdout.split()


def install_shims(stub_dir=None, python=None):
    """Write stubs/shims for modules the given interpreter cannot import (default: the current one).
    stub_dir goes on PYTHONPATH when Pointcept is launched. Returns the list of stubbed names."""
    stub_dir = Path(stub_dir or settings.STUBS)
    stub_dir.mkdir(parents=True, exist_ok=True)
    for f in stub_dir.glob("*.py"):          # start clean: a stub must never shadow a package that works
        f.unlink()
    stubbed = []
    bad = set(_unimportable(python, ["torch_scatter"] + STUB_NAMES))
    if "torch_scatter" in bad:
        (stub_dir / "torch_scatter.py").write_text(TORCH_SCATTER_SHIM)
        stubbed.append("torch_scatter (pure-torch shim)")
    for name in STUB_NAMES:
        if name in bad:
            (stub_dir / f"{name}.py").write_text(GENERIC_STUB)
            stubbed.append(name)
    return stubbed


SPHERE_CROP_ALL = '''

# ---- pcseg patch: SphereCrop mode="all" for evaluation fragmentation ----
@TRANSFORMS.register_module(name="SphereCrop", force=True)
class SphereCrop(object):
    """Evaluation fragmentation crop: guarantees 100% point coverage in sphere crops <= point_max."""

    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all", "given"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "coord" in data_dict.keys()
        if self.mode == "all":
            if "index" not in data_dict:
                data_dict["index"] = np.arange(data_dict["coord"].shape[0])
            if data_dict["coord"].shape[0] <= point_max:
                return [data_dict]

            coord = data_dict["coord"]
            n = coord.shape[0]
            uncovered = np.ones(n, dtype=bool)
            coord_p = np.zeros(n, dtype=np.float32)
            data_part_list = []

            if "index_valid_keys" in data_dict and "index" not in data_dict["index_valid_keys"]:
                data_dict["index_valid_keys"] = list(data_dict["index_valid_keys"]) + ["index"]

            while uncovered.any():
                uncovered_idx = np.where(uncovered)[0]
                init_idx = uncovered_idx[np.argmin(coord_p[uncovered_idx])]
                dist2 = np.sum(np.square(coord - coord[init_idx]), axis=1)
                idx_crop = np.argsort(dist2)[:point_max]

                uncovered[idx_crop] = False
                max_dist2 = dist2[idx_crop].max()
                if max_dist2 > 0:
                    delta = np.square(1.0 - dist2[idx_crop] / max_dist2)
                    coord_p[idx_crop] += delta

                data_crop_dict = index_operator(data_dict, idx_crop, duplicate=True)
                data_crop_dict["index"] = data_dict["index"][idx_crop]
                data_part_list.append(data_crop_dict)

            return data_part_list

        elif data_dict["coord"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["coord"][
                    np.random.randint(data_dict["coord"].shape[0])
                ]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            elif self.mode == "given":
                given_index = data_dict["correspondence"].reshape(
                    data_dict["correspondence"].shape[0], -1
                )
                given_index = np.all(
                    given_index != np.ones_like(given_index[0]) * -1, axis=1
                )
                given_coord = data_dict["coord"][given_index]
                if given_coord.shape[0] == 0:
                    center = data_dict["coord"][
                        np.random.randint(data_dict["coord"].shape[0])
                    ]
                else:
                    center = np.mean(given_coord, axis=0)
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[
                :point_max
            ]
            data_dict = index_operator(data_dict, idx_crop)
        return data_dict
'''


def patch_pointcept(pointcept_dir=None):
    """Idempotently append RandomSparsify and SphereCrop (mode='all') to Pointcept's transform module."""
    f = Path(pointcept_dir or settings.POINTCEPT) / "pointcept" / "datasets" / "transform.py"
    if not f.exists():
        return False
    text = f.read_text()
    patched = False
    if "class RandomSparsify" not in text:
        text += RANDOM_SPARSIFY
        patched = True
    if 'name="SphereCrop", force=True' not in text:
        text += SPHERE_CROP_ALL
        patched = True
    if patched:
        f.write_text(text)
    return patched

