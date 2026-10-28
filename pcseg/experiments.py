"""All experiment configs in one place, built from the shared knobs (pcseg_settings.json).
Using one function for every notebook guarantees that rows A, B, D and H use identical budgets."""
from pcseg import settings
from pcseg.make_configs import make_config


def mk(name, train_root, test_root, knobs=None, **kw):
    k = knobs or settings.load_knobs()
    p = dict(batch_size=k.batch, point_max=k.point_max, epoch=k.epoch, eval_epoch=k.eval_epoch, num_worker=k.num_worker, amp=k.amp)
    p.update(kw)
    return make_config(name, train_root, test_root, **p)


def make_all(knobs=None):
    """Config files for smoke test, benchmark and rows A, B, D (H is added in Part 4). Returns {name: config path}."""
    D, S25 = settings.DENSE, settings.SP25
    c = {}
    c["smoke"] = mk("smoke", D, D, knobs)
    c["bench"] = mk("bench", D, D, knobs, epoch=1, eval_epoch=1)
    c["A_dense"] = mk("A_dense", D, D, knobs)                                   # A: train dense   -> test dense
    c["B_sparse25"] = mk("B_sparse25", S25, S25, knobs)                         # B: train sparse  -> test sparse
    c["D_dense_aug"] = mk("D_dense_aug", D, S25, knobs, sparsify_aug=True)      # D: dense + random density augmentation -> test sparse
    return c


def make_H(knobs=None):
    """Row H: train on the upsampled training scenes, test on the upsampled Area 5 (same knobs as A, B, D)."""
    return mk("H_up_trained", settings.UP25X4, settings.UP25X4, knobs)
