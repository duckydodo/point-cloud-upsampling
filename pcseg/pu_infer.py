"""Upsample every room of a sparse S3DIS copy and write a Pointcept-format dataset.
  --method model : learned upsampler (needs --ckpt)
  --method interp: naive midpoint interpolation baseline
"""
import argparse
import time

import torch

from pcseg.pu_core import load_upsampler
from pcseg.pu_data import interp_scene, upsample_scene, write_upsampled_dataset


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["model", "interp"], default="model")
    ap.add_argument("--ckpt")
    ap.add_argument("--sparse-root", required=True)
    ap.add_argument("--dense-root", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--areas", nargs="+", required=True)
    ap.add_argument("--label-areas", nargs="*", default=[], help="areas whose upsampled points get 1-NN labels from dense GT (training areas)")
    ap.add_argument("--up-ratio", type=int, default=4)
    ap.add_argument("--rgb-mode", choices=["nn", "idw"], default="nn")
    ap.add_argument("--include-input", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--room-stride", type=int, default=1, help="process every k-th room only (cheaper; must be stated in the report)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if a.method == "model":
        model = load_upsampler(a.ckpt, dev)
        assert model.r == a.up_ratio, f"checkpoint is x{model.r}, requested x{a.up_ratio}"
        fn = lambda xyz, rgb: upsample_scene(model, xyz, rgb, up_ratio=a.up_ratio, batch=a.batch, device=dev,
                                             rgb_mode=a.rgb_mode, include_input=a.include_input, seed=a.seed)
    else:
        fn = lambda xyz, rgb: interp_scene(xyz, rgb, a.up_ratio)
    t0 = time.time()
    write_upsampled_dataset(a.dst, a.sparse_root, a.areas, fn, dense_root=a.dense_root, label_areas=set(a.label_areas), room_stride=a.room_stride)
    print(f"done in {time.time() - t0:.0f}s -> {a.dst}")


if __name__ == "__main__":
    main()
