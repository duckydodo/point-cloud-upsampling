"""Unit test for Pointcept configuration generation and hook filtering."""
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_make_config_hook_filtering():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pointcept_dir = tmp_path / "Pointcept"
        (pointcept_dir / "configs" / "s3dis").mkdir(parents=True)
        (pointcept_dir / "pointcept" / "utils").mkdir(parents=True)

        # Mock Pointcept Config class and base config file
        config_py = pointcept_dir / "pointcept" / "utils" / "config.py"
        config_py.write_text('''
class ConfigDict(dict):
    def __getattr__(self, name):
        v = self[name]
        if isinstance(v, dict):
            return ConfigDict(v)
        return v
    def __setattr__(self, name, value):
        self[name] = value

class Config:
    def __init__(self, d):
        self._d = ConfigDict(d)
    def __getattr__(self, name):
        return getattr(self._d, name)
    def __setattr__(self, name, value):
        if name == "_d":
            super().__setattr__(name, value)
        else:
            setattr(self._d, name, value)
    def __contains__(self, key):
        return key in self._d
    @staticmethod
    def fromfile(path):
        import ast
        with open(path) as f:
            code = f.read()
        ns = {}
        exec(code, ns)
        return Config(ns["cfg"])
    def dump(self, path):
        with open(path, "w") as f:
            f.write(repr(dict(self._d)))
''')

        base_cfg = pointcept_dir / "configs" / "s3dis" / "semseg-pt-v3m1-1-rpe.py"
        base_cfg.write_text('''
cfg = dict(
    model=dict(backbone=dict(in_channels=6)),
    data=dict(
        train=dict(data_root="", split=(), transform=[
            dict(type="SphereCrop", point_max=204800),
            dict(type="Collect", feat_keys=("color", "normal")),
            dict(type="GridSample")
        ]),
        val=dict(data_root="", split=(), transform=[dict(type="Collect", feat_keys=("color", "normal"))]),
        test=dict(data_root="", split=(), test_cfg=dict(
            post_transform=[dict(type="Collect", feat_keys=("color", "normal"))],
            aug_transform=[]
        ))
    ),
    optimizer=dict(lr=0.006),
    scheduler=dict(max_lr=[0.006, 0.0006]),
    param_dicts=[],
    hooks=[
        dict(type="CheckpointLoader"),
        dict(type="IterationTimer", warmup_iter=2),
        dict(type="InformationWriter"),
        dict(type="SemSegEvaluator"),
        dict(type="CheckpointSaver", save_freq=None),
        dict(type="PreciseEvaluator", test_last=False),
    ]
)
''')

        from pcseg.make_configs import make_config

        out_path = make_config(
            "bench",
            train_root=str(tmp_path / "data" / "s3dis"),
            test_root=str(tmp_path / "data" / "s3dis"),
            pointcept_dir=pointcept_dir,
            out_dir=pointcept_dir / "configs" / "s3dis",
            epoch=1,
            eval_epoch=1,
        )

        assert Path(out_path).exists(), f"Config file not written: {out_path}"
        content = Path(out_path).read_text()

        # Parse generated config content
        generated = eval(content)
        hook_types = [h["type"] for h in generated["hooks"]]

        assert "PreciseEvaluator" not in hook_types, f"PreciseEvaluator still present: {hook_types}"
        assert "SemSegEvaluator" not in hook_types, f"SemSegEvaluator still present: {hook_types}"
        assert "CheckpointSaver" in hook_types, f"CheckpointSaver missing: {hook_types}"
        assert "CheckpointLoader" in hook_types, f"CheckpointLoader missing: {hook_types}"
        assert "InformationWriter" in hook_types, f"InformationWriter missing: {hook_types}"
        assert generated["evaluate"] is False, "evaluate flag should be False"

        print("TEST PASSED: make_config hook filtering verified.")
        print("Generated hooks:", hook_types)


if __name__ == "__main__":
    test_make_config_hook_filtering()
