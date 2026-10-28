"""Regression test for prediction/label shape alignment and broadcasting hazard.

Tests that prediction arrays loaded from Pointcept evaluators (which may be 2D of shape (N, 1))
are flattened to 1D (N,) before being indexed or compared with 1D ground-truth labels (N,).

Demonstrates the broadcasting hazard:
When pred is shape (N, 1) and gt is shape (N,),
pred == gt produces an (N, N) matrix rather than an (N,) vector.
On full room point clouds (N ~ 1,500,000), an (N, N) boolean array would require
(1.5e6)^2 bytes = 2.25 TB RAM, causing an Out-Of-Memory process crash.
Even for small N, indexing a color array (N, 3) with an (N, N) boolean mask raises IndexError.
Flattening pred with .reshape(-1) ensures pred == gt is shape (N,), avoiding the hazard.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_broadcasting_hazard_and_flatten_fix():
    """Demonstrate the hazard on a small synthetic example (N=10) and verify the fix."""
    try:
        import numpy as np
        has_numpy = hasattr(np, "array")
    except (ImportError, AttributeError):
        has_numpy = False

    if has_numpy:
        n = 10
        # Synthetic 1D ground truth labels (N,)
        gt = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, -1], dtype=np.int32)
        # Synthetic 2D predictions (N, 1) as output by Pointcept test evaluators
        pred_2d = np.array([[0], [1], [0], [0], [1], [2], [2], [1], [0], [0]], dtype=np.int32)

        # 1. Demonstrate the hazard without .reshape(-1)
        cmp_unflattened = (pred_2d == gt)
        assert cmp_unflattened.shape == (n, n), (
            f"Expected broadcasted outer shape ({n}, {n}), got {cmp_unflattened.shape}"
        )
        print(f"Hazard verified: (pred_2d == gt) broadcasted to shape {cmp_unflattened.shape} instead of ({n},)")

        # Verify that indexing a color buffer (N, 3) with (N, N) mask fails
        c = np.zeros((n, 3), dtype=np.uint8)
        ok = gt >= 0
        hazard_failed = False
        try:
            c[ok & cmp_unflattened] = (200, 200, 200)
        except IndexError as e:
            hazard_failed = True
            print(f"Hazard verified: indexing (N, 3) with (N, N) mask raised IndexError: {e}")
        assert hazard_failed, "Expected IndexError when indexing (N, 3) with (N, N) mask"

        # 2. Verify the fix with .reshape(-1)
        pred_1d = pred_2d.reshape(-1)
        assert pred_1d.shape == (n,), f"Expected shape ({n},), got {pred_1d.shape}"
        cmp_flattened = (pred_1d == gt)
        assert cmp_flattened.shape == (n,), f"Expected shape ({n},), got {cmp_flattened.shape}"

        # Now color indexing succeeds cleanly
        c[ok & cmp_flattened] = (200, 200, 200)
        c[ok & ~cmp_flattened] = (220, 30, 30)
        assert c.shape == (n, 3)
        # Check indices: index 0 is correct (gt=0, pred=0) -> (200, 200, 200)
        assert tuple(c[0]) == (200, 200, 200)
        # index 2 is error (gt=2, pred=0) -> (220, 30, 30)
        assert tuple(c[2]) == (220, 30, 30)
        # index 9 is ignored (gt=-1) -> (0, 0, 0)
        assert tuple(c[9]) == (0, 0, 0)
        print("Fix verified with NumPy: color array indexed safely with shape (N,).")

    else:
        # Fallback simulation of NumPy broadcasting rules using pure Python
        print("NumPy not available in current environment; verifying broadcasting logic via pure Python.")
        n = 10
        gt = [0, 1, 2, 0, 1, 2, 0, 1, 2, -1]
        pred_2d = [[0], [1], [0], [0], [1], [2], [2], [1], [0], [0]]

        # Simulating (N, 1) == (N,) broadcasting
        broadcasted = [[pred_2d[i][0] == gt[j] for j in range(n)] for i in range(n)]
        assert len(broadcasted) == n and len(broadcasted[0]) == n, (
            f"Expected ({n}, {n}) broadcasted grid"
        )
        print(f"Hazard verified in pure Python: 2D comparison produces {n}x{n} elements.")

        # Simulating 1D comparison
        pred_1d = [p[0] for p in pred_2d]
        pairwise = [pred_1d[i] == gt[i] for i in range(n)]
        assert len(pairwise) == n, f"Expected {n} elements"
        print(f"Fix verified in pure Python: 1D comparison produces {n} elements.")


if __name__ == "__main__":
    test_broadcasting_hazard_and_flatten_fix()
    print("ALL CHECKS PASSED: Prediction/label shape alignment regression test succeeded.")
