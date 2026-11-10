"""Exact information content of candidate GRF ranges, used to choose the 2 x 2 design."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from grf import exact_mi, fisher, volume  # noqa: E402

if __name__ == "__main__":
    th = torch.tensor([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [8.0, 8.0], [12.0, 12.0]], device="cuda")
    g = fisher(th)
    for t, gi in zip(th.tolist(), g):
        print("theta", t, "G diag", gi.diagonal().tolist(), "sd", (1 / gi.diagonal().sqrt()).tolist())
    boxes = {
        "[2,3]": ((2, 3), (2, 3)), "[2.4,2.6]": ((2.4, 2.6), (2.4, 2.6)), "[2.3,2.7]": ((2.3, 2.7), (2.3, 2.7)),
        "[4,12]": ((4, 12), (4, 12)), "[5,12]": ((5, 12), (5, 12)), "[7,9]": ((7, 9), (7, 9)), "[6,8]": ((6, 8), (6, 8)),
        "[6,10]": ((6, 10), (6, 10)), "[3,6]": ((3, 6), (3, 6)), "[2,4]": ((2, 4), (2, 4)),
    }
    for name, b in boxes.items():
        t0 = time.perf_counter()
        v = volume(b)
        mi, se = exact_mi(b, samples=1024, m=128)
        print(f"{name:10s} area {(b[0][1]-b[0][0])*(b[1][1]-b[1][0]):7.2f}  V {v:9.2f}  I {mi:6.3f} +- {se:.3f} nats  exp(I) {torch.tensor(mi).exp().item():8.2f}  ({time.perf_counter()-t0:.1f}s)")
