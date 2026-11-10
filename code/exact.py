"""Exact information content of the four GRF ranges: I(theta; X), the Fisher volume V and the
Clarke-Barron approximation log V - (k/2) log(2 pi e). Also checks the quadrature resolution."""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import systems as S  # noqa: E402
from grf import exact_mi, volume  # noqa: E402

if __name__ == "__main__":
    out = {}
    for name in ("gA", "gB", "gC", "gD"):
        box = S.SYSTEMS[name]["box"]
        i, se = exact_mi(box, samples=8192, m=192, seed=1)
        i_coarse, _ = exact_mi(box, samples=8192, m=96, seed=1)
        v = volume(box, m=96)
        area = (box[0][1] - box[0][0]) * (box[1][1] - box[1][0])
        out[name] = dict(box=box, area=area, I=i, I_se=se, I_m96=i_coarse, V=v, clarke_barron=math.log(v) - math.log(2 * math.pi * math.e))
        print(name, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in out[name].items()}, flush=True)
    (HERE.parent / "results" / "exact_grf.json").write_text(json.dumps(out, indent=1))
