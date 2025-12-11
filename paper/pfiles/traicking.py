# /// script
# require-python = ">=3.14"
# dependencies = [
#     "pytools",
#     "cheartpy",
#     "aorta_personalization",
# ]
# ///


from pathlib import Path
from typing import Required, TypedDict

from aorta_personalization.problem.types import Labels, MaterialProperty, ProblemParameters
from cheartpy.io.api import chread_d, chwrite_d_utf
from pytools.progress import ProgressBar

TRACKING_FORWARD_BULGE = ProblemParameters(
    P=Labels(
        N="tracking_bulge_raw",
        D=Path("tracking_bulge_raw"),
        CL=None,
        CL_n=8,
        CL_i=4,
        DL=None,
        DL_n=8,
        DL_i=3,
    ),
    track=None,
    motion_var=None,
    matpars=MaterialProperty("const", 30.0, 0.0),
    pres=-10.0,
    ex_freq=1,
    dt=0.01,
    nt=100,
    target=0.0,
)


class _TrackingVar(TypedDict, total=False):
    home: Path | str
    disp: Required[str]


def create_tracking_disp(var: _TrackingVar, target: float, max_step: int = 100) -> None:
    home = Path(var.get("home", "."))
    raw_home = home.parent / f"{home.name}_raw"
    home.mkdir(exist_ok=True)
    mid = int(target * max_step)
    new_span = 1.0 - target
    zeros = 0.0 * chread_d(f"{raw_home}/{var['disp']}-0.D")
    bart = ProgressBar(n=max_step + 1)
    for i in range(mid):
        chwrite_d_utf(f"{home}/{var['disp']}-{i}.D", zeros)
        bart.next()
    for i in range(mid, max_step + 1):
        q = (i - mid) // new_span
        m = (i - mid) % new_span
        if q == max_step:
            q = max_step - 1
            m = 1.0
        left = chread_d(f"{raw_home}/{var['disp']}-{int(q)}.D")
        right = chread_d(f"{raw_home}/{var['disp']}-{int(q) + 1}.D")
        chwrite_d_utf(f"{home}/{var['disp']}-{i}.D", m * right + (1 - m) * left)
        bart.next()
    data = chread_d(f"{raw_home}/{var['disp']}-{max_step}.D")
    chwrite_d_utf(f"{home}/{var['disp']}-{max_step}.D", data)
    bart.next()
