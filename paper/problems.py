# /// script
# require-python = ">=3.14"
# dependencies = [
#     "numpy",
# ]
# ///

from pathlib import Path
from typing import Literal

from aorta_personalization.problem.types import Labels, MaterialProperty, ProblemParameters

_stiffness: float = 30.0
_mode: list[Literal["const", "grad", "sine", "circ"]] = ["const", "grad", "sine", "circ"]

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

PROBS_FORWARD_STRAIGHT = [
    ProblemParameters(
        P=Labels(
            N=f"forward_straight_{m}_{i}",
            D=Path(f"forward_straight_{m}_{i}"),
            CL="CL",
            CL_n=i,
            CL_i=4,
            DL=None,
            DL_n=i,
            DL_i=3,
        ),
        track=None,
        motion_var=None,
        matpars=MaterialProperty(m, _stiffness, _stiffness),
        pres=-10.0,
        ex_freq=1,
        dt=0.01,
        nt=100,
        target=0.4,
    )
    for m in _mode
    for i in [2, 4, 8, 16]
]
PROBS_FORWARD_BENT = [
    ProblemParameters(
        P=Labels(
            N=f"forward_bent_{m}_{i}",
            D=Path(f"forward_bent_{m}_{i}"),
            CL="CL",
            CL_n=i,
            CL_i=4,
            DL=None,
            DL_n=i,
            DL_i=3,
        ),
        track=None,
        motion_var=None,
        matpars=MaterialProperty(m, _stiffness, _stiffness),
        pres=-10.0,
        ex_freq=1,
        dt=0.01,
        nt=100,
        target=0.4,
    )
    for m in _mode
    for i in [2, 4, 8, 16]
]
PROBS_FORWARD_BULGE = [
    ProblemParameters(
        P=Labels(
            N=f"forward_bulge_{m}_{i}",
            D=Path(f"forward_bulge_{m}_{i}"),
            CL="CL",
            CL_n=i,
            CL_i=4,
            DL=None,
            DL_n=i,
            DL_i=3,
        ),
        track=Path("track_bent_bulge"),
        motion_var="Disp*",
        matpars=MaterialProperty(m, _stiffness, _stiffness),
        pres=-10.0,
        ex_freq=1,
        dt=0.01,
        nt=100,
        target=0.4,
    )
    for m in _mode
    for i in [2, 4, 8, 16]
]

PROBS_INVERSE_STRAIGHT = [
    ProblemParameters(
        P=Labels(
            N=f"results/inverse_straight_{m}_{i}",
            D=Path(f"results/inverse_straight_{m}_{i}"),
            CL="CL",
            CL_n=i,
            CL_i=4,
            DL="DL",
            DL_n=i,
            DL_i=3,
        ),
        track=Path(f"forward_straight_{m}_{i}"),
        motion_var="AUTO",
        matpars=MaterialProperty(m, _stiffness, _stiffness),
        pres=-10.0,
        ex_freq=1,
        t0=1,
        dt=0.01,
        nt=100,
        target=0.5,
    )
    for m in _mode
    for i in [2, 4, 8, 16]
]
PROBS_NOISE_STRAIGHT = [
    ProblemParameters(
        P=Labels(
            N=f"results/noise_straight_{m}_{i}_{j}_{n}_{k}",
            D=Path(f"results/noise_straight_{m}_{i}_{j}_{n}_{k}"),
            CL="CL",
            CL_n=i,
            CL_i=4,
            DL="DL",
            DL_n=i,
            DL_i=3,
        ),
        track=Path(f"forward_straight_{m}_{i}"),
        motion_var="AUTO",
        matpars=MaterialProperty(m, _stiffness, _stiffness),
        pres=-10.0,
        ex_freq=1,
        t0=100,
        dt=0.01,
        nt=100,
        target=0.5,
        spac=j,
        noise=n,
    )
    for m in _mode
    for i in [8]
    for j in [1, 2]
    for n in [0.5, 1.0, 1.5]
    for k in range(10)
]
PROBS_INVERSE_BENT = [
    ProblemParameters(
        P=Labels(
            N=f"results/inverse_bent_{m}_{i}",
            D=Path(f"results/inverse_bent_{m}_{i}"),
            CL="CL",
            CL_n=i,
            CL_i=4,
            DL="DL",
            DL_n=i,
            DL_i=3,
        ),
        track=Path(f"forward_bent_{m}_{i}"),
        motion_var="AUTO",
        matpars=MaterialProperty(m, _stiffness, _stiffness),
        pres=-10.0,
        ex_freq=1,
        t0=100,
        dt=0.01,
        nt=100,
        target=0.5,
    )
    for m in _mode
    for i in [2, 4, 8, 16]
]
PROBS_NOISE_BENT = [
    ProblemParameters(
        P=Labels(
            N=f"results/noise_bent_{m}_{i}_{j}_{n}_{k}",
            D=Path(f"results/noise_bent_{m}_{i}_{j}_{n}_{k}"),
            CL="CL",
            CL_n=i,
            CL_i=4,
            DL="DL",
            DL_n=i,
            DL_i=3,
        ),
        track=Path(f"forward_bent_{m}_{i}"),
        motion_var="AUTO",
        matpars=MaterialProperty(m, _stiffness, _stiffness),
        pres=-10.0,
        ex_freq=1,
        t0=100,
        dt=0.01,
        nt=100,
        target=0.5,
        spac=j,
        noise=n,
    )
    for m in _mode
    for i in [8]
    for j in [1, 2]
    for n in [0.5, 1.0, 1.5]
    for k in range(10)
]
PROBS_INVERSE_BULGE = [
    ProblemParameters(
        P=Labels(
            N=f"results/inverse_bulge_{m}_{i}",
            D=Path(f"results/inverse_bulge_{m}_{i}"),
            CL="CL",
            CL_n=i,
            CL_i=4,
            DL="DL",
            DL_n=i,
            DL_i=3,
        ),
        track=Path(f"forward_bulge_{m}_{i}"),
        motion_var="AUTO",
        matpars=MaterialProperty(m, _stiffness, _stiffness),
        pres=-10.0,
        ex_freq=1,
        t0=1,
        dt=0.01,
        nt=100,
        target=0.5,
    )
    for m in _mode
    for i in [2, 4, 8, 16]
]
PROBS_NOISE_BULGE = [
    ProblemParameters(
        P=Labels(
            N=f"results/noise_bulge_{m}_{i}_{j}_{n}_{k}",
            D=Path(f"results/noise_bulge_{m}_{i}_{j}_{n}_{k}"),
            CL="CL",
            CL_n=i,
            CL_i=4,
            DL="DL",
            DL_n=i,
            DL_i=3,
        ),
        track=Path(f"forward_bulge_{m}_{i}"),
        motion_var="AUTO",
        matpars=MaterialProperty(m, _stiffness, _stiffness),
        pres=-10.0,
        ex_freq=1,
        t0=100,
        dt=0.01,
        nt=100,
        target=0.5,
        spac=j,
        noise=n,
    )
    for m in _mode
    for i in [8]
    for j in [1, 2]
    for n in [0.5, 1.0, 1.5]
    for k in range(10)
]
