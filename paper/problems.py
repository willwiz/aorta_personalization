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
_mode: list[Literal["const", "grad", "sine", "circ"]] = ["sine", "grad"]
_target: float = 0.5
TRACKING_FORWARD_BULGE = ProblemParameters(
    P=Labels(
        N="tracking_bulge_raw",
        D=Path("tracking_bulge_raw"),
        CL=None,
        CL_n=16,
        CL_i=4,
        DL=None,
        DL_n=16,
        DL_i=3,
    ),
    track=None,
    init=None,
    motion_var=None,
    matpars=MaterialProperty("const", 35.0, 0.0),
    pres=-10.0,
    ex_freq=1,
    dt=0.01,
    nt=100,
    target=_target,
)

PROBS_FORWARD_STRAIGHT = {
    m: [
        ProblemParameters(
            P=Labels(
                N=f"forward/forward_straight_{m}_{i}",
                D=Path("forward") / f"forward_straight_{m}_{i}",
                CL="CL",
                CL_n=i,
                CL_i=4,
                DL=None,
                DL_n=i,
                DL_i=3,
            ),
            track=None,
            init=None,
            motion_var="Zeros",
            matpars=MaterialProperty(m, _stiffness, _stiffness),
            pres=-10.0,
            ex_freq=1,
            dt=0.01,
            nt=100,
            target=_target,
        )
        for i in [2, 4, 8, 16]
    ]
    for m in _mode
}
_selected_modes: list[Literal["const", "grad", "sine", "circ"]] = ["grad"]
PROBS_FORWARD_BENT = {
    m: [
        ProblemParameters(
            P=Labels(
                N=f"forward/forward_bent_{m}_{i}",
                D=Path("forward") / f"forward_bent_{m}_{i}",
                CL="CL",
                CL_n=i,
                CL_i=4,
                DL=None,
                DL_n=i,
                DL_i=3,
            ),
            track=None,
            init=None,
            motion_var="Zeros",
            matpars=MaterialProperty(m, _stiffness, _stiffness),
            pres=-10.0,
            ex_freq=1,
            dt=0.01,
            nt=100,
            target=_target,
        )
        for i in [2, 4, 8, 16]
    ]
    for m in _selected_modes
}
_selected_modes: list[Literal["const", "grad", "sine", "circ"]] = ["sine", "circ"]
PROBS_FORWARD_BULGE = {
    m: [
        ProblemParameters(
            P=Labels(
                N=f"forward/forward_bulge_{m}_{i}",
                D=Path("forward") / f"forward_bulge_{m}_{i}",
                CL="CL",
                CL_n=max(16, i),
                CL_i=4,
                DL=None,
                DL_n=i,
                DL_i=3,
            ),
            track=Path("tracking_bulge"),
            init=None,
            motion_var="DISP",
            matpars=MaterialProperty(m, _stiffness, _stiffness),
            pres=-10.0,
            ex_freq=1,
            dt=0.01,
            nt=100,
            target=_target,
        )
        for i in [2, 4, 8, 16]
    ]
    for m in _selected_modes
}

PROBS_INVERSE_STRAIGHT = {
    m: [
        ProblemParameters(
            P=Labels(
                N=f"inverse/inverse_straight_{m}_{i}",
                D=Path("inverse") / f"inverse_straight_{m}_{i}",
                CL="CL",
                CL_n=i,
                CL_i=4,
                DL="DL",
                DL_n=i,
                DL_i=3,
            ),
            track=Path("forward") / f"forward_straight_{m}_{16}",
            init=Path("forward") / f"forward_straight_{m}_{i}",
            motion_var="AUTO",
            matpars=MaterialProperty(m, _stiffness, _stiffness),
            pres=-10.0,
            ex_freq=1,
            t0=1,
            dt=0.01,
            nt=100,
            target=_target,
        )
        for i in [2, 4, 8, 16]
    ]
    for m in _mode
}
PROBS_NOISE_STRAIGHT = [
    ProblemParameters(
        P=Labels(
            N=f"noise/noise_straight_{m}_{i}_{j}_{n}_{k}",
            D=Path("noise") / f"noise_straight_{m}_{i}_{j}_{n}_{k}",
            CL="CL",
            CL_n=i,
            CL_i=4,
            DL="DL",
            DL_n=i,
            DL_i=3,
        ),
        track=Path("forward") / f"forward_straight_{m}_{i}",
        init=Path("forward") / f"forward_straight_{m}_{i}",
        motion_var="AUTO",
        matpars=MaterialProperty(m, _stiffness, _stiffness),
        pres=-10.0,
        ex_freq=1,
        t0=100,
        dt=0.01,
        nt=100,
        target=_target,
        spac=j,
        noise=n,
    )
    for m in _mode
    for i in [8]
    for j in [1, 2]
    for n in [0.5, 1.0]
    for k in range(10)
]
_temp: list[Literal["const", "grad", "sine", "circ"]] = ["grad"]
PROBS_INVERSE_BENT = {
    m: [
        ProblemParameters(
            P=Labels(
                N=f"inverse/inverse_bent_{m}_{i}",
                D=Path("inverse") / f"inverse_bent_{m}_{i}",
                CL="CL",
                CL_n=i,
                CL_i=4,
                DL="DL",
                DL_n=i,
                DL_i=3,
            ),
            track=Path("forward") / f"forward_bent_{m}_{i}",
            init=Path("forward") / f"forward_bent_{m}_{i}",
            motion_var="AUTO",
            matpars=MaterialProperty(m, _stiffness, _stiffness),
            pres=-10.0,
            ex_freq=1,
            t0=100,
            dt=0.01,
            nt=100,
            target=_target,
        )
        for i in [2, 4, 8, 16]
    ]
    for m in _temp
}
_temp: list[Literal["const", "grad", "sine", "circ"]] = ["grad"]
PROBS_NOISE_BENT = [
    ProblemParameters(
        P=Labels(
            N=f"noise/noise_bent_{m}_{i}_{j}_{n}_{k}",
            D=Path("noise") / f"noise_bent_{m}_{i}_{j}_{n}_{k}",
            CL="CL",
            CL_n=i,
            CL_i=4,
            DL="DL",
            DL_n=i,
            DL_i=3,
        ),
        track=Path("forward") / f"forward_bent_{m}_{i}",
        init=Path("forward") / f"forward_bent_{m}_{i}",
        motion_var="AUTO",
        matpars=MaterialProperty(m, _stiffness, _stiffness),
        pres=-10.0,
        ex_freq=1,
        t0=100,
        dt=0.01,
        nt=100,
        target=_target,
        spac=j,
        noise=n,
    )
    for m in _temp
    for i in [8]
    for j in [1, 2]
    for n in [0.5, 1.0]
    for k in range(10)
]
_temp_bulge: list[Literal["const", "grad", "sine", "circ"]] = ["sine", "circ"]
PROBS_INVERSE_BULGE = {
    m: [
        ProblemParameters(
            P=Labels(
                N=f"inverse/inverse_bulge_{m}_{i}",
                D=Path("inverse") / f"inverse_bulge_{m}_{i}",
                CL="CL",
                CL_n=16,
                CL_i=4,
                DL="DL",
                DL_n=i,
                DL_i=3,
            ),
            track=Path("tracking_bulge"),
            init=Path("forward") / f"forward_bulge_{m}_{i}",
            motion_var="AUTO",
            matpars=MaterialProperty(m, _stiffness, _stiffness),
            pres=-10.0,
            ex_freq=1,
            t0=100,
            dt=0.01,
            nt=100,
            target=_target,
        )
        for i in [2, 4, 8, 16]
    ]
    for m in _temp_bulge
}
_selected_modes: list[Literal["const", "grad", "sine", "circ"]] = ["sine"]
PROBS_NOISE_BULGE = [
    ProblemParameters(
        P=Labels(
            N=f"noise/noise_bulge_{m}_{i}_{j}_{n}_{k}",
            D=Path("noise") / f"noise_bulge_{m}_{i}_{j}_{n}_{k}",
            CL="CL",
            CL_n=i,
            CL_i=4,
            DL="DL",
            DL_n=i,
            DL_i=3,
        ),
        track=Path("forward") / f"forward_bulge_{m}_{i}",
        init=Path("forward") / f"forward_bulge_{m}_{i}",
        motion_var="AUTO",
        matpars=MaterialProperty(m, _stiffness, _stiffness),
        pres=-10.0,
        ex_freq=1,
        t0=100,
        dt=0.01,
        nt=100,
        target=_target,
        spac=j,
        noise=n,
    )
    for m in _selected_modes
    for i in [8]
    for j in [1, 2]
    for n in [0.5, 1.0]
    for k in range(10)
]
