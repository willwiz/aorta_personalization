import dataclasses as dc
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Literal

from cheartpy.fe.trait import IVariable

if TYPE_CHECKING:
    from pathlib import Path

    from pytools.logging.trait import LOG_LEVEL


CL_PARTITIONS = Literal["motion", "dilation"]
PRES_MODES = Literal["ramp", "sin"]
STIFF_MODES = Literal["const", "grad", "sine", "circ"]
MOTION_VAR = Literal["Zeros", "AUTO", "DISP", "VAR", "STEP"]

ClNodalLmType = Mapping[int, IVariable]


@dc.dataclass(slots=True, frozen=True)
class MaterialProperty:
    form: STIFF_MODES
    baseline: float
    amplitude: float


@dc.dataclass(slots=True, frozen=True)
class Labels:
    """Names for different problem components.

    Attributes:
    N: str
        Prefix names
    D: str
        output directory
    CL: str | None
        CL topology prefix
    CL_n: int
        Number of CL elements
    CL_i: int
        Surface id for CL topology
    DL: str | None
        DL topology prefix
    DL_n: int
        Number of DL elements
    DL_i: int
        Surface id for DL topology

    """

    N: str  # Prefix names
    D: Path  # output directory
    CL: str | None  # CL topology prefix
    CL_n: int  # CL topology prefix
    CL_i: int
    DL: str | None  # DL topology prefix
    DL_n: int  # DL topology prefix
    DL_i: int


@dc.dataclass(slots=True)
class ProblemParameters:
    P: Final[Labels]
    track: Path | None
    motion_var: MOTION_VAR | None
    matpars: MaterialProperty
    pres: float = -6.0
    target: float = 0.5
    ex_freq: int = 10
    t0: int = 1
    nt: int = 1000
    dt: float = 0.001
    noise: float = 0.0
    spac: int = 1
    log: LOG_LEVEL = "DEBUG"
