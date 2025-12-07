import dataclasses as dc
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypedDict, Unpack

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from aorta_personalization.material.types import MaterialProperty
    from aorta_personalization.mesh.types import CLPartitions, MeshInfo
    from cheartpy.fe.p_file import PFile
    from pytools.logging.trait import LOG_LEVEL


PRES_MODES = Literal["ramp", "sin"]
CL_PARTITIONS = Literal["motion", "dilation"]


class CLPartitionKwargs[F: np.floating, I: np.integer](TypedDict, total=False):
    motion: CLPartitions[F, I] | None
    dilation: CLPartitions[F, I] | None


class PFileGenerator[F: np.floating, I: np.integer](Protocol):
    def __call__(
        self, prob: ProblemParameters, mesh: MeshInfo, **part: Unpack[CLPartitionKwargs[F, I]]
    ) -> PFile: ...


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
    track: Path
    motion_var: Literal["zeros", "AUTO", "Disp*", "Disp.INIT", "Disp"] | None
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
