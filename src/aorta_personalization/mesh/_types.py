import dataclasses as dc
import enum
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from cheartpy.cl.struct import CLPartition
    from pytools.arrays import T3


class Geometries(enum.StrEnum):
    AORTA = "AORTA"
    STRAIGHT_CYLINDER = "STRAIGHT_CYLINDER"
    BENT_CYLINDER = "BENT_CYLINDER"
    BRANCHED_CYLINDER = "BRANCHED_CYLINDER"


class ElementTypes(enum.StrEnum):
    HEX = "HEX"
    TET = "TET"


@dc.dataclass(slots=True, frozen=True)
class CylinderDims:
    shape: T3[float] = (9.0, 12.0, 200.0)
    nelem: T3[int] = (3, 32, 100)


@dc.dataclass(slots=True, frozen=True)
class MeshInfo:
    GEO: Geometries
    DIR: Path
    SPEC: CylinderDims
    DISP: str
    PRES: str
    ELEM: ElementTypes
    ORDER: Literal[1, 2]
    FIELD: str
    NORMAL: str
    INNER: int
    OUTER: int
    INLET: int
    OUTLET: int
    ENDS: Sequence[int]


class CLPartitions[F: np.floating, I: np.integer](TypedDict):
    motion: CLPartition[F, I] | None
    dilation: CLPartition[F, I] | None
