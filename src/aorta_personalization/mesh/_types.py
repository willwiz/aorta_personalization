import dataclasses as dc
import enum
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple

from cheartpy.fe.trait import ICheartTopology

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    from cheartpy.mesh.struct import CheartMesh
    from pytools.arrays import A2, T3


class MeshTuple[F: np.floating, I: np.integer](NamedTuple):
    mesh: CheartMesh[F, I]
    cl: A2[F]


class Geometries(enum.StrEnum):
    AORTA = "AORTA"
    STRAIGHT_CYLINDER = "STRAIGHT_CYLINDER"
    BENT_CYLINDER = "BENT_CYLINDER"
    BRANCHED_CYLINDER = "BRANCHED_CYLINDER"


class ElementTypes(enum.StrEnum):
    HEX = "HEX"
    TET = "TET"


class BCPatchTag(NamedTuple):
    name: str
    side: int


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
    INNER: BCPatchTag
    OUTER: BCPatchTag
    INLET: BCPatchTag
    OUTLET: BCPatchTag
    ENDS: Sequence[int]


# symbols follow continuum-mechanics notation


@dc.dataclass(slots=True, frozen=True)
class ProblemTopologies:
    U: ICheartTopology
    P: ICheartTopology
    X: ICheartTopology
    inner: ICheartTopology
    outer: ICheartTopology
    inlet: ICheartTopology | None
    outlet: ICheartTopology | None


type ProblemInterfaces = Sequence[ICheartTopology]
