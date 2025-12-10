from typing import TYPE_CHECKING, Literal

from aorta_personalization.mesh.types import Geometries, MeshInfo
from cheartpy.fe.api import create_bcpatch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from aorta_personalization.solid.types import SolidProbVars
    from cheartpy.fe.trait import IBCPatch, IVariable


def create_aorta_bcs(mesh: MeshInfo, disp: IVariable, motion: IVariable | None) -> list[IBCPatch]:
    if motion is None:
        return [create_bcpatch(i, disp, "dirichlet", 0, 0, 0) for i in mesh.ENDS]
    return [create_bcpatch(i, disp, "dirichlet", motion) for i in mesh.ENDS]


def create_cylinder_bcs(
    disp: IVariable,
    motion: IVariable | None,
    patchs: Mapping[int, Literal[1, 2, 3]],
) -> list[IBCPatch]:
    return [
        item
        for k, v in patchs.items()
        for item in [
            # create_bcpatch(k, disp, "neumann", 0, 0, 0),
            create_bcpatch(k, (disp, v), "dirichlet", (motion, v) if motion else 0),
        ]
    ]


def create_boundary_condition_list(
    mesh: MeshInfo, v: SolidProbVars, motion: IVariable | None
) -> list[IBCPatch]:
    match mesh.GEO:
        case Geometries.AORTA:
            return create_aorta_bcs(mesh, v.U, motion)
        case Geometries.BENT_CYLINDER:
            return create_cylinder_bcs(v.U, motion, {1: 1, 2: 3})
        case Geometries.STRAIGHT_CYLINDER:
            return create_cylinder_bcs(v.U, motion, {1: 1, 2: 1})
        case Geometries.BRANCHED_CYLINDER:
            return create_cylinder_bcs(v.U, motion, {1: 1, 2: 1})
