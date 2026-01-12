from typing import TYPE_CHECKING

from cheartpy.io.api import check_for_meshes, chread_d
from cheartpy.mesh.api import import_cheart_mesh
from pytools.result import Err, Ok

from ._aorta import setup_aorta_mesh
from ._cylinder import remake_cylinder_mesh
from ._types import Geometries, MeshInfo, MeshTuple

if TYPE_CHECKING:
    import numpy as np
    from pytools.logging.trait import ILogger


def find_meshes(
    mesh: MeshInfo,
) -> Ok[MeshTuple[np.float64, np.intc]] | Err:
    if not check_for_meshes(mesh.DISP, mesh.PRES, home=mesh.DIR):
        return Err(FileExistsError(f"Mesh for {mesh.DIR} does not exist"))
    match import_cheart_mesh(mesh.DIR / mesh.DISP):
        case Ok(disp):
            pass
        case Err(e):
            return Err(e)
    match import_cheart_mesh(mesh.DIR / mesh.PRES):
        case Ok(pres):
            pass
        case Err(e):
            return Err(e)
    if disp.top.n != pres.top.n:
        return Err(ValueError("Displacement and Pressure mesh do not match"))
    cl = chread_d(mesh.DIR / mesh.FIELD)
    return Ok(MeshTuple(disp, cl))


_QUAD_IS_2_RUFF = 2


def create_mesh(mesh: MeshInfo, *, log: ILogger) -> Ok[MeshTuple[np.float64, np.intc]] | Err:
    match mesh.GEO:
        case Geometries.STRAIGHT_CYLINDER:
            return Ok(
                remake_cylinder_mesh(
                    mesh, mesh.SPEC, quad=(mesh.ORDER == _QUAD_IS_2_RUFF), warp=False, log=log
                )
            )
        case Geometries.BENT_CYLINDER:
            return Ok(
                remake_cylinder_mesh(
                    mesh, mesh.SPEC, quad=(mesh.ORDER == _QUAD_IS_2_RUFF), warp=True, log=log
                )
            )
        case Geometries.AORTA:
            return setup_aorta_mesh(mesh, log=log).next()
        case Geometries.BRANCHED_CYLINDER:
            raise NotImplementedError


def prep_cheart_mesh(
    mesh: MeshInfo, *, log: ILogger, override: bool = False
) -> Ok[MeshTuple[np.float64, np.intc]] | Err:
    if not override:
        match find_meshes(mesh):
            case Ok() as res:
                return res
            case Err(e):
                log.info(str(e), "Creating new mesh")
    return create_mesh(mesh, log=log).next()
