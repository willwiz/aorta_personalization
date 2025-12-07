from typing import TYPE_CHECKING

from cheartpy.io.api import check_for_meshes, chread_d
from cheartpy.mesh.api import import_cheart_mesh
from pytools.result import Err, Ok

if TYPE_CHECKING:
    import numpy as np
    from cheartpy.mesh.struct import CheartMesh
    from pytools.arrays import A2
    from pytools.logging.trait import ILogger

    from ._types import MeshInfo


def find_meshes(
    mesh: MeshInfo,
) -> Ok[tuple[CheartMesh[np.float64, np.intc], A2[np.float64]]] | Err:
    if not check_for_meshes(mesh.DISP, mesh.PRES, home=mesh.DIR):
        return Err(FileExistsError(f"Mesh for {mesh.DIR} does not exist"))
    disp = import_cheart_mesh(mesh.DIR / mesh.DISP)
    pres = import_cheart_mesh(mesh.DIR / mesh.PRES)
    if disp.top.n != pres.top.n:
        return Err(ValueError("Displacement and Pressure mesh do not match"))
    cl = chread_d(mesh.DIR / mesh.FIELD)
    return Ok((disp, cl))


def create_mesh(
    mesh: MeshInfo,
) -> Ok[tuple[CheartMesh[np.float64, np.intc], A2[np.float64]]] | Err: ...


def prep_mesh(
    mesh: MeshInfo, *, log: ILogger, override: bool = False
) -> Ok[tuple[CheartMesh[np.float64, np.intc], A2[np.float64]]] | Err:
    if not override:
        match find_meshes(mesh):
            case Ok((disp, cl)):
                return Ok((disp, cl))
            case Err(e):
                log.info(str(e), "Creating new mesh")
    return create_mesh(mesh)
