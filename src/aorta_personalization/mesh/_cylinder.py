from typing import TYPE_CHECKING, Literal, NamedTuple

from cheartpy.io.api import chwrite_d_utf
from cheartpy.mesh.cylinder_core.api import create_cylinder_mesh
from cheartpy.mesh.surface_core.normals import normalize_by_row
from pytools.logging import ILogger, LogEnum
from pytools.path import clear_dir

from ._types import MeshTuple
from ._variables import create_center_pos, create_fiber_field, define_centerline_field, warp_in_y

if TYPE_CHECKING:
    import numpy as np

    from ._types import CylinderDims, MeshInfo


class _TupleLogMessage(NamedTuple):
    info: list[str]
    debug: list[str]


def _remake_cylinder_mesh_msgs(
    mesh: MeshInfo, *, level: LogEnum, quad: Literal[True, False], warp: bool
) -> _TupleLogMessage:
    info = [
        f"Clearing all files in {mesh.DIR}",
        f"Remaking Mesh with as Quad={quad} and Warped={warp}",
    ]
    if level < LogEnum.DEBUG:
        return _TupleLogMessage(info, [])
    _map = [
        mesh.FIELD,
        mesh.NORMAL,
        *[f"{s}-0.D" for s in ("Fibers", "Z", "C", "R")],
        *[f"fix_suffix({v}).{x}" for v in [mesh.DISP, mesh.PRES] for x in ("X", "T", "B")],
    ]
    debug = ["Exporting CL Field is saved to:", *[str(mesh.DIR / v) for v in _map]]
    return _TupleLogMessage(info, debug)


def remake_cylinder_mesh(
    mesh: MeshInfo, dim: CylinderDims, *, quad: bool, warp: bool, log: ILogger
) -> MeshTuple[np.float64, np.intc]:
    log_info, log_debug = _remake_cylinder_mesh_msgs(mesh, level=log.level, quad=quad, warp=warp)
    log.info(*log_info)
    log.debug(*log_debug)
    mesh.DIR.mkdir(exist_ok=True)
    clear_dir(mesh.DIR)
    lin_mesh, quad_mesh = create_cylinder_mesh((*dim.shape, 0.0), dim.nelem, "x", make_quad=quad)
    disp_mesh = quad_mesh or lin_mesh
    cl = define_centerline_field(disp_mesh)
    center = create_center_pos(disp_mesh, cl)
    if warp:
        lin_mesh.space.v = warp_in_y(lin_mesh.space.v)
        disp_mesh.space.v = warp_in_y(disp_mesh.space.v)
        center = warp_in_y(center)
    normal = disp_mesh.space.v - center
    normal = normalize_by_row(normal)
    fibers = create_fiber_field(cl, normal, warp=warp)
    lin_mesh.save(mesh.DIR / mesh.PRES)
    disp_mesh.save(mesh.DIR / mesh.DISP)
    for k, v in [
        (mesh.FIELD, cl),
        (mesh.NORMAL, normal),
        ("Fibers-0.D", fibers),
        ("Z-0.D", fibers[:, 0:3]),
        ("C-0.D", fibers[:, 3:6]),
        ("R-0.D", fibers[:, 6:9]),
    ]:
        chwrite_d_utf(mesh.DIR / k, v)
    return MeshTuple(disp_mesh, cl)
