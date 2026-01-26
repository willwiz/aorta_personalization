from typing import TYPE_CHECKING

import numpy as np
from cheartpy.cl.mesh import (
    create_cheart_cl_nodal_meshes,
    create_cheart_cl_topology_meshes,
    create_cl_partition,
)
from cheartpy.io.api import (
    check_for_meshes,
    chread_d,
    chwrite_d_utf,
)
from pytools.path import clear_dir
from pytools.result import Err, Ok

if TYPE_CHECKING:
    from cheartpy.cl.struct import CLPartition
    from cheartpy.mesh.struct import CheartMesh
    from pytools.arrays import A2
    from pytools.logging import ILogger

    from ._types import MeshInfo

type _MeshInput[F: np.floating, I: np.integer] = tuple[MeshInfo, CheartMesh[F, I], A2[F]]


def setup_separated_cl_meshes[F: np.floating, I: np.integer](
    prefix: str | None,
    in_surf: int,
    n_seg: int,
    mesh_tuple: _MeshInput[F, I],
    *,
    log: ILogger,
) -> Ok[CLPartition[F, I]] | Ok[None] | Err:
    if prefix is None:
        return Ok(None)
    mesh, cheart_mesh, cl = mesh_tuple
    ftype = cheart_mesh.space.v.dtype
    dtype = cheart_mesh.top.v.dtype
    cl_top: CLPartition[F, I] = create_cl_partition(
        (prefix, in_surf), n_seg, log=log, ftype=ftype, dtype=dtype
    )
    cur_tops = mesh.DIR.glob(f"{cl_top.prefix}*_FE.T")
    n_tops = len(list(cur_tops))
    log.debug(
        f"Number of {cl_top.prefix} topologies currently is {n_tops}"
        f" should be {len(cl_top.n_prefix)}",
    )
    if (n_tops == len(cl_top.n_prefix)) and check_for_meshes(
        *cl_top.n_prefix.values(), home=mesh.DIR, bc=False
    ):
        log.info("CL topology already exists, skipped")
        return Ok(cl_top)
    norm_field = chread_d(mesh.DIR / mesh.NORMAL, dtype=ftype)
    log.debug("Creating cl topologies")
    log.debug("Creating cl meshes")
    match create_cheart_cl_nodal_meshes(
        mesh.DIR, cheart_mesh, cl, cl_top, in_surf, normal_check=norm_field, log=log
    ):
        case Ok(cl_meshs):
            pass
        case Err(e):
            return Err(e)
    log.debug("Saving cl meshes")
    for v in cl_meshs.values():
        v["mesh"].save(v["file"])
        chwrite_d_utf(v["file"].parent / (v["file"].name + "Normal-0.D"), v["n"])
    return Ok(cl_top)


def prep_topology_meshes[F: np.floating, I: np.integer](
    prefix: str | None,
    in_surf: int,
    n_seg: int,
    mesh_tuple: _MeshInput[F, I],
    *,
    log: ILogger,
) -> Ok[CLPartition[F, I]] | Ok[None] | Err:
    if prefix is None:
        return Ok(None)
    mesh, cheart_mesh, cl = mesh_tuple
    ftype = cheart_mesh.space.v.dtype
    dtype = cheart_mesh.top.v.dtype
    cl_top = create_cl_partition((prefix, in_surf), n_seg, log=log, ftype=ftype, dtype=dtype)
    cur_tops = mesh.DIR.glob(f"{cl_top.prefix}*_FE.T")
    log.debug(
        f"Number of {cl_top.prefix} topologies currently is {len(list(cur_tops))}"
        f" It should be {len(cl_top.n_prefix)}",
    )
    n_tops = len(list(cur_tops))
    if (n_tops == len(cl_top.n_prefix)) and check_for_meshes(
        *cl_top.n_prefix.values(), home=mesh.DIR, bc=False
    ):
        log.info("CL topology already exists, skipped")
        return Ok(cl_top)
    clear_dir(
        mesh.DIR,
        *[rf"{cl_top.prefix}*.{s}" for s in ["T", "X", "B", "PART", "INIT"]],
        rf"interface-{cl_top.prefix}*.IN",
        log=log,
    )
    norm_field = chread_d(mesh.DIR / mesh.NORMAL, dtype=ftype)
    log.debug("Creating cl topologies")
    log.debug("Creating cl meshes")
    match create_cheart_cl_topology_meshes(
        mesh.DIR, cheart_mesh, cl, cl_top, in_surf, normal_check=norm_field, log=log
    ):
        case Ok((lin_mesh, interface_mesh)):
            pass
        case Err(e):
            return Err(e)
    log.debug("Saving cl meshes")
    lin_mesh.save(mesh.DIR / f"{prefix}Az{mesh.ORDER}")
    # const_mesh.save(path(M.DIR, f"{prefix}Az{0}"))
    interface_mesh.save(mesh.DIR / f"{prefix}Az{'L'}")
    chwrite_d_utf(mesh.DIR / f"{prefix}Az{'L'}V_Support.INIT", cl_top.support)
    chwrite_d_utf(
        mesh.DIR / f"{prefix}Az{'L'}V_Elem.INIT",
        np.identity(cl_top.nn, dtype=float),
    )
    return Ok(cl_top)
