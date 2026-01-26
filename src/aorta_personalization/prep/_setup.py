from typing import TYPE_CHECKING, NamedTuple

from aorta_personalization.mesh.api import prep_cheart_mesh, prep_topology_meshes
from pytools.result import Err, Ok

if TYPE_CHECKING:
    import numpy as np
    from aorta_personalization.mesh.types import MeshInfo
    from aorta_personalization.problem.types import ProblemParameters
    from cheartpy.cl.struct import CLPartition
    from pytools.arrays import A2
    from pytools.logging import ILogger


class _SetupReturnType(NamedTuple):
    cl: A2[np.float64]
    cl_top: CLPartition[np.float64, np.intc] | None
    dl_top: CLPartition[np.float64, np.intc] | None


def run_setup(
    prob: ProblemParameters, mesh: MeshInfo, *, log: ILogger, override: bool = False
) -> Ok[_SetupReturnType] | Err:
    """Run the setup phase for aorta personalization problem.

    Parameters
    ----------
    prob : ProblemParameters
        Problem parameters.
    mesh : MeshInfo
        Mesh information.
    log : ILogger
        Logger instance.
    override : bool, optional
        Whether to override existing setup data, by default False.

    Returns
    -------
    Ok[Tuple]|Err
        On success, returns a tuple containing:
            centerline: A2[np.float64]
                Centerline data
            cl_partition: CLPartition[np.float64, np.intp] | None
                CL partition.
            dl_partition: CLPartition[np.float64, np.intp] | None
                DL partition.

    """
    name = f"{prob.P.N}.P"
    log.info(f"Prepping problem {name}")
    match prep_cheart_mesh(mesh, log=log, override=override):
        case Ok((cheart_mesh, cl_arr)):
            pass
        case Err(e):
            return Err(e)
    match prep_topology_meshes(
        prob.P.CL, prob.P.CL_i, prob.P.CL_n, (mesh, cheart_mesh, cl_arr), log=log
    ):
        case Ok(cl_top):
            pass
        case Err(e):
            return Err(e)
    match prep_topology_meshes(
        prob.P.DL, prob.P.DL_i, prob.P.DL_n, (mesh, cheart_mesh, cl_arr), log=log
    ):
        case Ok(dl_top):
            pass
        case Err(e):
            return Err(e)
    log.info("Topology prep done")
    return Ok(_SetupReturnType(cl=cl_arr, cl_top=cl_top, dl_top=dl_top))
