from pathlib import Path
from typing import NamedTuple, Required, TypedDict, Unpack

import numpy as np
from aorta_personalization.mesh._centerline import prep_topology_meshes
from aorta_personalization.mesh._generation import prep_cheart_mesh
from aorta_personalization.mesh.types import MeshInfo
from aorta_personalization.problem.types import ProblemParameters
from cheartpy.cl.struct import CLPartition
from pytools.arrays import A2
from pytools.logging.trait import ILogger
from pytools.result import Err, Ok


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
            log.info("Mesh prep done")
        case Err(e):
            log.error(f"Mesh prep failed: {e}")
            return Err(e)
    cl_top = prep_topology_meshes(
        prob.P.CL, prob.P.CL_i, prob.P.CL_n, (mesh, cheart_mesh, cl_arr), log=log
    )
    dl_top = prep_topology_meshes(
        prob.P.DL, prob.P.DL_i, prob.P.DL_n, (mesh, cheart_mesh, cl_arr), log=log
    )
    log.info("Topology prep done")
    return Ok(_SetupReturnType(cl=cl_arr, cl_top=cl_top, dl_top=dl_top))


class _PostProcessPhysicalSpaceKwargs(TypedDict, total=False):
    home: Path
    prefix: str
    cores: int


def postprocess_physical_space(
    ref_space: str, disp: str, **kwargs: Unpack[_PostProcessPhysicalSpaceKwargs]
) -> None:
    """Post-process the physical space data after simulation.

    Returns
    -------
    None

    """
    raise NotImplementedError


def postprocess_stiffness_field(
    stiff: str, **kwargs: Unpack[_PostProcessPhysicalSpaceKwargs]
) -> None:
    """Post-process the stiffness field data after simulation.

    Returns
    -------
    None

    """
    raise NotImplementedError


class _PostProcessInverseProbKwargs(TypedDict, total=False):
    log: Required[ILogger]
    cores: int


def postprocess_inverse_prob[F: np.floating, I: np.integer](
    pb: ProblemParameters,
    mesh: MeshInfo,
    cl: A2[np.float64],
    cl_top: CLPartition[F, I] | None,
    dl_top: CLPartition[F, I],
    **kwargs: Unpack[_PostProcessInverseProbKwargs],
) -> list[str]: ...
