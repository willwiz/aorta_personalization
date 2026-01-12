from pathlib import Path
from typing import TYPE_CHECKING, Required, TypedDict, Unpack, cast

import numpy as np
from cheartpy.cl.noise import create_noise
from cheartpy.io.api import chread_d, chwrite_d_utf
from cheartpy.search.api import get_var_index
from pytools.logging.api import NLOGGER
from pytools.result import Err, Ok
from scipy.interpolate import PchipInterpolator

from ._types import ProblemVariableNames

if TYPE_CHECKING:
    from aorta_personalization.mesh.types import MeshInfo
    from aorta_personalization.problem.types import ProblemParameters
    from cheartpy.cl.struct import CLPartition
    from pytools.arrays import A2
    from pytools.logging.trait import ILogger


class _MakeReferenceDataKwargs(TypedDict, total=False):
    p_noise: str
    p_xi: str
    p_x0: str
    p_xt: str
    p_u0: str
    p_ut: str
    p_p0: str
    p_pt: str
    p_dm: str
    p_data: str
    log: Required[ILogger]
    pedantic: bool
    cores: int


def _unpack_variable_prefixes(**kwargs: Unpack[_MakeReferenceDataKwargs]) -> ProblemVariableNames:
    return ProblemVariableNames(
        noise=kwargs.get("p_noise", "Noise"),
        xi=kwargs.get("p_xi", "Xi"),
        x0=kwargs.get("p_x0", "X0"),
        xt=kwargs.get("p_xt", "Xt"),
        u0=kwargs.get("p_u0", "U0"),
        ut=kwargs.get("p_ut", "Ut"),
        p0=kwargs.get("p_p0", "P0"),
        pt=kwargs.get("p_pt", "Pt"),
        dm=kwargs.get("p_dm", "DLDM"),
        data=kwargs.get("p_data", "CLDispt"),
    )


def make_reference_data_for_inverse_estimation[F: np.floating, I: np.integer](
    pb: ProblemParameters,
    mesh: MeshInfo,
    cl: A2[F],
    cl_part: CLPartition[F, I] | None,
    dl_part: CLPartition[F, I],
    **kwargs: Unpack[_MakeReferenceDataKwargs],
) -> Ok[None] | Err:
    log = kwargs.get("log", NLOGGER)
    _track = pb.track or Path()
    _init_folder = pb.init or Path()
    log.debug(f"Importing mesh from {mesh.DIR / mesh.DISP}")
    log.debug("Creating Noise Field for displacement")
    _pfx = _unpack_variable_prefixes(**kwargs)
    files = (f.name for f in _track.glob("Disp-*.D"))
    normal = chread_d(mesh.DIR / mesh.NORMAL)
    noise = create_noise(pb.noise, cl, normal, spatial_freq=(pb.spac * 3, pb.spac * 5))
    t = np.linspace(0, 1, dl_part.nn)
    match pb.matpars.form:
        case "const":
            dm = np.full_like(
                t,
                0.1 * (pb.matpars.baseline) - 1.0,
                dtype=cl.dtype,
            )
        case "grad":
            dm = 0.1 * (pb.matpars.baseline + pb.matpars.amplitude * np.exp(-2.0 * t)) - 1.0
        case "sine":
            dm = 0.1 * (pb.matpars.baseline + pb.matpars.amplitude * np.cos(np.pi * t) ** 2) - 1.0
        case "circ":
            dm = 0.1 * (pb.matpars.baseline + 0.0 * pb.matpars.amplitude * np.exp(-2.0 * t)) - 1.0
    match get_var_index(files, "Disp"):
        case Ok(items):
            final = max(items)
            rest = int(pb.target * final)
            init = pb.t0 / pb.nt
        case Err(e):
            return Err(e)
    log.debug(
        f"Reference data will be taken from {_init_folder}",
        f"The reference time step is taken as {rest}",
        f"The final time step is taken as {final}",
    )
    xi = chread_d(_init_folder / f"Space-{rest}.D")
    scale_factor = 1.0
    u0 = chread_d(_init_folder / f"Disp-{rest}.D")
    ut = chread_d(_init_folder / f"Disp-{final}.D")
    data = {
        _pfx.noise: noise,
        _pfx.xi: xi,
        _pfx.x0: xi - u0 * init,
        _pfx.xt: xi - u0 * init,
        _pfx.u0: u0 * init,
        _pfx.ut: ut * init * scale_factor,
        _pfx.p0: chread_d(_init_folder / f"Pres-{rest}.D") * init,
        _pfx.pt: chread_d(_init_folder / f"Pres-{final}.D") * init,
        _pfx.dm: dm[None, :],
        _pfx.data: ut - u0 + noise,
    }
    log.debug(f"Exporting initial values to {pb.P.D}")
    for k, v in data.items():
        chwrite_d_utf((pb.P.D / f"{k}.INIT"), v)
    if cl_part is None:
        return Ok(None)
    cl0data = chread_d(_init_folder / f"CLLM-{rest}.D")
    cl0 = cast(
        "A2[F]",
        PchipInterpolator(np.linspace(0, 1, len(cl0data)), cl0data)(cl_part.node),
    )
    cltdata = chread_d(_init_folder / f"CLLM-{final}.D")
    clt = cast(
        "A2[F]",
        PchipInterpolator(np.linspace(0, 1, len(cltdata)), cltdata)(cl_part.node),
    )
    for k, v in [("CL0LM.INIT", cl0), ("CLtLM.INIT", clt)]:
        chwrite_d_utf((pb.P.D / k), v)
    return Ok(None)
