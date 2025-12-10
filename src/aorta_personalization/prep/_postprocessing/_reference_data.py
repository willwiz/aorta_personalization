from pathlib import Path
from typing import TYPE_CHECKING, Required, TypedDict, Unpack, cast

import numpy as np
from cheartpy.cl.noise import create_noise
from cheartpy.io.api import chread_d, chwrite_d_utf
from cheartpy.search.api import get_var_index
from pytools.logging.api import NLOGGER
from pytools.result import Err, Ok
from scipy.interpolate import PchipInterpolator

if TYPE_CHECKING:
    from aorta_personalization.mesh.types import MeshInfo
    from aorta_personalization.problem.types import ProblemParameters
    from cheartpy.cl.struct import CLPartition
    from pytools.arrays import A2
    from pytools.logging.trait import ILogger


class _MakeReferenceDataKwargs(TypedDict, total=False):
    log: Required[ILogger]
    pedantic: bool
    cores: int


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
    log.debug(f"Importing mesh from {mesh.DIR / mesh.DISP}")
    log.debug("Creating Noise Field for displacement")
    files = (f.name for f in _track.glob("Disp-*.D"))
    match get_var_index(files, "Disp"):
        case Ok(items):
            final = max(items)
            rest = int(pb.target * final)
            init = pb.t0 / pb.nt
        case Err(e):
            return Err(e)
    log.debug(
        f"Reference data will be taken from {_track}",
        f"The reference time step is taken as {rest}",
        f"The final time step is taken as {final}",
    )
    normal = chread_d(mesh.DIR / mesh.NORMAL)
    noise = create_noise(pb.noise, cl, normal, spatial_freq=(pb.spac * 3, pb.spac * 5)) * 0.0
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
            dm = 0.1 * (pb.matpars.baseline + pb.matpars.amplitude * np.exp(-2.0 * t)) - 1.0
    xi = chread_d(_track / f"Space-{rest}.D")
    u0 = chread_d(_track / f"Disp-{rest}.D")
    ut = chread_d(_track / f"Disp-{final}.D")
    data = {
        "Noise": noise,
        "Xi": xi,
        "X0": xi - u0 * init,
        "Xt": xi - u0 * init,
        "U0": -u0 * init,
        "Ut": ut * init,
        "P0": chread_d(_track / f"Pres-{rest}.D") * init,
        "Pt": chread_d(_track / f"Pres-{final}.D") * init,
        "DLDM": dm[None, :],
        "CLDispt": ut - u0 + noise,
    }
    log.debug(f"Exporting initial values to {pb.P.D}")
    for k, v in data.items():
        chwrite_d_utf((pb.P.D / f"{k}.INIT"), v)
    if cl_part is None:
        return Ok(None)
    cl0data = chread_d(_track / f"CLLM-{rest}.D")
    cl0 = cast(
        "A2[F]",
        PchipInterpolator(np.linspace(0, 1, len(cl0data)), cl0data)(cl_part.node),
    )
    cltdata = chread_d(_track / f"CLLM-{final}.D")
    clt = cast(
        "A2[F]",
        PchipInterpolator(np.linspace(0, 1, len(cltdata)), cltdata)(cl_part.node),
    )
    for k, v in [("CL0LM.INIT", cl0), ("CLtLM.INIT", clt)]:
        chwrite_d_utf((pb.P.D / k), v)
    return Ok(None)
