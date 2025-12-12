from pathlib import Path
from typing import TYPE_CHECKING, Required, TypedDict, Unpack

import numpy as np
from aorta_personalization.prep._cl_variables import expand_cl_variables_to_main_topology
from aorta_personalization.prep._tools import addwrite_var
from cheartpy.io.api import chread_d, chwrite_d_utf
from cheartpy.search.api import get_var_index
from pytools.logging.api import NLOGGER
from pytools.result import Err, Ok

from ._forward import postprocess_physical_space

if TYPE_CHECKING:
    from aorta_personalization.mesh.types import MeshInfo
    from aorta_personalization.problem.types import ProblemParameters
    from cheartpy.cl.struct import CLPartition
    from pytools.arrays import A2
    from pytools.logging.trait import ILogger


class _UpdateStiffnessKwargs(TypedDict, total=False):
    lm: str
    output: str


def update_stiffness(root: Path, i: int, **kwargs: Unpack[_UpdateStiffnessKwargs]) -> None:
    var = kwargs.get("lm", "DM")
    out = kwargs.get("output", "Stiff")
    lm = chread_d(root / f"{var}-{i}.D")
    chwrite_d_utf(root / f"{out}-{i}.D", 10.0 * (1.0 + lm))


def compute_stiffness_from_dl_field[F: np.floating, I: np.integer](
    part: CLPartition[F, I] | None,
    prefix: str,
    cl: A2[F],
    *,
    root_dir: Path,
    **kwargs: Unpack[_UpdateStiffnessKwargs],
) -> Ok[str] | Ok[None] | Err:
    """Post-process the stiffness field data after simulation.

    Returns
    -------
    None

    """
    match expand_cl_variables_to_main_topology(part, cl, f"{prefix}", root_dir=root_dir):
        case Ok([dl, *_]):
            files = (f.name for f in Path(root_dir).glob(f"{dl}-*.D"))
        case Ok(list()):
            return Ok(None)
        case Err(e):
            return Err(e)
    match get_var_index(files, dl):
        case Ok(items):
            pass
        case Err(e):
            return Err(e)
    for i in items:
        update_stiffness(root_dir, i, **kwargs)
    return Ok(prefix)


def invert_var_for_inverse_mechanics(var: Path, out: Path) -> None:
    if not var.is_file():
        return
    data = chread_d(var)
    chwrite_d_utf(out, -data)


def postprocess_inverse_mechanics(
    *var: tuple[str, str],
    root_dir: Path,
    log: ILogger,
) -> None:
    for v_in, v_out in var:
        files = (f.name for f in root_dir.glob(f"{v_in}-*.D"))
        match get_var_index(files, f"{v_in}"):
            case Ok(items):
                if not items:
                    log.debug(f"No variable output found for {v_in}")
                    continue
            case Err(e):
                log.error(f"Failed to get variable indices for {v_in}: {e}")
                continue
        for i in items:
            invert_var_for_inverse_mechanics(
                root_dir / f"{v_in}-{i}.D", root_dir / f"{v_out}-{i}.D"
            )


class _PostProcessInverseProbKwargs(TypedDict, total=False):
    log: Required[ILogger]
    prog_bar: bool
    cores: int


def postprocess_inverse_prob[F: np.floating, I: np.integer](
    pb: ProblemParameters,
    mesh: MeshInfo,
    cl: A2[F],
    cl_top: CLPartition[F, I] | None,
    dl_top: CLPartition[F, I],
    **kwargs: Unpack[_PostProcessInverseProbKwargs],
) -> list[str]:
    log = kwargs.get("log", NLOGGER)
    _bar = kwargs.get("prog_bar", True)
    _cores = kwargs.get("cores", 1)
    log.info("Post processing exported variables")
    postprocess_physical_space(
        mesh.DIR / (mesh.DISP + "_FE.X"), "Disp", home=pb.P.D, cores=_cores, prog_bar=_bar
    )
    postprocess_inverse_mechanics(("U0", "RefDisp"), root_dir=pb.P.D, log=log)
    match expand_cl_variables_to_main_topology(cl_top, cl, "0LM", "tLM", root_dir=pb.P.D):
        case Ok(cl_vars):
            pass
        case Err(e):
            log.error(f"Failed to expand CL variables: {e}")
            cl_vars = []
    log.info("Computing stiffness")
    match compute_stiffness_from_dl_field(dl_top, "DM", cl, root_dir=pb.P.D):
        case Ok(None):
            stiff = []
        case Ok(stiff):
            stiff = ["Stiff"]
        case Err(e):
            log.error(f"Failed to compute stiffness from DL field: {e}")
            stiff = []
    # pres0, prest = "P0", "Pt"
    # if M.ORDER == 2:
    #     post_process_vars_to_quad(
    #         mesh,
    #         opts.P.D,
    #         ("PL0", "P0"),
    #         ("PLt", "Pt"),
    #         cores=cores,
    #         LOG=LOG,
    #     )
    log.info("Calculating Disp Var")
    files = (f.name for f in pb.P.D.glob("Ut-*.D"))
    match get_var_index(files, "Ut"):
        case Ok(items):
            pass
        case Err(e):
            log.error(f"Failed to get variable indices for Ut: {e}")
            items = []
    for i in items:
        addwrite_var(pb.P, i, disp_i=f"Ut-{i}.D", disp_t=f"Disp-{i}.D", disp=f"RefDisp-{i}.D")
    log.info("Creating vtus")
    export_vars = ["Disp", "RefDisp", "Stiff", "CLField", "X0", "Xt", "Xi", "U0", "Ut", "CLz"]
    return [*export_vars, *cl_vars, *stiff]
