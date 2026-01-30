# /// script
# require-python = ">=3.14"
# dependencies = [
#     "numpy",
#     "cheartpy",
#     "aorta_personalization"
# ]
# ///

from typing import TYPE_CHECKING, TypedDict, Unpack

from aorta_personalization.prep._cl_variables import expand_cl_variables_to_main_topology
from aorta_personalization.prep.api import (
    check_for_vars,
    make_longitudinal_field,
    postprocess_physical_space,
    run_setup,
    run_simulation,
    run_vtu,
)
from cheartpy.io.api import fix_ch_sfx
from meshes import BENT_CYLINDER_QUAD_MESH, BULGE_CYLINDER_QUAD_MESH, STRAIGHT_CYLINDER_QUAD_MESH
from pfiles.forward_centerline_constrained import create_pfile as create_forward_pfile
from problems import PROBS_FORWARD_BENT, PROBS_FORWARD_BULGE, PROBS_FORWARD_STRAIGHT
from pytools.logging import LogLevel, get_logger
from pytools.path import clear_dir
from pytools.result import Err, Ok

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aorta_personalization.mesh.types import MeshInfo
    from aorta_personalization.problem.types import ProblemParameters


class MainSimKwargs(TypedDict, total=False):
    cores: int
    log: LogLevel
    prog_bar: bool
    overwrite: bool


_SIMULATION_OUTPUTS = [
    "Disp",
    "Pres",
    "CLLM",
]

_POSTPROCESSING_OUTPUTS = [
    "LM",
]


def is_completed(pb: ProblemParameters, vs: Sequence[str]) -> bool:
    return all((pb.P.D / f"{v}-{pb.nt}.D").exists() for v in vs)


def main_forward(pb: ProblemParameters, mesh: MeshInfo, **kwargs: Unpack[MainSimKwargs]) -> None:
    _cores = kwargs.get("cores", 16)
    _bar = kwargs.get("prog_bar", True)
    _overwrite = kwargs.get("overwrite", False)
    log = get_logger(level=kwargs.get("log", "INFO"))
    log.brief(f"Starting forward simulation for problem at {pb.P.D}")
    pb.P.D.mkdir(parents=True, exist_ok=True)
    cl, cl_top, _ = run_setup(pb, mesh, log=log).unwrap()
    if not _overwrite and is_completed(pb, _SIMULATION_OUTPUTS):
        log.info(f"Output directory {pb.P.D} is not empty, skipping simulation.")
    else:
        log.info(
            f"Running forward simulation for problem with output dir:"
            f"{pb.P.D}={any(pb.P.D.iterdir())}."
        )
        clear_dir(pb.P.D)
        run_simulation(create_forward_pfile, pb, mesh, cl_top, log=log, pedantic=True, cores=_cores)
    if is_completed(pb, _POSTPROCESSING_OUTPUTS):
        log.info(f"Postprocessing already completed for problem with output dir: {pb.P.D}.")
        return
    postprocess_physical_space(
        mesh.DIR / (fix_ch_sfx(mesh.DISP) + "X"), "Disp", home=pb.P.D, cores=_cores, prog_bar=_bar
    ).unwrap()
    match expand_cl_variables_to_main_topology(cl_top, cl, "LM", root_dir=pb.P.D):
        case Ok(cl_vars):
            cl_vars = ["CLz", *cl_vars]
        case Err(e):
            log.error(f"Failed to expand CL variables: {e}")
            cl_vars: list[str] = []
    if cl_top is not None:
        make_longitudinal_field(pb.P.D, cores=_cores, prog_bar=_bar).unwrap()
    export_vars = check_for_vars(pb.P.D, "Space", "Disp", "CLField", "Stiff", *cl_vars)
    run_vtu(mesh, pb, *export_vars, cores=_cores)


def main_cli(**kwargs: Unpack[MainSimKwargs]) -> None:
    for p in (i for vec in PROBS_FORWARD_STRAIGHT.values() for i in vec):
        main_forward(p, STRAIGHT_CYLINDER_QUAD_MESH, **kwargs)
    for p in (i for vec in PROBS_FORWARD_BENT.values() for i in vec):
        main_forward(p, BENT_CYLINDER_QUAD_MESH, **kwargs)
    for p in (i for vec in PROBS_FORWARD_BULGE.values() for i in vec):
        main_forward(p, BULGE_CYLINDER_QUAD_MESH, **kwargs)


def main_selection(**kwargs: Unpack[MainSimKwargs]) -> None:
    # for p in (i for vec in PROBS_FORWARD_STRAIGHT.values() for i in vec):
    #     main_forward(p, STRAIGHT_CYLINDER_QUAD_MESH, **kwargs)
    for p in (i for vec in PROBS_FORWARD_BENT.values() for i in vec):
        main_forward(p, BENT_CYLINDER_QUAD_MESH, **kwargs)
    # for p in (i for vec in PROBS_FORWARD_BULGE.values() for i in vec):
    #     main_forward(p, BULGE_CYLINDER_QUAD_MESH, **kwargs)


def main_test(**kwargs: Unpack[MainSimKwargs]) -> None:
    main_forward(PROBS_FORWARD_STRAIGHT["const"][0], STRAIGHT_CYLINDER_QUAD_MESH, **kwargs)


if __name__ == "__main__":
    # main_cli(cores=16, prog_bar=True, overwrite=True)
    main_selection(cores=16, prog_bar=True, overwrite=True)
    # main_test(cores=16, prog_bar=True, overwrite=True)
