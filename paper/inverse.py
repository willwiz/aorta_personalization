# /// script
# require-python = ">=3.14"
# dependencies = [
#     "numpy",
#     "cheartpy",
#     "aorta_personalization"
# ]
# ///

from typing import TYPE_CHECKING, Unpack

from aorta_personalization.prep.api import (
    make_longitudinal_field,
    make_reference_data_for_inverse_estimation,
    postprocess_inverse_prob,
    run_setup,
    run_simulation,
    run_vtu,
)
from meshes import BENT_CYLINDER_QUAD_MESH, BULGE_CYLINDER_QUAD_MESH, STRAIGHT_CYLINDER_QUAD_MESH
from pfiles.inverse_parameter_estimation import create_inverse_pfile
from problems import (
    PROBS_INVERSE_BENT,
    PROBS_INVERSE_BULGE,
    PROBS_INVERSE_STRAIGHT,
    PROBS_NOISE_BENT,
)
from pytools.logging import get_logger
from pytools.path import clear_dir

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aorta_personalization.mesh.types import MeshInfo
    from aorta_personalization.problem.types import ProblemParameters
    from forward import MainSimKwargs


_SIMULATION_OUTPUTS = [
    "CL0LM",
    "CLDispt",
    "CLDisptData",
    "CLField",
    "CLtLM",
    "DLDM",
    "P0",
    "Pt",
    "Stiff",
    "U0",
    "Ut",
    "X0",
    "Xi",
    "Xt",
]

_POSTPROCESSING_OUTPUTS = [
    "0LM",
    "tLM",
    "CLz",
    "Disp",
    "RefDisp",
]


def is_completed(pb: ProblemParameters, vs: Sequence[str]) -> bool:
    return all((pb.P.D / f"{v}-{pb.nt}.D").exists() for v in vs)


def main_reverse(pb: ProblemParameters, mesh: MeshInfo, **kwargs: Unpack[MainSimKwargs]) -> None:
    _cores = kwargs.get("cores", 32)
    _bar = kwargs.get("prog_bar", True)
    log = get_logger(level=kwargs.get("log", "INFO"))
    log.brief(f"Starting inverse simulation for problem at {pb.P.D}")
    pb.P.D.mkdir(parents=True, exist_ok=True)
    cl, cl_top, dl_top = run_setup(pb, mesh, log=log).unwrap()
    if dl_top is None:
        log.error("DL partition is None")
        raise SystemExit(1)
    if is_completed(pb, _SIMULATION_OUTPUTS) and not kwargs.get("overwrite", False):
        log.info(f"{pb.P.D} is complete, skipping simulation.")
    else:
        clear_dir(pb.P.D)
        make_reference_data_for_inverse_estimation(pb, mesh, cl, cl_top, dl_top, log=log)
        run_simulation(
            create_inverse_pfile, pb, mesh, cl_top, dl_top, pedantic=True, log=log, cores=_cores
        )
    if not is_completed(pb, _SIMULATION_OUTPUTS):
        log.error("Simulation did not complete successfully.")
        return
    if is_completed(pb, _POSTPROCESSING_OUTPUTS) and not kwargs.get("overwrite", False):
        log.info(f"{pb.P.D} post-processing is complete, skipping.")
        # return
    exported_vars = postprocess_inverse_prob(
        pb, mesh, cl, cl_top, dl_top, log=log, cores=_cores, prog_bar=_bar
    )
    make_longitudinal_field(pb.P.D, cores=_cores, prog_bar=_bar)
    run_vtu(mesh, pb, *exported_vars, space=str(pb.P.D / "Xi.INIT"), cores=_cores)


def main_cli(**kwargs: Unpack[MainSimKwargs]) -> None:
    for p in [p for ps in PROBS_INVERSE_STRAIGHT.values() for p in ps]:
        main_reverse(p, STRAIGHT_CYLINDER_QUAD_MESH, **kwargs)
    for p in [p for ps in PROBS_INVERSE_BENT.values() for p in ps]:
        main_reverse(p, BENT_CYLINDER_QUAD_MESH, **kwargs)
    for p in [p for ps in PROBS_INVERSE_BULGE.values() for p in ps]:
        main_reverse(p, BULGE_CYLINDER_QUAD_MESH, **kwargs)


def main_selection(**kwargs: Unpack[MainSimKwargs]) -> None:
    for p in (i for vec in PROBS_INVERSE_BENT.values() for i in vec):
        main_reverse(p, BENT_CYLINDER_QUAD_MESH, **kwargs)
    # for p in PROBS_INVERSE_BENT["grad"]:
    #     main_reverse(p, BENT_CYLINDER_QUAD_MESH, **kwargs)
    # for p in PROBS_INVERSE_BULGE["sine"]:
    #     main_reverse(p, BULGE_CYLINDER_QUAD_MESH, **kwargs)
    # for p in (i for vec in PROBS_INVERSE_BULGE.values() for i in vec):
    #     main_reverse(p, BULGE_CYLINDER_QUAD_MESH, **kwargs)


def main_test(**kwargs: Unpack[MainSimKwargs]) -> None:
    main_reverse(PROBS_NOISE_BENT[0], BENT_CYLINDER_QUAD_MESH, **kwargs)


if __name__ == "__main__":
    # main_cli(cores=16, prog_bar=True)
    main_selection(cores=16, prog_bar=True, overwrite=True)
    # main_test(cores=16, prog_bar=True)
