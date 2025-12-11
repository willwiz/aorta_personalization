# /// script
# require-python = ">=3.14"
# dependencies = [
#     "numpy",
#     "cheartpy",
#     "aorta_personalization"
# ]
# ///

from typing import TYPE_CHECKING

from aorta_personalization.prep.api import (
    make_longitudinal_field,
    make_reference_data_for_inverse_estimation,
    postprocess_inverse_prob,
    run_setup,
    run_simulation,
    run_vtu,
)
from meshes import STRAIGHT_CYLINDER_QUAD_MESH
from pfiles.inverse_parameter_estimation import create_inverse_pfile
from problems import PROBS_INVERSE_STRAIGHT
from pytools.logging.api import BLogger
from pytools.path import clear_dir

if TYPE_CHECKING:
    from aorta_personalization.mesh.types import MeshInfo
    from aorta_personalization.problem.types import ProblemParameters


def main_reverse(opts: ProblemParameters, mesh: MeshInfo) -> None:
    clear_dir(opts.P.D)
    log = BLogger(opts.log)
    cl, cl_top, dl_top = run_setup(opts, mesh, log=log).unwrap()
    if dl_top is None:
        msg = "DL partition is None"
        raise ValueError(msg)
    make_reference_data_for_inverse_estimation(opts, mesh, cl, cl_top, dl_top, log=log)
    run_simulation(
        create_inverse_pfile, opts, mesh, cl_top, dl_top, pedantic=True, log=log, cores=16
    )
    exported_vars = postprocess_inverse_prob(opts, mesh, cl, cl_top, dl_top, cores=16, log=log)
    make_longitudinal_field(opts.P.D)
    run_vtu(mesh, opts, *exported_vars, space=str(opts.P.D / "Xi.INIT"), cores=16)


def main_cli() -> None:
    main_reverse(PROBS_INVERSE_STRAIGHT[1], STRAIGHT_CYLINDER_QUAD_MESH)
    # for s in S_BENT_NOISE:
    #     main_reverse(MESH_BENT_CYLINDER_QUAD, s)
    # for s in S_STRAIGHT_NOISE:
    #     main_reverse(MESH_STRAIGHT_CYLINDER_QUAD, s)
    # for s in S_BULGE:
    #     main_reverse(MESH_BENT_CYLINDER_QUAD, s)


if __name__ == "__main__":
    main_cli()
