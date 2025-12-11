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
    check_for_vars,
    compute_stiffness_from_dl_field,
    make_longitudinal_field,
    postprocess_physical_space,
    run_setup,
    run_simulation,
    run_vtu,
)
from cheartpy.io.api import fix_suffix
from meshes import STRAIGHT_CYLINDER_QUAD_MESH
from pfiles.forward_centerline_constrained import create_pfile as create_forward_pfile
from problems import PROBS_FORWARD_STRAIGHT
from pytools.logging.api import BLogger
from pytools.path import clear_dir

if TYPE_CHECKING:
    from aorta_personalization.mesh.types import MeshInfo
    from aorta_personalization.problem.types import ProblemParameters


def main_forward(pb: ProblemParameters, mesh: MeshInfo) -> None:
    log = BLogger("INFO")
    clear_dir(pb.P.D)
    cl, cl_top, _ = run_setup(pb, mesh, log=log).unwrap()
    run_simulation(create_forward_pfile, pb, mesh, cl_top, log=log, pedantic=True, cores=16)
    postprocess_physical_space(mesh.DIR / (fix_suffix(mesh.DISP) + ".X"), "Disp", home=pb.P.D)
    compute_stiffness_from_dl_field(cl_top, "Stiff", cl, root_dir=pb.P.D)
    make_longitudinal_field(pb.P.D)
    export_vars = check_for_vars(pb.P.D, "Space", "Disp", "CLField", "CLz", "Stiff")
    run_vtu(mesh, pb, *export_vars, cores=16)


def main_cli() -> None:
    main_forward(PROBS_FORWARD_STRAIGHT[1], STRAIGHT_CYLINDER_QUAD_MESH)
    # for p in PROBS_STRAIGHT:
    #     main_forward(MESH_STRAIGHT_CYLINDER_QUAD, p)
    # for p in PROBS_BENT:
    #     main_forward(MESH_BENT_CYLINDER_QUAD, p)
    # for p in PROBS_BULGE:
    #     main_forward(MESH_BENT_CYLINDER_QUAD, p)


if __name__ == "__main__":
    main_cli()
