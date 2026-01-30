# /// script
# require-python = ">=3.14"
# dependencies = [
#     "numpy",
#     "cheartpy",
#     "aorta_personalization"
# ]
# ///
from typing import TYPE_CHECKING

from aorta_personalization.mesh.api import create_topology_list
from aorta_personalization.problem.api import (
    create_boundary_condition_list,
    create_centerline_topology_list,
    create_motion_variable,
    create_pres_expressions,
    create_rigid_body_constraints,
    create_stiffness_expressions,
)
from aorta_personalization.solid.api import create_solid_problem, create_solid_vars
from cheartpy.cl.api import create_cl_motion_constraint_problem, create_lm_on_cl
from cheartpy.fe.api import (
    create_solver_group,
    create_solver_matrix,
    create_solver_subgroup,
    create_time_scheme,
    create_variable,
)
from cheartpy.fe.p_file import PFile
from cheartpy.fe.physics.l2_projection import L2SolidProjection
from pytools.result import Err, Ok

if TYPE_CHECKING:
    import numpy as np
    from aorta_personalization.mesh.types import MeshInfo
    from aorta_personalization.problem.types import ProblemParameters
    from cheartpy.cl.struct import CLPartition


def create_pfile[F: np.floating, I: np.integer](
    prob: ProblemParameters,
    mesh: MeshInfo,
    *part: CLPartition[F, I] | None,
) -> Ok[PFile] | Err:
    match part:
        case ():
            cl = None
        case (cl,):
            pass
        case _:
            msg = "Must provide zero or one CL partition only"
            return Err(ValueError(msg))
    prob.P.D.mkdir(parents=True, exist_ok=True)
    time = create_time_scheme("time", prob.t0, prob.nt, prob.dt)
    tops, interfaces = create_topology_list(mesh)
    field = create_variable("CLField", tops.U, 2, mesh.DIR / mesh.FIELD, freq=prob.ex_freq)
    cl_top, cl_interfaces = create_centerline_topology_list(mesh, tops, cl, field).unwrap()
    svars = create_solid_vars(tops, tops.U, freq=prob.ex_freq)
    lm_cl = create_lm_on_cl(cl_top, 3, freq=prob.ex_freq)
    motion_var = create_motion_variable(prob.motion_var, "CLDisp", tops, prob).unwrap()
    pres_expr = create_pres_expressions("loading_pres_expr", "ramp", amp=prob.pres)
    stiff_expr = create_stiffness_expressions(prob.matpars, field=field).unwrap()
    bcs_list = create_boundary_condition_list(mesh, svars, motion_var)
    motion_prob = create_cl_motion_constraint_problem(cl_top, svars.X, lm_cl, svars.U, motion_var)
    solid_prob = create_solid_problem(mesh, svars, stiff_expr, bcs=bcs_list, pres=pres_expr)
    rigid_prob = create_rigid_body_constraints(tops, svars.X, svars, mesh.GEO, motion=motion_prob)
    solid_matrix = create_solver_matrix(
        "SolidMatrix", "SOLVER_MUMPS", solid_prob, motion_prob, *rigid_prob.values()
    )
    solid_matrix.add_setting("ordering", "parallel")
    solid_matrix.add_setting("SolverMatrixCalculation", "evaluate_every_build")
    strain_var = create_variable("Strain", tops.P, 9, freq=prob.ex_freq)
    strain_prob = L2SolidProjection(
        "strain_prob", svars.X, strain_var, solid_prob, "deformation_gradient"
    )
    strain_matrix = create_solver_matrix("StrainMatrix", "SOLVER_MUMPS", strain_prob)
    sg_solid = create_solver_subgroup("seq_fp_linesearch", solid_matrix)
    sg_strain = create_solver_subgroup("seq_fp", strain_matrix)
    g = create_solver_group("Main", time)
    g.set_convergence("L2TOL", 1e-11)
    g.set_iteration("LINESEARCHITER", 8)
    g.set_iteration("SUBITERATION", 5)
    g.add_solversubgroup(sg_solid, sg_strain)
    pfile = PFile(h="Forward CL constrained simulation", output_dir=prob.P.D)
    pfile.add_interface(*interfaces, *cl_interfaces)
    pfile.add_solvergroup(g)
    return Ok(pfile)
