# /// script
# require-python = ">=3.14"
# dependencies = [
#     "numpy",
#     "cheartpy",
#     "aorta_personalization"
# ]
# ///

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from aorta_personalization.mesh.api import create_topology_list
from aorta_personalization.problem.api import (
    create_boundary_condition_list,
    create_centerline_topology_list,
    create_motion_variable,
    create_pres_expressions,
    create_pressure_coupling_problem,
    create_reference_space_problem,
    create_rigid_body_constraints,
    create_stiffness_expressions,
)
from aorta_personalization.solid.api import create_solid_problem, create_solid_vars, set_solid_ic
from cheartpy.cl.api import (
    create_cl_dilation_constraint_problem,
    create_cl_motion_constraint_problem,
    create_dm_on_cl,
    create_lm_on_cl,
    set_clvar_ic,
)
from cheartpy.cl.struct import CLPartition
from cheartpy.fe.api import (
    add_statevar,
    create_solver_group,
    create_solver_matrix,
    create_solver_subgroup,
    create_time_scheme,
    create_variable,
)
from cheartpy.fe.p_file import PFile
from pytools.result import Err, Ok

if TYPE_CHECKING:
    from aorta_personalization.mesh.types import MeshInfo
    from aorta_personalization.problem.types import ProblemParameters


def create_inverse_pfile[F: np.floating, I: np.integer](
    prob: ProblemParameters,
    mesh: MeshInfo,
    *part: CLPartition[F, I] | None,
) -> Ok[PFile] | Err:
    match part:
        case cl, CLPartition() as dl:
            pass
        case _:
            return Err(ValueError("Invalid CL/DL partitions"))
    Path(prob.P.D).mkdir(parents=True, exist_ok=True)
    time = create_time_scheme("time", prob.t0, prob.nt, prob.dt)
    tops, interfaces = create_topology_list(mesh)
    field = create_variable("CLField", tops.U, 2, mesh.DIR / mesh.FIELD, freq=prob.ex_freq)
    cl_top, cl_interfaces = create_centerline_topology_list(mesh, tops, cl, field)
    dl_top, dl_interfaces = create_centerline_topology_list(mesh, tops, dl, field)
    # Initial Solid variable
    svar = {
        s: create_solid_vars(tops, tops.U, freq=prob.ex_freq, pfx=("X", "U", "P"), sfx=s)
        for s in ["i", "0", "t"]
    }
    [set_solid_ic(s, root=prob.P.D) for s in svar.values()]
    # Lagrange multipliers
    lm = {s: create_lm_on_cl(cl_top, 3, freq=prob.ex_freq, sfx=f"{s}LM") for s in ["0", "t"]}
    dm = create_dm_on_cl(dl_top, dl_top.nn, freq=prob.ex_freq)
    [set_clvar_ic(v, prob.P.D / f"{v}.INIT") for v in [*lm.values(), dm]]
    stiffness = create_stiffness_expressions(dm, top=dl_top)
    # Loading and BCs
    motion_var = create_motion_variable(prob.motion_var, "CLDispt", tops, prob)
    pres = {
        s: create_pres_expressions(f"P{s}_expr", "ramp", amp=b)
        for s, b in zip(["0", "t"], [prob.target * prob.pres, -1.0 * prob.pres], strict=True)
    }
    bcs = {s: create_boundary_condition_list(mesh, svar[s], None) for s in ["0", "t"]}
    # CL motion constraints
    motion_0_prob = create_cl_motion_constraint_problem(
        cl_top, svar["i"].X, lm["0"], svar["0"].U, sfx="0CL"
    )
    motion_t_prob = create_cl_motion_constraint_problem(
        cl_top, svar["t"].X, lm["t"], svar["t"].U, svar["0"].U, motion_var, sfx="tCL"
    )
    # Dilation
    dilation_t_prob = create_cl_dilation_constraint_problem(
        dl_top, svar["t"].X, dm, svar["t"].U, svar["0"].U, motion_var, sfx="tDM"
    )
    [add_statevar(p, svar["t"].X, svar["0"].U) for p in [motion_t_prob, dilation_t_prob]]
    # Solid
    solid_prob = {
        s: create_solid_problem(
            mesh,
            svar[s],
            stiffness,
            state_vars=[dm],
            bcs=bcs[s],
            pres=pres[s] if s == "0" else None,
        )
        for s in ["0", "t"]
    }
    solid_prob["0"].set_flags("Inverse-mechanics")
    solid_prob["t"].add_state_variable(svar["t"].X)
    # Current Pressure
    space_t_prob = create_reference_space_problem(tops.U, svar["i"].X, svar["t"].X, svar["0"].U)
    pres_t_prob = create_pressure_coupling_problem(
        tops.inner, svar["t"].X, svar["t"].U, pres["t"], state_vars=[svar["t"].X]
    )
    # Rigid Body
    rigid_probs = {
        s: create_rigid_body_constraints(tops, svar["i"].X, svar[s], mesh.GEO, motion_0_prob)
        for s in ["0", "t"]
    }
    # Matrix
    solid_matrix = create_solver_matrix(
        "SolidMatrix",
        "SOLVER_MUMPS",
        *solid_prob.values(),
        space_t_prob,
        pres_t_prob,
        motion_0_prob,
        motion_t_prob,
        dilation_t_prob,
        *[v for p in rigid_probs.values() for v in p.values()],
    )
    solid_matrix.suppress_output = True
    solid_matrix.add_setting("ordering", "parallel")
    solid_matrix.add_setting("SolverMatrixCalculation", "evaluate_every_build")
    sg_solid = create_solver_subgroup("seq_fp_linesearch", solid_matrix)
    sg_solid.scale_first_residual = 100000.0
    g = create_solver_group("Main", time)
    g.export_initial_condition = False
    g.set_convergence("L2TOL", "1e-10")
    g.set_iteration("LINESEARCHITER", 2)
    g.set_iteration("SUBITERATION", 6)
    g.add_solversubgroup(sg_solid)
    pfile = PFile(h="message", output_dir=prob.P.D)
    pfile.add_interface(*interfaces, *cl_interfaces, *dl_interfaces)
    pfile.add_solvergroup(g)
    return Ok(pfile)
