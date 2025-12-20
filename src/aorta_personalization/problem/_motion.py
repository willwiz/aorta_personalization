from typing import TYPE_CHECKING, TypedDict, Unpack, overload

from cheartpy.fe.api import create_expr, create_variable
from pytools.result import Err, Ok

if TYPE_CHECKING:
    from aorta_personalization.mesh.types import ProblemTopologies
    from cheartpy.fe.trait import ICheartTopology, IVariable

    from ._types import MOTION_VAR, ProblemParameters


class _MotionVarKwargs(TypedDict, total=False):
    step: int
    amplitude: float


def _update_motionvar_auto(
    var: IVariable, top: ICheartTopology, pb: ProblemParameters
) -> IVariable:
    motion_data = create_variable(f"{var}Data", top, 3, (pb.P.D / f"{var}.INIT"), freq=pb.ex_freq)
    motion_expr = create_expr(
        f"{var}_expr",
        [f"{motion_data}.{i} * min(t/{pb.nt * pb.dt}, 1.0)" for i in [1, 2, 3]],
    )
    motion_expr.add_deps(motion_data)
    var.add_setting("TEMPORAL_UPDATE_EXPR", motion_expr)
    return var


def _update_motionvar_step(
    var: IVariable, top: ICheartTopology, pb: ProblemParameters, step: int
) -> IVariable:
    motion_data = create_variable(
        f"{var}Data", top, 3, (pb.P.D / f"Disp-{step}.D"), freq=pb.ex_freq
    )
    motion_expr = create_expr(
        f"{var}_expr",
        [f"{motion_data}.{i} * min(t/{pb.nt * pb.dt}, 1.0)" for i in [1, 2, 3]],
    )
    motion_expr.add_deps(motion_data)
    var.add_setting("TEMPORAL_UPDATE_EXPR", motion_expr)
    return var


@overload
def create_motion_variable(
    motion: None,
    prefix: str,
    top: ProblemTopologies,
    prob: ProblemParameters,
    **kwargs: Unpack[_MotionVarKwargs],
) -> Ok[None]: ...
@overload
def create_motion_variable(
    motion: MOTION_VAR,
    prefix: str,
    top: ProblemTopologies,
    prob: ProblemParameters,
    **kwargs: Unpack[_MotionVarKwargs],
) -> Ok[IVariable] | Err: ...
def create_motion_variable(
    motion: MOTION_VAR | None,
    prefix: str,
    top: ProblemTopologies,
    prob: ProblemParameters,
    **kwargs: Unpack[_MotionVarKwargs],
) -> Ok[IVariable] | Ok[None] | Err:
    if motion is None:
        return Ok(None)
    var = create_variable(prefix, top.U, 3, freq=prob.ex_freq)
    match motion:
        case "Zeros":
            zeros_3_expr = create_expr("zeros_3_expr", [0, 0, 0])
            var.add_setting("INIT_EXPR", zeros_3_expr)
        case "AUTO":
            var = _update_motionvar_auto(var, top.U, prob)
        case "DISP":
            if prob.track is None:
                msg = "ProblemParameters.track must be set for motion_var='DISP'"
                return Err(ValueError(msg))
            var.add_setting("TEMPORAL_UPDATE_FILE", prob.track / "Disp*")
        case "VAR":
            if prob.track is None:
                msg = "ProblemParameters.track must be set for motion_var='VAR'"
                return Err(ValueError(msg))
            var.add_setting("TEMPORAL_UPDATE_FILE", prob.track / f"{var}*")
        case "STEP":
            step = kwargs.get("step", prob.nt)
            var = _update_motionvar_step(var, top.U, prob, step)
    return Ok(var)
