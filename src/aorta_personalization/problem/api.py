from typing import TypedDict, Unpack

from aorta_personalization.problem._types import ProblemParameters
from aorta_personalization.topology.types import ProblemTopologies
from cheartpy.fe.trait import IVariable

from ._bcs import get_boundary_conditions_list
from ._expressions import create_pres_expressions, create_stiffness_expressions

__all__ = [
    "create_motion_variable",
    "create_pres_expressions",
    "create_stiffness_expressions",
    "get_boundary_conditions_list",
]


class _MotionVarKwargs(TypedDict, total=False):
    step: int | None
    amplitude: float


def create_motion_variable(
    prefix: str, top: ProblemTopologies, prob: ProblemParameters, **kwargs: Unpack[_MotionVarKwargs]
) -> IVariable | None: ...
