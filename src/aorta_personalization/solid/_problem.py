from collections.abc import Sequence
from typing import TypedDict, Unpack

from cheartpy.fe.physics.solid_mechanics.solid_problems import SolidProblem
from cheartpy.fe.trait import IBCPatch, IExpression, IVariable

from ._types import SolidProbVars


class _SolidProblemParameters(TypedDict, total=False):
    state_vars: Sequence[IVariable]
    bcs: Sequence[IBCPatch]
    pres: IExpression | None
    fibers: IVariable | None


def create_solid_problem(
    v: SolidProbVars, pars: IExpression, **kwargs: Unpack[_SolidProblemParameters]
) -> SolidProblem: ...
