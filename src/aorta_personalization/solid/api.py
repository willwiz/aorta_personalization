from typing import TYPE_CHECKING, TypedDict, Unpack

from ._problem import create_solid_problem

if TYPE_CHECKING:
    from aorta_personalization.topology._types import ProblemTopologies
    from cheartpy.fe.trait.basic import ICheartTopology
    from pytools.arrays import T3

    from ._types import SolidProbVars

__all__ = ["create_solid_problem", "create_solid_vars"]


class _CreateSolidVarsKwargs(TypedDict, total=False):
    ex_freq: int
    pfx: T3[str]
    sfx: str


def create_solid_vars(
    top: ProblemTopologies, space_top: ICheartTopology, **kwargs: Unpack[_CreateSolidVarsKwargs]
) -> SolidProbVars: ...
