from typing import TypedDict, Unpack

from aorta_personalization.material.types import STIFF_MODES
from cheartpy.fe.trait import IExpression

from ._types import PRES_MODES


class _PressureParameters(TypedDict, total=False):
    baseline: float
    amplitude: float


def create_pres_expressions(
    prefix: str, mode: PRES_MODES, **kwargs: Unpack[_PressureParameters]
) -> IExpression: ...


class _StiffnessParameters(TypedDict, total=False):
    baseline: float
    amplitude: float


def create_stiffness_expressions(
    prefix: str, mode: STIFF_MODES, **kwargs: Unpack[_StiffnessParameters]
) -> IExpression: ...
