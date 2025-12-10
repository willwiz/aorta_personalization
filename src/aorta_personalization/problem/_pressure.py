from typing import TYPE_CHECKING, Required, TypedDict, Unpack

import numpy as np
from cheartpy.fe.api import create_expr

if TYPE_CHECKING:
    from cheartpy.fe.trait import IExpression

    from ._types import PRES_MODES


class _PressureParameters(TypedDict, total=False):
    amp: Required[float]
    rate: float


def create_pres_expressions(
    prefix: str, mode: PRES_MODES, **kwargs: Unpack[_PressureParameters]
) -> IExpression:
    amplitude = kwargs["amp"]
    rate = kwargs.get("rate", 1.0)
    match mode:
        case "ramp":
            return create_expr(prefix, [f"{amplitude}*min({rate}*t, 1)"])
        case "sin":
            return create_expr(prefix, [f"{amplitude}*sin({np.pi}*t)*sin({np.pi}*t)"])
