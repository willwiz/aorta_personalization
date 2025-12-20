from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict, Unpack, overload

import numpy as np
from cheartpy.fe.api import create_expr
from cheartpy.fe.trait import IVariable
from pytools.result import Err, Ok

from ._types import ClNodalLmType, MaterialProperty

if TYPE_CHECKING:
    from cheartpy.cl.struct import CLStructure
    from cheartpy.fe.trait import IExpression


class _StiffnessExpressionKwargs(TypedDict, total=False):
    prefix: str
    m_prefix: str
    exponent: float


def _create_single_variable_stiffness_expr(
    dm: IVariable, dl: CLStructure, **kwargs: Unpack[_StiffnessExpressionKwargs]
) -> IExpression:
    prefix = kwargs.get("prefix", "stiff_expr")
    m_prefix = kwargs.get("m_prefix", "LE_expr")
    e = kwargs.get("exponent", 0.5)
    multiplier = create_expr(
        m_prefix,
        ["+".join([f"{dm}.{i}*{dl.b_vec}.{i}" for i in range(1, dl.nn + 1)])],
    )
    multiplier.add_deps(dm, dl.b_vec)
    modulus = create_expr(prefix, [f"10.0 * (1 + {multiplier})", f"{e}"])
    modulus.add_deps(multiplier)
    return modulus


def _create_multi_variable_stiffness_expr(
    dms: ClNodalLmType, dl: CLStructure, **kwargs: Unpack[_StiffnessExpressionKwargs]
) -> IExpression:
    prefix = kwargs.get("prefix", "stiff_expr")
    m_prefix = kwargs.get("m_prefix", "LE_expr")
    e = kwargs.get("exponent", 0.5)
    multiplier = create_expr(
        m_prefix, ["+".join([f"({dms[k]}*{dl.b_vec}.{k + 1})" for k in range(dl.nn)])]
    )
    multiplier.add_deps(dl.b_vec, *dms.values())
    modulus = create_expr(prefix, [f"10.0 * (1 + {multiplier})", f"{e}"])
    modulus.add_deps(multiplier)
    return modulus


def create_variable_stiffness_expr(
    dms: ClNodalLmType | IVariable, dl: CLStructure, **kwargs: Unpack[_StiffnessExpressionKwargs]
) -> IExpression:
    match dms:
        case IVariable():
            return _create_single_variable_stiffness_expr(dms, dl, **kwargs)
        case Mapping():
            return _create_multi_variable_stiffness_expr(dms, dl, **kwargs)


def create_constant_stiffness_expr(
    pars: MaterialProperty, _field: IVariable, **kwargs: Unpack[_StiffnessExpressionKwargs]
) -> IExpression:
    b = pars.baseline
    e = kwargs.get("exponent", 0.5)
    return create_expr("stiff_expr", [f"{b}", f"{e}"])


def create_grad_stiffness_expr(
    pars: MaterialProperty, field: IVariable, **kwargs: Unpack[_StiffnessExpressionKwargs]
) -> IExpression:
    a, b = pars.amplitude, pars.baseline
    e = kwargs.get("exponent", 0.5)
    stiff_expr = create_expr("stiff_expr", [f"{b} + {a} * exp(-2*{field}.1)", f"{e}"])
    stiff_expr.add_deps(field)
    return stiff_expr


def create_sine_stiffness_expr(
    pars: MaterialProperty, field: IVariable, **kwargs: Unpack[_StiffnessExpressionKwargs]
) -> IExpression:
    a, b = pars.amplitude, pars.baseline
    e = kwargs.get("exponent", 0.5)
    stiff_expr = create_expr(
        "stiff_expr", [f"{b} + {a}*cos({np.pi}*{field}.1)*cos({np.pi}*{field}.1)", f"{e}"]
    )
    stiff_expr.add_deps(field)
    return stiff_expr


def create_circ_stiffness_expr(
    pars: MaterialProperty, field: IVariable, **kwargs: Unpack[_StiffnessExpressionKwargs]
) -> IExpression:
    a, b = pars.amplitude, pars.baseline
    e = kwargs.get("exponent", 0.5)
    stiff_expr = create_expr(
        "stiff_expr",
        [f"{b} + {a}*(exp(-3*(1-{field}.1))*cos({2 * np.pi} * {field}.2))", f"{e}"],
    )
    stiff_expr.add_deps(field)
    return stiff_expr


def create_material_stiffness_expr(
    pars: MaterialProperty, field: IVariable, **kwargs: Unpack[_StiffnessExpressionKwargs]
) -> IExpression:
    match pars.form:
        case "const":
            return create_constant_stiffness_expr(pars, field, **kwargs)
        case "grad":
            return create_grad_stiffness_expr(pars, field, **kwargs)
        case "sine":
            return create_sine_stiffness_expr(pars, field, **kwargs)
        case "circ":
            return create_circ_stiffness_expr(pars, field, **kwargs)


@overload
def create_stiffness_expressions(
    mode: ClNodalLmType | IVariable,
    *,
    top: CLStructure,
    **kwargs: Unpack[_StiffnessExpressionKwargs],
) -> Ok[IExpression] | Err: ...
@overload
def create_stiffness_expressions(
    mode: MaterialProperty,
    *,
    field: IVariable,
    **kwargs: Unpack[_StiffnessExpressionKwargs],
) -> Ok[IExpression] | Err: ...
def create_stiffness_expressions(
    mode: ClNodalLmType | IVariable | MaterialProperty,
    *,
    field: IVariable | None = None,
    top: CLStructure | None = None,
    **kwargs: Unpack[_StiffnessExpressionKwargs],
) -> Ok[IExpression] | Err:
    match mode:
        case IVariable():
            if top is None:
                msg = "topology must be provided when mode is IVariable"
                return Err(ValueError(msg))
            val = _create_single_variable_stiffness_expr(mode, top, **kwargs)
        case Mapping():
            if top is None:
                msg = "topology must be provided when mode is ClNodalLmType"
                return Err(ValueError(msg))
            val = _create_multi_variable_stiffness_expr(mode, top, **kwargs)
        case MaterialProperty():
            if field is None:
                msg = "field must be provided when mode is MaterialProperty"
                return Err(ValueError(msg))
            val = create_material_stiffness_expr(mode, field, **kwargs)
    return Ok(val)
