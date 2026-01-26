from typing import TYPE_CHECKING, TypedDict, Unpack

from cheartpy.fe.physics.fs_coupling import FSCouplingProblem, FSExpr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cheartpy.fe.trait import ICheartTopology, IExpression, IVariable


class _ProbKwargs(TypedDict, total=False):
    state_vars: Sequence[IVariable]


def create_reference_space_problem(
    top: ICheartTopology,
    x0: IVariable,
    xt: IVariable,
    u0: IVariable,
    **kwargs: Unpack[_ProbKwargs],
) -> FSCouplingProblem:
    fsbc = FSCouplingProblem(f"P{x0}_{xt}", x0, top)
    fsbc.perturbation = True
    fsbc.set_lagrange_mult(xt, FSExpr(x0, -1), FSExpr(u0, 1), FSExpr(xt, 1))
    fsbc.add_term(u0, FSExpr(u0, 0))
    if (state_vars := kwargs.get("state_vars")) is not None:
        fsbc.add_state_variable(*state_vars)
    return fsbc


def create_pressure_coupling_problem(
    top: ICheartTopology,
    xt: IVariable,
    ut: IVariable,
    pres: IExpression,
    **kwargs: Unpack[_ProbKwargs],
) -> FSCouplingProblem:
    fsbc = FSCouplingProblem(f"PPres{ut}", xt, top)
    fsbc.set_lagrange_mult(ut, FSExpr(pres, op="trace"))
    fsbc.perturbation = True
    if (state_vars := kwargs.get("state_vars")) is not None:
        fsbc.add_state_variable(*state_vars)
    return fsbc
