from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, Unpack

from cheartpy.fe.api import create_bcpatch, create_variable
from cheartpy.fe.physics.solid_mechanics.matlaws import Matlaw
from cheartpy.fe.physics.solid_mechanics.solid_problems import (
    SolidProblem,
    create_solid_mechanics_problem,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aorta_personalization.mesh.types import MeshInfo
    from cheartpy.fe.trait import IBCPatch, IExpression, IVariable

    from ._types import SolidProbVars


class _SetSolidICKwargs(TypedDict, total=False):
    root: Path
    suffix: str


def set_solid_ic(var: SolidProbVars, **kwargs: Unpack[_SetSolidICKwargs]) -> None:
    root = kwargs.get("root", Path())
    _sfx = kwargs.get("suffix", ".INIT")
    var.X.add_data((root / f"{var.X}").with_suffix(_sfx))
    var.U.add_data((root / f"{var.U}").with_suffix(_sfx))
    var.P.add_data((root / f"{var.P}").with_suffix(_sfx))


class _SolidProblemParameters(TypedDict, total=False):
    state_vars: Sequence[IVariable]
    bcs: Sequence[IBCPatch]
    pres: IExpression | None
    fibers: IVariable | None


def create_solid_problem(
    mesh: MeshInfo, v: SolidProbVars, pars: IExpression, **kwargs: Unpack[_SolidProblemParameters]
) -> SolidProblem:
    mp = create_solid_mechanics_problem(f"Solid{v.U}", "QUASI_STATIC", v.X, v.U, pres=v.P)
    # mp.AddMatlaw(Matlaw("neohookean", [1.0]))
    mp.add_matlaw(Matlaw("isotropic-exponential", [pars]))
    # mp.AddMatlaw(Matlaw("planer-bifiber-struct3D-crimped", [pars]))
    mp.add_expr_deps(pars)
    if (pres := kwargs.get("pres")) is not None:
        mp.bc.add_patch(create_bcpatch(mesh.INNER[1], v.U, "scaled_normal", pres))
        # mp.bc.AddPatch(create_bcpatch(5, v.U, "scaled_normal", pres))
        mp.add_expr_deps(pres)
    if (fibers := kwargs.get("fibers")) is not None:
        mp.add_variable("GenStruc", fibers)
    for b in kwargs.get("bcs", []):
        mp.bc.add_patch(b)
    mp.use_option("Density", 1.0e-6)
    if v.U.order == v.P.order:
        mp.stabilize("Nearly-incompressible", 100)
    mp.add_state_variable(*kwargs.get("state_vars", []))
    stiff = create_variable(
        "Stiff",
        v.U.get_top(),
        len(pars),
        freq=v.U.get_export_frequency(),
    )
    stiff.add_setting("TEMPORAL_UPDATE_EXPR", pars)
    mp.add_var_deps(stiff)
    return mp
